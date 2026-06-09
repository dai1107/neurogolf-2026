"""Rewrite sparse one-hot shift Conv kernels into Pad/Slice/Concat graphs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .cost_estimator import estimate_model_cost


FIELDS = [
    "task_id",
    "source_model_path",
    "output_model_path",
    "weight_name",
    "rewritten_conv_nodes",
    "removed_initializer_elements",
    "added_initializer_elements",
    "source_cost",
    "output_cost",
    "cost_delta",
    "source_file_size_bytes",
    "output_file_size_bytes",
    "file_size_delta",
]


def _initializer_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }


def _value_shapes(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = onnx.shape_inference.infer_shapes(model)
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    shapes: dict[str, list[int]] = {}
    for value in values:
        tensor_type = value.type.tensor_type
        if not value.type.HasField("tensor_type") or not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                break
            dims.append(int(dim.dim_value))
        else:
            shapes[value.name] = dims
    return shapes


def _int_attr(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _ints_attr(node: onnx.NodeProto, name: str, default: list[int]) -> list[int]:
    for attr in node.attribute:
        if attr.name == name:
            return [int(value) for value in attr.ints]
    return list(default)


def _validate_sparse_shift_weight(weight: np.ndarray, weight_name: str) -> list[tuple[int, int]]:
    if weight.ndim != 4:
        raise ValueError(f"{weight_name} must be a 4D Conv weight, got shape {weight.shape}")
    if weight.shape[1] != 1:
        raise ValueError(f"{weight_name} must have one input channel, got shape {weight.shape}")
    if weight.shape[2] % 2 != 1 or weight.shape[3] % 2 != 1:
        raise ValueError(f"{weight_name} kernel height/width must be odd, got shape {weight.shape}")

    locations: list[tuple[int, int]] = []
    for channel in range(weight.shape[0]):
        kernel = weight[channel, 0]
        nonzero = np.argwhere(np.abs(kernel) > 1e-6)
        if nonzero.shape != (1, 2):
            raise ValueError(f"{weight_name}[{channel}] must contain exactly one nonzero value")
        row, col = (int(nonzero[0, 0]), int(nonzero[0, 1]))
        value = float(kernel[row, col])
        if abs(value - 1.0) > 1e-6:
            raise ValueError(f"{weight_name}[{channel}] nonzero value must be 1.0, got {value}")
        locations.append((row, col))
    return locations


def _validate_conv_node(node: onnx.NodeProto, weight: np.ndarray, output_shape: list[int]) -> None:
    if len(node.input) != 2:
        raise ValueError(f"{node.name or node.output[0]} must be a bias-free Conv")
    if _int_attr(node, "group", 1) != 1:
        raise ValueError(f"{node.name or node.output[0]} group must be 1")
    if _ints_attr(node, "strides", [1, 1]) != [1, 1]:
        raise ValueError(f"{node.name or node.output[0]} strides must be [1, 1]")
    if _ints_attr(node, "dilations", [1, 1]) != [1, 1]:
        raise ValueError(f"{node.name or node.output[0]} dilations must be [1, 1]")

    kernel_height = int(weight.shape[2])
    kernel_width = int(weight.shape[3])
    expected_pads = [
        kernel_height // 2,
        kernel_width // 2,
        kernel_height // 2,
        kernel_width // 2,
    ]
    pads = _ints_attr(node, "pads", [0, 0, 0, 0])
    if pads != expected_pads:
        raise ValueError(f"{node.name or node.output[0]} pads must be {expected_pads}, got {pads}")
    if output_shape[:2] != [1, int(weight.shape[0])]:
        raise ValueError(
            f"{node.name or node.output[0]} output shape must start with "
            f"[1, {weight.shape[0]}], got {output_shape}"
        )


def _make_i64(values: list[int], name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def rewrite_sparse_shift_convs(
    input_model: str,
    output_model: str,
    weight_name: str = "wk",
) -> dict[str, Any]:
    """Replace sparse one-hot same-padded Conv nodes that use ``weight_name``.

    The rewrite is graph-equivalent for the matched Conv form:
    each output channel contains exactly one unit kernel tap, with no bias,
    stride 1, dilation 1, group 1, and symmetric same padding.
    """
    input_path = Path(input_model)
    output_path = Path(output_model)
    if not input_path.is_file():
        raise FileNotFoundError(f"input model does not exist: {input_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    onnx.checker.check_model(model)
    source_cost = estimate_model_cost(str(input_path))

    arrays = _initializer_arrays(model)
    if weight_name not in arrays:
        raise ValueError(f"model is missing initializer {weight_name}")
    weight = arrays[weight_name]
    locations = _validate_sparse_shift_weight(weight, weight_name)
    conv_nodes = [node for node in model.graph.node if node.op_type == "Conv" and len(node.input) >= 2 and node.input[1] == weight_name]
    if not conv_nodes:
        raise ValueError(f"model has no Conv nodes using {weight_name}")

    shapes = _value_shapes(model)
    added_initializers: list[onnx.TensorProto] = []
    axes_name = f"{weight_name}_shift_axes_hw"
    pads_name = f"{weight_name}_shift_pads"
    added_initializers.append(_make_i64([2, 3], axes_name))
    added_initializers.append(
        _make_i64(
            [
                0,
                0,
                int(weight.shape[2]) // 2,
                int(weight.shape[3]) // 2,
                0,
                0,
                int(weight.shape[2]) // 2,
                int(weight.shape[3]) // 2,
            ],
            pads_name,
        )
    )
    for channel, (row, col) in enumerate(locations):
        added_initializers.append(_make_i64([row, col], f"{weight_name}_shift_starts_{channel:03d}"))

    conv_outputs = {node.output[0] for node in conv_nodes}
    rewritten = 0
    new_graph_nodes: list[onnx.NodeProto] = []
    end_initializer_names: dict[tuple[int, int, int], str] = {}
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2 or node.input[1] != weight_name:
            new_graph_nodes.append(node)
            continue

        output_name = node.output[0]
        output_shape = shapes.get(output_name)
        if output_shape is None or len(output_shape) != 4:
            raise ValueError(f"missing static output shape for {output_name}")
        _validate_conv_node(node, weight, output_shape)
        out_height = int(output_shape[2])
        out_width = int(output_shape[3])

        pad_output = f"{output_name}_{weight_name}_shift_pad"
        new_graph_nodes.append(
            helper.make_node(
                "Pad",
                [node.input[0], pads_name],
                [pad_output],
                name=f"{node.name or output_name}_ShiftPad",
            )
        )

        slice_outputs: list[str] = []
        for channel, (row, col) in enumerate(locations):
            # Slice end coordinates include the inferred Conv output extent.
            end_key = (out_height, out_width, channel)
            end_name = end_initializer_names.get(end_key)
            if end_name is None:
                end_name = f"{weight_name}_shift_ends_h{out_height}_w{out_width}_{channel:03d}"
                added_initializers.append(_make_i64([row + out_height, col + out_width], end_name))
                end_initializer_names[end_key] = end_name
            slice_output = f"{output_name}_{weight_name}_shift_slice_{channel:03d}"
            slice_outputs.append(slice_output)
            new_graph_nodes.append(
                helper.make_node(
                    "Slice",
                    [
                        pad_output,
                        f"{weight_name}_shift_starts_{channel:03d}",
                        end_name,
                        axes_name,
                    ],
                    [slice_output],
                    name=f"{node.name or output_name}_ShiftSlice_{channel:03d}",
                )
            )

        new_graph_nodes.append(
            helper.make_node(
                "Concat",
                slice_outputs,
                [output_name],
                name=f"{node.name or output_name}_ShiftConcat",
                axis=1,
            )
        )
        rewritten += 1

    if rewritten != len(conv_nodes):
        raise ValueError(f"expected to rewrite {len(conv_nodes)} Conv nodes, rewrote {rewritten}")

    del model.graph.node[:]
    model.graph.node.extend(new_graph_nodes)
    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name != weight_name and initializer.name not in conv_outputs
    ]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    model.graph.initializer.extend(added_initializers)

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path))
    output_cost = estimate_model_cost(str(output_path))
    added_elements = sum(int(np.prod(initializer.dims)) for initializer in added_initializers)
    removed_elements = int(weight.size)
    return {
        "source_model_path": str(input_path),
        "output_model_path": str(output_path),
        "weight_name": weight_name,
        "rewritten_conv_nodes": rewritten,
        "removed_initializer_elements": removed_elements,
        "added_initializer_elements": added_elements,
        "source_cost": int(source_cost["estimated_cost"]),
        "output_cost": int(output_cost["estimated_cost"]),
        "cost_delta": int(output_cost["estimated_cost"]) - int(source_cost["estimated_cost"]),
        "source_file_size_bytes": int(source_cost["file_size_bytes"]),
        "output_file_size_bytes": int(output_cost["file_size_bytes"]),
        "file_size_delta": int(output_cost["file_size_bytes"]) - int(source_cost["file_size_bytes"]),
    }


def rewrite_task_models(
    model_dir: str,
    output_dir: str,
    report_path: str,
    task_ids: list[str],
    weight_name: str = "wk",
) -> dict[str, Any]:
    """Rewrite selected task models and write a CSV report."""
    model_root = Path(model_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        source = model_root / f"{task_id}.onnx"
        destination = output_root / f"{task_id}_SparseShiftConvRewrite.onnx"
        try:
            row = rewrite_sparse_shift_convs(str(source), str(destination), weight_name=weight_name)
        except Exception as exc:
            source_cost = estimate_model_cost(str(source)) if source.is_file() else {}
            row = {
                "source_model_path": str(source),
                "output_model_path": "",
                "weight_name": weight_name,
                "rewritten_conv_nodes": 0,
                "removed_initializer_elements": 0,
                "added_initializer_elements": 0,
                "source_cost": source_cost.get("estimated_cost", ""),
                "output_cost": source_cost.get("estimated_cost", ""),
                "cost_delta": 0,
                "source_file_size_bytes": source_cost.get("file_size_bytes", ""),
                "output_file_size_bytes": source_cost.get("file_size_bytes", ""),
                "file_size_delta": 0,
                "failure_reason": str(exc),
            }
        rows.append({"task_id": task_id, **row})

    fieldnames = list(FIELDS)
    if any("failure_reason" in row for row in rows):
        fieldnames.append("failure_reason")
    with Path(report_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "task_ids": task_ids,
        "report_path": report_path,
        "output_dir": output_dir,
        "improvement_count": sum(1 for row in rows if int(row.get("cost_delta") or 0) < 0),
        "total_cost_delta": sum(int(row.get("cost_delta") or 0) for row in rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_task_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/onnx")
    parser.add_argument("--output-dir", default="outputs/candidates/sparse_shift_conv_rewrite")
    parser.add_argument("--report", default="outputs/reports/sparse_shift_conv_rewrite.csv")
    parser.add_argument("--task-ids", default="task363")
    parser.add_argument("--weight-name", default="wk")
    args = parser.parse_args()
    task_ids = _parse_task_ids(args.task_ids)
    if not task_ids:
        raise ValueError("--task-ids must contain at least one task id")
    rewrite_task_models(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        task_ids=task_ids,
        weight_name=args.weight_name,
    )


if __name__ == "__main__":
    main()
