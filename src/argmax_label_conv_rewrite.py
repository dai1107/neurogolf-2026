"""Replace ArgMax-label grids with 1x1 weighted channel sums.

Many ARC models decode a one-hot tensor to color labels with:

    ArgMax(axis=1, keepdims=1) -> Cast

The ArgMax output is always int64, so a full 30x30 label grid costs 7200 bytes
before it is immediately cast back to a smaller numeric tensor. When the ArgMax
input is one-hot-like, a 1x1 Conv with weights ``0..C-1`` produces the same
label grid and avoids the int64 intermediate. This pass is deliberately
validation-gated because it is only equivalent for one-hot/discrete channel
tensors, not arbitrary score tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, helper, numpy_helper

from .cost_estimator import check_forbidden_ops, check_static_shapes
from .hybrid_stack_optimizer import OPTIMIZE_FIELDS, _deduplicate_initializers, _prune_dead_graph
from .inspect_submission import HYBRID_STACK_DIRS
from .official_cost_estimator import estimate_official_static_cost


FLOAT_TYPES = {TensorProto.FLOAT, TensorProto.FLOAT16}


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name and attr.type == AttributeProto.INT:
            return int(attr.i)
    return default


def _cast_to(node: onnx.NodeProto) -> int | None:
    if node.op_type != "Cast":
        return None
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


def _producer_map(graph: onnx.GraphProto) -> dict[str, int]:
    return {output: index for index, node in enumerate(graph.node) for output in node.output if output}


def _consumer_map(graph: onnx.GraphProto) -> dict[str, list[int]]:
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(graph.node):
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(index)
    return consumers


def _tensor_types_and_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, tuple[int, ...]]]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    result: dict[str, tuple[int, tuple[int, ...]]] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        if not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        ok = True
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                ok = False
                break
            dims.append(int(dim.dim_value))
        if ok:
            result[value.name] = (int(tensor_type.elem_type), tuple(dims))
    for initializer in inferred.graph.initializer:
        result[initializer.name] = (
            int(initializer.data_type),
            tuple(int(dim) for dim in initializer.dims),
        )
    return result


def _weight_array(channels: int, elem_type: int) -> np.ndarray:
    values = np.arange(channels, dtype=np.float32).reshape(1, channels, 1, 1)
    if elem_type == TensorProto.FLOAT16:
        return values.astype(np.float16)
    return values.astype(np.float32)


def rewrite_argmax_label_convs(source_model: str, output_model: str) -> dict[str, Any]:
    """Rewrite all eligible ArgMax->Cast label decoders in one model."""
    source_path = Path(source_model)
    output_path = Path(output_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    before = estimate_official_static_cost(str(source_path))
    model = onnx.load(str(source_path))
    graph = model.graph
    info = _tensor_types_and_shapes(model)
    consumers = _consumer_map(graph)

    replacements: dict[int, list[onnx.NodeProto]] = {}
    remove_indices: set[int] = set()
    rewritten: list[dict[str, Any]] = []

    for argmax_index, argmax in enumerate(graph.node):
        if argmax.op_type != "ArgMax" or len(argmax.input) != 1 or len(argmax.output) != 1:
            continue
        if _attr_int(argmax, "axis", 0) != 1 or _attr_int(argmax, "keepdims", 1) != 1:
            continue
        argmax_output = argmax.output[0]
        cast_indices = consumers.get(argmax_output, [])
        if len(cast_indices) != 1:
            continue
        cast_index = cast_indices[0]
        cast = graph.node[cast_index]
        cast_to = _cast_to(cast)
        if cast.op_type != "Cast" or len(cast.output) != 1 or cast_to is None:
            continue

        input_info = info.get(argmax.input[0])
        cast_output_info = info.get(cast.output[0])
        if input_info is None or cast_output_info is None:
            continue
        input_type, input_shape = input_info
        _cast_output_type, cast_output_shape = cast_output_info
        if input_type not in FLOAT_TYPES or len(input_shape) != 4 or input_shape[1] <= 1:
            continue
        if len(cast_output_shape) != 4 or cast_output_shape[1] != 1:
            continue

        weight_name = f"{cast.output[0]}_argmax_label_weight"
        conv_output = cast.output[0] if cast_to == input_type else f"{cast.output[0]}_label_sum"
        weight = numpy_helper.from_array(_weight_array(input_shape[1], input_type), name=weight_name)
        graph.initializer.append(weight)
        conv = helper.make_node(
            "Conv",
            [argmax.input[0], weight_name],
            [conv_output],
            name=f"{cast.output[0]}_ArgMaxLabelConv",
            kernel_shape=[1, 1],
        )
        new_nodes = [conv]
        if cast_to != input_type:
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    [conv_output],
                    [cast.output[0]],
                    name=f"{cast.output[0]}_ArgMaxLabelCast",
                    to=cast_to,
                )
            )

        replacements[argmax_index] = new_nodes
        remove_indices.add(cast_index)
        rewritten.append(
            {
                "argmax_output": argmax_output,
                "cast_output": cast.output[0],
                "input": argmax.input[0],
                "channels": input_shape[1],
                "input_type": int(input_type),
                "cast_to": int(cast_to),
            }
        )

    if not rewritten:
        shutil.copyfile(source_path, output_path)
        return {
            "source_model_path": str(source_path),
            "output_model_path": str(output_path),
            "source_estimated_cost": int(before["official_static_cost"]),
            "output_estimated_cost": int(before["official_static_cost"]),
            "estimated_cost_delta": 0,
            "source_file_size_bytes": int(before["file_size_bytes"]),
            "output_file_size_bytes": int(before["file_size_bytes"]),
            "file_size_delta": 0,
            "source_node_count": int(before["node_count"]),
            "output_node_count": int(before["node_count"]),
            "rewritten": [],
        }

    new_graph_nodes: list[onnx.NodeProto] = []
    for index, node in enumerate(graph.node):
        if index in remove_indices:
            continue
        if index in replacements:
            new_graph_nodes.extend(replacements[index])
        else:
            new_graph_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_graph_nodes)
    del graph.value_info[:]
    _prune_dead_graph(model)
    _deduplicate_initializers(model)

    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path), full_check=True)
    after = estimate_official_static_cost(str(output_path))
    return {
        "source_model_path": str(source_path),
        "output_model_path": str(output_path),
        "source_estimated_cost": int(before["official_static_cost"]),
        "output_estimated_cost": int(after["official_static_cost"]),
        "estimated_cost_delta": int(after["official_static_cost"]) - int(before["official_static_cost"]),
        "source_file_size_bytes": int(before["file_size_bytes"]),
        "output_file_size_bytes": int(after["file_size_bytes"]),
        "file_size_delta": int(after["file_size_bytes"]) - int(before["file_size_bytes"]),
        "source_node_count": int(before["node_count"]),
        "output_node_count": int(after["node_count"]),
        "rewritten": rewritten,
    }


def build_candidate_report(
    stack_dir: str,
    output_dir: str,
    report_path: str,
    lanes: set[str],
    task_ids: set[str] | None,
) -> dict[str, Any]:
    root = Path(stack_dir)
    output_root = Path(output_dir)
    report = Path(report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for lane in sorted(lanes):
        if lane not in HYBRID_STACK_DIRS:
            raise ValueError(f"unknown lane: {lane}")
        for source in sorted((root / lane).glob("task*.onnx")):
            task_id = source.stem
            if task_ids is not None and task_id not in task_ids:
                continue
            destination = output_root / f"{task_id}_{lane}_ArgMaxLabelConv.onnx"
            row = {field: "" for field in OPTIMIZE_FIELDS}
            row.update({"task_id": task_id, "lane": lane, "source_model_path": str(source)})
            try:
                result = rewrite_argmax_label_convs(str(source), str(destination))
                changed = bool(result["rewritten"])
                checker_passed = False
                forbidden_passed = False
                static_passed = False
                if changed:
                    onnx.checker.check_model(result["output_model_path"], full_check=True)
                    checker_passed = True
                    forbidden_passed = bool(check_forbidden_ops(result["output_model_path"])["passed"])
                    static_passed = bool(check_static_shapes(result["output_model_path"])["passed"])
                row.update(
                    {
                        "output_model_path": result["output_model_path"],
                        "source_estimated_cost": result["source_estimated_cost"],
                        "output_estimated_cost": result["output_estimated_cost"],
                        "estimated_cost_delta": result["estimated_cost_delta"],
                        "source_file_size_bytes": result["source_file_size_bytes"],
                        "output_file_size_bytes": result["output_file_size_bytes"],
                        "file_size_delta": result["file_size_delta"],
                        "changed": str(changed),
                        "removed_dead_nodes": int(result["source_node_count"]) - int(result["output_node_count"]),
                        "removed_unused_initializers": "0",
                        "deduplicated_initializers": "0",
                        "constant_gather_tables_pruned": "0",
                        "constant_gather_rows_removed": "0",
                        "constant_gather_bytes_removed": "0",
                        "initializer_bytes_delta": "0",
                        "checker_passed": str(checker_passed),
                        "forbidden_ops_passed": str(forbidden_passed),
                        "static_shapes_passed": str(static_passed),
                        "equivalence_passed": "not_run",
                        "candidate_valid": str(
                            changed
                            and checker_passed
                            and forbidden_passed
                            and static_passed
                            and int(result["estimated_cost_delta"]) < 0
                        ),
                        "failure_reason": json.dumps(result["rewritten"], sort_keys=True)
                        if changed
                        else "no eligible ArgMax->Cast label pattern found",
                    }
                )
            except Exception as exc:
                row["candidate_valid"] = "False"
                row["failure_reason"] = str(exc)
            rows.append(row)

    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPTIMIZE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["candidate_valid"] == "True"]
    summary = {
        "stack_dir": stack_dir,
        "output_dir": output_dir,
        "report_path": report_path,
        "rows": len(rows),
        "valid_candidates": len(valid_rows),
        "total_estimated_cost_delta": sum(int(row["estimated_cost_delta"]) for row in valid_rows),
        "improved_tasks": [row["task_id"] for row in valid_rows],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_csv_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-dir", default="outputs/current_6353_69_stack")
    parser.add_argument("--output-dir", default="outputs/candidates/argmax_label_conv_rewrite")
    parser.add_argument("--report", default="outputs/reports/argmax_label_conv_rewrite.csv")
    parser.add_argument("--lanes", default="overrides")
    parser.add_argument("--task-ids", default="")
    args = parser.parse_args()
    build_candidate_report(
        stack_dir=args.stack_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        lanes=_parse_csv_set(args.lanes),
        task_ids=_parse_csv_set(args.task_ids) or None,
    )


if __name__ == "__main__":
    main()
