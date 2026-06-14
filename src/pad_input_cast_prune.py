"""Remove Cast nodes that only feed a terminal Pad output.

The official verifier reads the graph output named ``output`` and thresholds it
with ``> 0``. For models shaped as

    Cast(binary_or_small_dtype) -> Pad -> output

the Cast output is a counted intermediate tensor. If Pad can consume the source
dtype directly, the graph can preserve decoded output while saving that tensor.
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
from onnx import AttributeProto, TensorProto, numpy_helper

from .cost_estimator import check_forbidden_ops, check_static_shapes
from .hybrid_stack_optimizer import OPTIMIZE_FIELDS, _deduplicate_initializers, _prune_dead_graph
from .inspect_submission import HYBRID_STACK_DIRS
from .official_cost_estimator import estimate_official_static_cost


def _producer_map(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for output_name in node.output:
            if output_name:
                producers[output_name] = node
    return producers


def _consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for input_name in node.input:
            if input_name:
                consumers.setdefault(input_name, []).append(node)
    return consumers


def _tensor_info(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }


def _value_shape_and_type(value: onnx.ValueInfoProto) -> tuple[list[int], int]:
    tensor_type = value.type.tensor_type
    if not value.type.HasField("tensor_type") or not tensor_type.HasField("shape"):
        raise ValueError(f"missing tensor type/shape for {value.name}")
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            raise ValueError(f"non-static dim for {value.name}")
        dims.append(int(dim.dim_value))
    return dims, int(tensor_type.elem_type)


def _constant_tensor_attribute(node: onnx.NodeProto) -> onnx.AttributeProto | None:
    if node.op_type != "Constant":
        return None
    for attr in node.attribute:
        if attr.name == "value" and attr.type == AttributeProto.TENSOR:
            return attr
    return None


def _find_initializer(graph: onnx.GraphProto, name: str) -> onnx.TensorProto | None:
    for initializer in graph.initializer:
        if initializer.name == name:
            return initializer
    return None


def _np_dtype_for_elem_type(elem_type: int) -> np.dtype:
    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type))


def _convert_zero_constant_to_type(
    model: onnx.ModelProto,
    value_name: str,
    elem_type: int,
) -> None:
    """Convert Pad constant_value to zero in elem_type.

    Only zero padding is rewritten. Nonzero pad values are rejected because the
    thresholded output equivalence would no longer be guaranteed across dtypes.
    """
    graph = model.graph
    initializer = _find_initializer(graph, value_name)
    np_dtype = _np_dtype_for_elem_type(elem_type)
    if initializer is not None:
        array = numpy_helper.to_array(initializer)
        if array.size == 0 or not np.all(array == 0):
            raise ValueError(f"Pad constant_value is not all zero: {value_name}")
        initializer.CopyFrom(
            numpy_helper.from_array(array.astype(np_dtype, copy=False), name=value_name)
        )
        return

    producer = _producer_map(graph).get(value_name)
    if producer is None:
        raise ValueError(f"Pad constant_value source not found: {value_name}")
    attr = _constant_tensor_attribute(producer)
    if attr is None:
        raise ValueError(f"Pad constant_value is not a tensor Constant: {value_name}")
    array = numpy_helper.to_array(attr.t)
    if array.size == 0 or not np.all(array == 0):
        raise ValueError(f"Pad constant_value is not all zero: {value_name}")
    replacement = numpy_helper.from_array(array.astype(np_dtype, copy=False))
    replacement.name = attr.t.name
    attr.t.CopyFrom(replacement)


def prune_pad_input_cast(source_model: str, output_model: str) -> dict[str, Any]:
    """Write a model with a terminal Pad's input Cast removed."""
    source_path = Path(source_model)
    output_path = Path(output_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    before = estimate_official_static_cost(str(source_path))
    model = onnx.load(str(source_path))
    graph = model.graph
    output_name = graph.output[0].name

    producers = _producer_map(graph)
    consumers = _consumer_map(graph)
    output_producer = producers.get(output_name)
    if output_producer is None or output_producer.op_type != "Pad":
        raise ValueError("graph output is not produced by Pad")
    if not output_producer.input:
        raise ValueError("terminal Pad has no data input")
    pad_data = output_producer.input[0]
    cast = producers.get(pad_data)
    if cast is None or cast.op_type != "Cast" or len(cast.input) != 1 or len(cast.output) != 1:
        raise ValueError("terminal Pad input is not produced by a single-output Cast")
    if consumers.get(pad_data, []) != [output_producer]:
        raise ValueError("Cast output has non-Pad consumers")

    info = _tensor_info(model)
    source_name = cast.input[0]
    source_value = info.get(source_name)
    output_value = info.get(output_name)
    if source_value is None or output_value is None:
        raise ValueError("missing static type info for Cast source or graph output")
    _source_shape, source_elem_type = _value_shape_and_type(source_value)
    output_shape, _old_output_elem_type = _value_shape_and_type(output_value)

    if len(output_producer.input) >= 3 and output_producer.input[2]:
        _convert_zero_constant_to_type(model, output_producer.input[2], source_elem_type)

    output_producer.input[0] = source_name
    kept_nodes = [node for node in graph.node if node is not cast]
    del graph.node[:]
    graph.node.extend(kept_nodes)

    tensor_type = graph.output[0].type.tensor_type
    tensor_type.elem_type = source_elem_type
    del tensor_type.shape.dim[:]
    for dim in output_shape:
        tensor_type.shape.dim.add().dim_value = int(dim)

    kept_value_info = [value for value in graph.value_info if value.name != pad_data]
    del graph.value_info[:]
    graph.value_info.extend(kept_value_info)
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
        "source_elem_type": source_elem_type,
        "output_shape": "x".join(str(dim) for dim in output_shape),
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
            destination = output_root / f"{task_id}_{lane}_PadInputCastPruned.onnx"
            row = {field: "" for field in OPTIMIZE_FIELDS}
            row.update({"task_id": task_id, "lane": lane, "source_model_path": str(source)})
            try:
                result = prune_pad_input_cast(str(source), str(destination))
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
                        "changed": "True",
                        "removed_dead_nodes": int(result["source_node_count"])
                        - int(result["output_node_count"]),
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
                            checker_passed
                            and forbidden_passed
                            and static_passed
                            and int(result["estimated_cost_delta"]) < 0
                        ),
                        "failure_reason": json.dumps(
                            {
                                "source_elem_type": result["source_elem_type"],
                                "output_shape": result["output_shape"],
                            },
                            sort_keys=True,
                        ),
                    }
                )
            except Exception as exc:
                if source.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.exists():
                        shutil.copyfile(source, destination)
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
    parser.add_argument("--stack-dir", default="outputs/current_6352_53_stack")
    parser.add_argument("--output-dir", default="outputs/candidates/pad_input_cast_prune")
    parser.add_argument("--report", default="outputs/reports/pad_input_cast_prune.csv")
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
