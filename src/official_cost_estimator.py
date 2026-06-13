"""Approximate the official NeuroGolf dtype-aware static cost.

The older local estimator counts initializer elements plus initializer bytes.
The public utility in ``neurogolf_utils.py`` uses a different objective:
parameter count plus the static memory footprint of graph tensors.  This module
keeps that estimate separate so existing local-cost experiments remain
reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from .cost_estimator import FILE_SIZE_LIMIT_BYTES, check_forbidden_ops
from .inspect_submission import HYBRID_STACK_DIRS


FIELDS = [
    "task_id",
    "lane",
    "model_path",
    "valid",
    "failure_reason",
    "params",
    "tensor_memory_bytes",
    "official_static_cost",
    "official_static_score",
    "initializer_params",
    "constant_params",
    "node_count",
    "op_count",
    "file_size_bytes",
]


def _num_elements(dims: list[int]) -> int:
    if not dims:
        return 1
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"non-positive tensor dims: {dims}")
    return math.prod(int(dim) for dim in dims)


def _tensor_dtype_size(elem_type: int) -> int:
    return int(np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type)).itemsize)


def _value_info_memory(value_info: onnx.ValueInfoProto) -> int:
    tensor_type = value_info.type.tensor_type
    if value_info.type.HasField("sequence_type"):
        raise ValueError(f"{value_info.name}: sequence type is not allowed")
    if not value_info.type.HasField("tensor_type"):
        return 0
    if not tensor_type.HasField("shape"):
        raise ValueError(f"{value_info.name}: missing tensor shape")
    dims: list[int] = []
    for index, dim in enumerate(tensor_type.shape.dim):
        if dim.HasField("dim_param"):
            raise ValueError(f"{value_info.name}[{index}]: dynamic dim_param {dim.dim_param}")
        if not dim.HasField("dim_value"):
            raise ValueError(f"{value_info.name}[{index}]: missing dim_value")
        if dim.dim_value <= 0:
            raise ValueError(f"{value_info.name}[{index}]: non-positive dim {dim.dim_value}")
        dims.append(int(dim.dim_value))
    return _num_elements(dims) * _tensor_dtype_size(tensor_type.elem_type)


def _constant_attribute_params(node: onnx.NodeProto) -> int:
    if node.op_type != "Constant":
        return 0
    params = 0
    for attr in node.attribute:
        if attr.name == "value":
            params += _num_elements(list(attr.t.dims))
        elif attr.name == "sparse_value":
            params += _num_elements(list(attr.sparse_tensor.values.dims))
        elif attr.name == "value_floats":
            params += len(attr.floats)
        elif attr.name == "value_ints":
            params += len(attr.ints)
        elif attr.name == "value_strings":
            params += len(attr.strings)
    return params


def _parameter_counts(model: onnx.ModelProto) -> tuple[int, int, int]:
    initializer_params = 0
    for initializer in model.graph.initializer:
        initializer_params += _num_elements(list(initializer.dims))
    for sparse_initializer in model.graph.sparse_initializer:
        initializer_params += _num_elements(list(sparse_initializer.values.dims))

    constant_params = sum(_constant_attribute_params(node) for node in model.graph.node)
    return initializer_params + constant_params, initializer_params, constant_params


def _tensor_memory_bytes(model: onnx.ModelProto) -> int:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    if len(graph.input) > 1 or len(graph.output) > 1:
        raise ValueError("model must have exactly one input and one output")
    init_names = {init.name for init in graph.initializer}
    init_names.update(init.name for init in graph.sparse_initializer)
    io_names = {value.name for value in list(graph.input) + list(graph.output)}
    if io_names.intersection(init_names):
        raise ValueError("input/output name collides with an initializer")

    tensor_map = {
        value.name: value
        for value in list(graph.input) + list(graph.value_info) + list(graph.output)
    }
    if len(tensor_map) != len(list(graph.input) + list(graph.value_info) + list(graph.output)):
        raise ValueError("duplicate graph value_info/input/output names")

    tensor_names = set(tensor_map)
    for node in graph.node:
        for attr in node.attribute:
            if attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}:
                raise ValueError("subgraph attributes are not allowed")
        for output_name in node.output:
            if output_name:
                tensor_names.add(output_name)

    total = 0
    for tensor_name in sorted(tensor_names):
        if tensor_name in {"input", "output"}:
            continue
        value_info = tensor_map.get(tensor_name)
        if value_info is None:
            raise ValueError(f"missing static shape for tensor: {tensor_name}")
        total += _value_info_memory(value_info)
    return total


def estimate_official_static_cost(model_path: str) -> dict[str, Any]:
    """Estimate official score from static tensor memory and parameter count."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"model does not exist: {model_path}")
    model = onnx.load(str(path))
    onnx.checker.check_model(model, full_check=True)
    if model.functions:
        raise ValueError("ONNX functions are not allowed")
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            raise ValueError(f"custom opset domain is not allowed: {opset.domain}")
    forbidden = check_forbidden_ops(str(path))
    if not forbidden["passed"]:
        raise ValueError(f"forbidden ops: {forbidden['forbidden_ops_found']}")
    if path.stat().st_size > FILE_SIZE_LIMIT_BYTES:
        raise ValueError(f"file size exceeds limit: {path.stat().st_size}")

    params, initializer_params, constant_params = _parameter_counts(model)
    memory = _tensor_memory_bytes(model)
    cost = params + memory
    score = max(1.0, 25.0 - math.log(max(1.0, cost)))
    return {
        "params": int(params),
        "tensor_memory_bytes": int(memory),
        "official_static_cost": int(cost),
        "official_static_score": float(score),
        "initializer_params": int(initializer_params),
        "constant_params": int(constant_params),
        "node_count": len(model.graph.node),
        "op_count": len(model.graph.node),
        "file_size_bytes": int(path.stat().st_size),
    }


def _estimate_row(task_id: str, lane: str, model_path: Path) -> dict[str, Any]:
    try:
        estimate = estimate_official_static_cost(str(model_path))
        return {
            "task_id": task_id,
            "lane": lane,
            "model_path": str(model_path),
            "valid": True,
            "failure_reason": "",
            **estimate,
        }
    except Exception as exc:
        return {
            "task_id": task_id,
            "lane": lane,
            "model_path": str(model_path),
            "valid": False,
            "failure_reason": str(exc),
            "params": "",
            "tensor_memory_bytes": "",
            "official_static_cost": "",
            "official_static_score": "",
            "initializer_params": "",
            "constant_params": "",
            "node_count": "",
            "op_count": "",
            "file_size_bytes": model_path.stat().st_size if model_path.exists() else "",
        }


def build_stack_official_cost_report(
    stack_dir: str = "outputs/reference_6348_56_stack",
    report_path: str = "outputs/reports/ref6348_official_static_costs_20260613.csv",
) -> dict[str, Any]:
    """Write official-static cost estimates for every model in a hybrid stack."""
    root = Path(stack_dir)
    task_ids = sorted(path.stem for path in (root / "base_submission").glob("task*.onnx"))
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        for lane in HYBRID_STACK_DIRS:
            rows.append(_estimate_row(task_id, lane, root / lane / f"{task_id}.onnx"))

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["valid"]]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in valid_rows:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    best_rows = [
        min(task_rows, key=lambda row: int(row["official_static_cost"]))
        for task_rows in by_task.values()
    ]
    summary = {
        "report_path": str(report),
        "models": len(rows),
        "valid_models": len(valid_rows),
        "invalid_models": len(rows) - len(valid_rows),
        "task_ids_with_valid_model": len(best_rows),
        "sum_best_official_static_score": round(
            sum(float(row["official_static_score"]) for row in best_rows),
            6,
        ),
        "sum_best_official_static_cost": sum(int(row["official_static_cost"]) for row in best_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    one = subparsers.add_parser("one", help="Estimate one ONNX model")
    one.add_argument("--model", required=True)

    stack = subparsers.add_parser("stack", help="Estimate every model in a hybrid stack")
    stack.add_argument("--stack-dir", default="outputs/reference_6348_56_stack")
    stack.add_argument(
        "--report",
        default="outputs/reports/ref6348_official_static_costs_20260613.csv",
    )

    args = parser.parse_args()
    if args.command == "one":
        print(json.dumps(estimate_official_static_cost(args.model), ensure_ascii=False, indent=2))
    elif args.command == "stack":
        build_stack_official_cost_report(stack_dir=args.stack_dir, report_path=args.report)


if __name__ == "__main__":
    main()
