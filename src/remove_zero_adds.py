"""Remove identity ops: Add(0,X), Mul(1,X), Sub(X,0).

These are no-ops that waste nodes and initializers:
  Add(Zero, X) = X    Mul(Ones, X) = X    Sub(X, Zero) = X
This pass deletes the op node and rewires the non-trivial input directly
to the consumers.  Pure subtraction: deletes nodes AND zero/ones initializers.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from .cost_estimator import estimate_model_cost


FIELDS = [
    "task_id",
    "source_model_path",
    "output_model_path",
    "source_cost",
    "output_cost",
    "cost_delta",
    "source_file_size_bytes",
    "output_file_size_bytes",
    "file_size_delta",
    "removed_nodes",
    "removed_inits",
    "failure_reason",
]


def _is_all_zero(arr: np.ndarray) -> bool:
    return bool(np.count_nonzero(arr) == 0)


def _is_all_ones(arr: np.ndarray) -> bool:
    return bool(np.all(arr == 1.0))


def _shape_tuple(value_info: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    if not value_info.type.HasField("tensor_type"):
        return None
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    dims: list[int] = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        dims.append(int(dim.dim_value))
    return tuple(dims)


def _static_shapes(model: onnx.ModelProto) -> dict[str, tuple[int, ...]]:
    """Infer known static shapes for values and initializers."""
    try:
        graph = onnx.shape_inference.infer_shapes(model).graph
    except Exception:
        graph = model.graph

    shapes: dict[str, tuple[int, ...]] = {}
    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        shape = _shape_tuple(value_info)
        if shape is not None:
            shapes[value_info.name] = shape
    for initializer in graph.initializer:
        shapes[initializer.name] = tuple(int(dim) for dim in initializer.dims)
    return shapes


def _remove_identity_ops(model: onnx.ModelProto) -> tuple[int, int]:
    """Remove Add(0,X), Mul(1,X), Sub(X,0) identity ops.

    Returns (nodes_removed, inits_removed).
    """
    init_map = {}
    for init in model.graph.initializer:
        init_map[init.name] = numpy_helper.to_array(init)

    zero_init_names = {name for name, arr in init_map.items() if _is_all_zero(arr)}
    ones_init_names = {name for name, arr in init_map.items() if _is_all_ones(arr)}
    graph_output_names = {output.name for output in model.graph.output}
    shapes = _static_shapes(model)

    remove_node_indices: set[int] = set()
    rewires: dict[str, str] = {}

    def can_rewire(output_name: str, passthrough_name: str) -> bool:
        # Add/Mul/Sub identity constants can still broadcast the passthrough
        # value to a different rank/shape. Only remove when static shapes match.
        return shapes.get(output_name) is not None and shapes.get(output_name) == shapes.get(passthrough_name)

    for node_index, node in enumerate(model.graph.node):
        if len(node.input) < 2:
            continue
        if any(output_name in graph_output_names for output_name in node.output):
            continue
        inp0, inp1 = node.input[0], node.input[1]

        if node.op_type == "Add":
            # Add(0, X) = X or Add(X, 0) = X
            if inp0 in zero_init_names and can_rewire(node.output[0], inp1):
                remove_node_indices.add(node_index)
                rewires[node.output[0]] = inp1
            elif inp1 in zero_init_names and can_rewire(node.output[0], inp0):
                remove_node_indices.add(node_index)
                rewires[node.output[0]] = inp0

        elif node.op_type == "Mul":
            # Mul(1, X) = X or Mul(X, 1) = X
            if inp0 in ones_init_names and can_rewire(node.output[0], inp1):
                remove_node_indices.add(node_index)
                rewires[node.output[0]] = inp1
            elif inp1 in ones_init_names and can_rewire(node.output[0], inp0):
                remove_node_indices.add(node_index)
                rewires[node.output[0]] = inp0

        elif node.op_type == "Sub":
            # Sub(X, 0) = X (but NOT Sub(0, X))
            if inp1 in zero_init_names and can_rewire(node.output[0], inp0):
                remove_node_indices.add(node_index)
                rewires[node.output[0]] = inp0

    if not remove_node_indices:
        return 0, 0

    new_nodes = [node for node_index, node in enumerate(model.graph.node) if node_index not in remove_node_indices]

    for node in new_nodes:
        for i, inp in enumerate(node.input):
            if inp in rewires:
                node.input[i] = rewires[inp]

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    all_referenced = set()
    for node in model.graph.node:
        all_referenced.update(node.input)

    removed_inits = 0
    zero_and_ones = zero_init_names | ones_init_names
    kept_inits = []
    for init in model.graph.initializer:
        if init.name in zero_and_ones and init.name not in all_referenced:
            removed_inits += 1
        else:
            kept_inits.append(init)

    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits)

    return len(remove_node_indices), removed_inits


def optimize_model(
    input_model: str,
    output_model: str,
) -> dict[str, Any]:
    input_path = Path(input_model)
    output_path = Path(output_model)
    if not input_path.is_file():
        raise FileNotFoundError(f"not found: {input_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    onnx.checker.check_model(model)
    source_cost = estimate_model_cost(str(input_path))

    nodes, inits = _remove_identity_ops(model)

    if nodes == 0:
        import shutil
        shutil.copyfile(input_path, output_path)
        return {
            "source_model_path": str(input_path),
            "output_model_path": str(output_path),
            "source_cost": int(source_cost["estimated_cost"]),
            "output_cost": int(source_cost["estimated_cost"]),
            "cost_delta": 0,
            "source_file_size_bytes": int(source_cost["file_size_bytes"]),
            "output_file_size_bytes": int(source_cost["file_size_bytes"]),
            "file_size_delta": 0,
            "removed_nodes": 0,
            "removed_inits": 0,
            "failure_reason": "no zero-add patterns found",
        }

    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop()
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))

    output_cost = estimate_model_cost(str(output_path))
    return {
        "source_model_path": str(input_path),
        "output_model_path": str(output_path),
        "source_cost": int(source_cost["estimated_cost"]),
        "output_cost": int(output_cost["estimated_cost"]),
        "cost_delta": int(output_cost["estimated_cost"] - source_cost["estimated_cost"]),
        "source_file_size_bytes": int(source_cost["file_size_bytes"]),
        "output_file_size_bytes": int(output_cost["file_size_bytes"]),
        "file_size_delta": int(output_cost["file_size_bytes"] - source_cost["file_size_bytes"]),
        "removed_nodes": nodes,
        "removed_inits": inits,
        "failure_reason": "",
    }


def _parse_task_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _discover_task_ids(model_dir: str) -> list[str]:
    return sorted(path.stem for path in Path(model_dir).glob("task*.onnx"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/onnx")
    parser.add_argument("--output-dir", default="outputs/candidates/zero_add_removed")
    parser.add_argument("--report", default="outputs/reports/zero_add_removed.csv")
    parser.add_argument("--task-ids", default="")
    args = parser.parse_args()

    task_ids = _parse_task_ids(args.task_ids) if args.task_ids else _discover_task_ids(args.model_dir)
    if not task_ids:
        raise ValueError("no task ids")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for task_id in task_ids:
        src = Path(args.model_dir) / f"{task_id}.onnx"
        dst = output_dir / f"{task_id}_ZeroAddRemoved.onnx"
        row = optimize_model(str(src), str(dst))
        rows.append({"task_id": task_id, **row})

    with open(args.report, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    improved = [r for r in rows if int(r["cost_delta"]) < 0]
    summary = {
        "scanned": len(rows),
        "improved": len(improved),
        "total_nodes_removed": sum(int(r["removed_nodes"]) for r in rows),
        "total_inits_removed": sum(int(r["removed_inits"]) for r in rows),
        "total_cost_delta": sum(int(r["cost_delta"]) for r in rows),
        "improved_tasks": [r["task_id"] for r in improved],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
