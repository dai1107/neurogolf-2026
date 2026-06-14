"""Remove exact no-op ONNX nodes and report official-static cost changes.

This pass is intentionally conservative. It only rewires nodes whose output is
provably identical to one input under static shape/type inference:

* Identity
* Cast where input and output dtype are the same
* Reshape where input and output shape are the same
* Transpose with identity permutation
* Add(0, X), Mul(1, X), and Sub(X, 0) when output shape equals X
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import AttributeProto, numpy_helper

from .cost_estimator import check_forbidden_ops, check_static_shapes
from .hybrid_stack_optimizer import OPTIMIZE_FIELDS, _deduplicate_initializers, _prune_dead_graph
from .inspect_submission import HYBRID_STACK_DIRS
from .official_cost_estimator import estimate_official_static_cost


@dataclass(frozen=True)
class TensorInfo:
    elem_type: int
    shape: tuple[int, ...]


@dataclass(frozen=True)
class NoopStats:
    removed_nodes: int = 0
    terminal_rewrites: int = 0
    removed_identity: int = 0
    removed_cast: int = 0
    removed_reshape: int = 0
    removed_transpose: int = 0
    removed_arithmetic: int = 0

    def changed(self) -> bool:
        return bool(self.removed_nodes or self.terminal_rewrites)

    def plus(self, other: "NoopStats") -> "NoopStats":
        return NoopStats(
            removed_nodes=self.removed_nodes + other.removed_nodes,
            terminal_rewrites=self.terminal_rewrites + other.terminal_rewrites,
            removed_identity=self.removed_identity + other.removed_identity,
            removed_cast=self.removed_cast + other.removed_cast,
            removed_reshape=self.removed_reshape + other.removed_reshape,
            removed_transpose=self.removed_transpose + other.removed_transpose,
            removed_arithmetic=self.removed_arithmetic + other.removed_arithmetic,
        )


def _tensor_info_from_value(value_info: onnx.ValueInfoProto) -> TensorInfo | None:
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
    return TensorInfo(elem_type=int(tensor_type.elem_type), shape=tuple(dims))


def _infer_tensor_info(model: onnx.ModelProto) -> dict[str, TensorInfo]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    info: dict[str, TensorInfo] = {}
    for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
        tensor_info = _tensor_info_from_value(value)
        if tensor_info is not None:
            info[value.name] = tensor_info
    for initializer in inferred.graph.initializer:
        info[initializer.name] = TensorInfo(
            elem_type=int(initializer.data_type),
            shape=tuple(int(dim) for dim in initializer.dims),
        )
    return info


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


def _constant_tensor_attribute(node: onnx.NodeProto) -> onnx.AttributeProto | None:
    if node.op_type != "Constant":
        return None
    for attr in node.attribute:
        if attr.name == "value" and attr.type == AttributeProto.TENSOR:
            return attr
    return None


def _constant_array(model: onnx.ModelProto, name: str) -> np.ndarray | None:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return numpy_helper.to_array(initializer)
    producer = _producer_map(model.graph).get(name)
    if producer is None:
        return None
    attr = _constant_tensor_attribute(producer)
    if attr is None:
        return None
    return numpy_helper.to_array(attr.t)


def _is_all_zero(model: onnx.ModelProto, name: str) -> bool:
    array = _constant_array(model, name)
    return bool(array is not None and array.size and np.all(array == 0))


def _is_all_one(model: onnx.ModelProto, name: str) -> bool:
    array = _constant_array(model, name)
    return bool(array is not None and array.size and np.all(array == 1))


def _attr_ints(node: onnx.NodeProto, name: str) -> list[int] | None:
    for attr in node.attribute:
        if attr.name == name and attr.type == AttributeProto.INTS:
            return [int(value) for value in attr.ints]
    return None


def _same_info(info: dict[str, TensorInfo], left: str, right: str) -> bool:
    return info.get(left) is not None and info.get(left) == info.get(right)


def _same_shape(info: dict[str, TensorInfo], left: str, right: str) -> bool:
    return info.get(left) is not None and info.get(right) is not None and info[left].shape == info[right].shape


def _passthrough_input(
    model: onnx.ModelProto,
    info: dict[str, TensorInfo],
    node: onnx.NodeProto,
) -> tuple[str, str] | None:
    if len(node.output) != 1:
        return None
    output = node.output[0]
    if not output:
        return None

    if node.op_type == "Identity" and len(node.input) == 1:
        source = node.input[0]
        if _same_info(info, output, source):
            return source, "Identity"

    if node.op_type == "Cast" and len(node.input) == 1:
        source = node.input[0]
        if _same_shape(info, output, source) and info.get(output) and info.get(source):
            if info[output].elem_type == info[source].elem_type:
                return source, "Cast"

    if node.op_type == "Reshape" and len(node.input) >= 1:
        source = node.input[0]
        if _same_info(info, output, source):
            return source, "Reshape"

    if node.op_type == "Transpose" and len(node.input) == 1:
        source = node.input[0]
        source_info = info.get(source)
        if source_info is None or not _same_info(info, output, source):
            return None
        perm = _attr_ints(node, "perm")
        if perm is None:
            perm = list(reversed(range(len(source_info.shape))))
        if perm == list(range(len(source_info.shape))):
            return source, "Transpose"

    if node.op_type in {"Add", "Mul", "Sub"} and len(node.input) == 2:
        left, right = node.input[0], node.input[1]
        if node.op_type == "Add":
            if _is_all_zero(model, left) and _same_info(info, output, right):
                return right, "Arithmetic"
            if _is_all_zero(model, right) and _same_info(info, output, left):
                return left, "Arithmetic"
        elif node.op_type == "Mul":
            if _is_all_one(model, left) and _same_info(info, output, right):
                return right, "Arithmetic"
            if _is_all_one(model, right) and _same_info(info, output, left):
                return left, "Arithmetic"
        elif node.op_type == "Sub":
            if _is_all_zero(model, right) and _same_info(info, output, left):
                return left, "Arithmetic"

    return None


def _resolve_rewire(name: str, rewires: dict[str, str]) -> str:
    seen: set[str] = set()
    while name in rewires and name not in seen:
        seen.add(name)
        name = rewires[name]
    return name


def _replace_output_type_shape(model: onnx.ModelProto, output_name: str, source_info: TensorInfo) -> None:
    for output in model.graph.output:
        if output.name != output_name:
            continue
        tensor_type = output.type.tensor_type
        tensor_type.elem_type = source_info.elem_type
        del tensor_type.shape.dim[:]
        for dim in source_info.shape:
            tensor_type.shape.dim.add().dim_value = int(dim)
        return
    raise ValueError(f"graph output not found: {output_name}")


def _prune_one_round(model: onnx.ModelProto) -> NoopStats:
    graph = model.graph
    info = _infer_tensor_info(model)
    graph_outputs = {output.name for output in graph.output}
    producers = _producer_map(graph)
    consumers = _consumer_map(graph)

    remove_indices: set[int] = set()
    rewires: dict[str, str] = {}
    stats = NoopStats()

    for index, node in enumerate(graph.node):
        passthrough = _passthrough_input(model, info, node)
        if passthrough is None:
            continue
        source, kind = passthrough
        output = node.output[0]

        if output in graph_outputs:
            producer = producers.get(source)
            if producer is None:
                continue
            if consumers.get(source, []) != [node]:
                continue
            for producer_output_index, producer_output in enumerate(producer.output):
                if producer_output == source:
                    producer.output[producer_output_index] = output
                    break
            else:
                continue
            source_info = info.get(source)
            if source_info is not None:
                _replace_output_type_shape(model, output, source_info)
            remove_indices.add(index)
            stats = stats.plus(NoopStats(removed_nodes=1, terminal_rewrites=1))
        else:
            remove_indices.add(index)
            rewires[output] = source
            stats = stats.plus(NoopStats(removed_nodes=1))

        if kind == "Identity":
            stats = stats.plus(NoopStats(removed_identity=1))
        elif kind == "Cast":
            stats = stats.plus(NoopStats(removed_cast=1))
        elif kind == "Reshape":
            stats = stats.plus(NoopStats(removed_reshape=1))
        elif kind == "Transpose":
            stats = stats.plus(NoopStats(removed_transpose=1))
        elif kind == "Arithmetic":
            stats = stats.plus(NoopStats(removed_arithmetic=1))

    if not remove_indices:
        return NoopStats()

    kept_nodes = [node for index, node in enumerate(graph.node) if index not in remove_indices]
    for node in kept_nodes:
        for input_index, input_name in enumerate(node.input):
            if input_name:
                node.input[input_index] = _resolve_rewire(input_name, rewires)

    del graph.node[:]
    graph.node.extend(kept_nodes)
    return stats


def prune_noop_nodes(source_model: str, output_model: str, max_rounds: int = 8) -> dict[str, Any]:
    """Write a no-op-pruned model and return official-static cost metadata."""
    source_path = Path(source_model)
    output_path = Path(output_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    before = estimate_official_static_cost(str(source_path))
    model = onnx.load(str(source_path))
    onnx.checker.check_model(model, full_check=True)

    stats = NoopStats()
    for _ in range(max_rounds):
        round_stats = _prune_one_round(model)
        if not round_stats.changed():
            break
        stats = stats.plus(round_stats)
        _prune_dead_graph(model)
        _deduplicate_initializers(model)

    if not stats.changed():
        shutil.copyfile(source_path, output_path)
        after = before
    else:
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
        "removed_nodes": stats.removed_nodes,
        "terminal_rewrites": stats.terminal_rewrites,
        "removed_identity": stats.removed_identity,
        "removed_cast": stats.removed_cast,
        "removed_reshape": stats.removed_reshape,
        "removed_transpose": stats.removed_transpose,
        "removed_arithmetic": stats.removed_arithmetic,
    }


def build_candidate_report(
    stack_dir: str,
    output_dir: str,
    report_path: str,
    lanes: set[str],
    task_ids: set[str] | None = None,
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
            destination = output_root / f"{task_id}_{lane}_NoopNodePruned.onnx"
            row = {field: "" for field in OPTIMIZE_FIELDS}
            row.update({"task_id": task_id, "lane": lane, "source_model_path": str(source)})
            try:
                result = prune_noop_nodes(str(source), str(destination))
                changed = int(result["removed_nodes"]) > 0
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
                        "source_initializer_count": "",
                        "output_initializer_count": "",
                        "changed": str(changed),
                        "removed_dead_nodes": "",
                        "removed_unused_initializers": "",
                        "removed_unused_value_info": "",
                        "deduplicated_initializers": "",
                        "constant_gather_tables_pruned": "",
                        "constant_gather_rows_removed": "",
                        "constant_gather_bytes_removed": "",
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
                        "failure_reason": json.dumps(
                            {
                                "removed_nodes": result["removed_nodes"],
                                "terminal_rewrites": result["terminal_rewrites"],
                                "removed_identity": result["removed_identity"],
                                "removed_cast": result["removed_cast"],
                                "removed_reshape": result["removed_reshape"],
                                "removed_transpose": result["removed_transpose"],
                                "removed_arithmetic": result["removed_arithmetic"],
                            },
                            sort_keys=True,
                        )
                        if changed
                        else "no exact no-op nodes found",
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
    parser.add_argument("--stack-dir", default="outputs/current_6352_53_stack")
    parser.add_argument("--output-dir", default="outputs/candidates/noop_node_prune")
    parser.add_argument("--report", default="outputs/reports/noop_node_prune.csv")
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
