"""Rewrite groups of Cast(... -> float32) feeding one consumer to float16.

Single-Cast fp16 rewrites miss patterns such as Concat/Mul/Sub where all inputs
must share a dtype. This pass tests one consumer group at a time and keeps only
checker/static/cost-improving candidates for later runtime equivalence checks.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto

from .cost_estimator import check_forbidden_ops, check_static_shapes
from .hybrid_stack_optimizer import OPTIMIZE_FIELDS, _deduplicate_initializers, _prune_dead_graph
from .inspect_submission import HYBRID_STACK_DIRS
from .official_cost_estimator import estimate_official_static_cost


def _safe_fragment(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")[:80] or "group"


def _cast_to(node: onnx.NodeProto) -> int | None:
    if node.op_type != "Cast":
        return None
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


def _set_cast_to(node: onnx.NodeProto, elem_type: int) -> None:
    for attr in node.attribute:
        if attr.name == "to":
            attr.i = int(elem_type)
            return
    attr = node.attribute.add()
    attr.name = "to"
    attr.i = int(elem_type)


def _value_memory_bytes(model: onnx.ModelProto) -> dict[str, int]:
    graph = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    result: dict[str, int] = {}
    for value in list(graph.input) + list(graph.value_info) + list(graph.output):
        if not value.type.HasField("tensor_type"):
            continue
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        elements = 1
        static = True
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                static = False
                break
            elements *= int(dim.dim_value)
        if not static:
            continue
        itemsize = int(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type).itemsize)
        result[value.name] = elements * itemsize
    return result


def _producer_indices(graph: onnx.GraphProto) -> dict[str, int]:
    return {output: index for index, node in enumerate(graph.node) for output in node.output if output}


def _candidate_groups(
    model: onnx.ModelProto,
    min_group_bytes: int,
) -> list[tuple[int, int, tuple[int, ...]]]:
    graph = model.graph
    producers = _producer_indices(graph)
    output_names = {value.name for value in graph.output}
    memory = _value_memory_bytes(model)
    groups: list[tuple[int, int, tuple[int, ...]]] = []

    for consumer_index, consumer in enumerate(graph.node):
        cast_indices: list[int] = []
        total_memory = 0
        for input_name in consumer.input:
            producer_index = producers.get(input_name)
            if producer_index is None:
                continue
            producer = graph.node[producer_index]
            if producer.op_type != "Cast" or len(producer.output) != 1:
                continue
            if _cast_to(producer) != TensorProto.FLOAT:
                continue
            if producer.output[0] in output_names:
                continue
            cast_indices.append(producer_index)
            total_memory += int(memory.get(producer.output[0], 0))
        unique_cast_indices = tuple(sorted(set(cast_indices)))
        if len(unique_cast_indices) < 2 or total_memory < min_group_bytes:
            continue
        groups.append((total_memory, consumer_index, unique_cast_indices))

    groups.sort(reverse=True)
    return groups


def best_float_cast_group_to_float16(
    source_model: str,
    output_model: str,
    min_group_bytes: int = 1000,
    max_groups: int = 0,
) -> dict[str, Any]:
    """Write the cheapest valid grouped float32-Cast-to-float16 rewrite."""
    source_path = Path(source_model)
    output_path = Path(output_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_model_proto = onnx.load(str(source_path))
    before = estimate_official_static_cost(str(source_path))
    source_groups = _candidate_groups(source_model_proto, min_group_bytes=min_group_bytes)
    if max_groups > 0:
        source_groups = source_groups[:max_groups]

    candidates: list[dict[str, Any]] = []
    for _memory, consumer_index, cast_indices in source_groups:
        model = copy.deepcopy(source_model_proto)
        graph = model.graph
        rewritten_outputs: list[str] = []
        for cast_index in cast_indices:
            cast_node = graph.node[cast_index]
            _set_cast_to(cast_node, TensorProto.FLOAT16)
            rewritten_outputs.append(cast_node.output[0])
        consumer = graph.node[consumer_index]
        fragment = _safe_fragment(f"{consumer.op_type}_{consumer_index}_{'_'.join(rewritten_outputs)}")
        candidate_path = output_path.with_name(f"{output_path.stem}_{fragment}{output_path.suffix}")
        try:
            kept_value_info = [
                value for value in graph.value_info if value.name not in set(rewritten_outputs)
            ]
            del graph.value_info[:]
            graph.value_info.extend(kept_value_info)
            _prune_dead_graph(model)
            _deduplicate_initializers(model)
            onnx.checker.check_model(model, full_check=True)
            onnx.save(model, str(candidate_path))
            onnx.checker.check_model(str(candidate_path), full_check=True)
            forbidden = check_forbidden_ops(str(candidate_path))
            static = check_static_shapes(str(candidate_path))
            if not forbidden["passed"]:
                raise ValueError(f"forbidden_ops={forbidden['forbidden_ops_found']}")
            if not static["passed"]:
                raise ValueError(f"static_shapes={static['failures'][:3]}")
            after = estimate_official_static_cost(str(candidate_path))
            delta = int(after["official_static_cost"]) - int(before["official_static_cost"])
            if delta < 0:
                candidates.append(
                    {
                        "candidate_path": candidate_path,
                        "rewritten_cast_outputs": rewritten_outputs,
                        "consumer_op": consumer.op_type,
                        "after": after,
                        "delta": delta,
                    }
                )
        except Exception:
            if candidate_path.exists():
                candidate_path.unlink()
            continue

    if not candidates:
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
            "rewritten_cast_outputs": [],
            "consumer_op": "",
            "candidate_count": 0,
        }

    best = min(candidates, key=lambda item: int(item["delta"]))
    shutil.copyfile(best["candidate_path"], output_path)
    for item in candidates:
        path = Path(item["candidate_path"])
        if path != output_path and path.exists():
            path.unlink()
    after = best["after"]
    return {
        "source_model_path": str(source_path),
        "output_model_path": str(output_path),
        "source_estimated_cost": int(before["official_static_cost"]),
        "output_estimated_cost": int(after["official_static_cost"]),
        "estimated_cost_delta": int(best["delta"]),
        "source_file_size_bytes": int(before["file_size_bytes"]),
        "output_file_size_bytes": int(after["file_size_bytes"]),
        "file_size_delta": int(after["file_size_bytes"]) - int(before["file_size_bytes"]),
        "source_node_count": int(before["node_count"]),
        "output_node_count": int(after["node_count"]),
        "rewritten_cast_outputs": best["rewritten_cast_outputs"],
        "consumer_op": best["consumer_op"],
        "candidate_count": len(candidates),
    }


def build_candidate_report(
    stack_dir: str,
    output_dir: str,
    report_path: str,
    lanes: set[str],
    task_ids: set[str] | None,
    min_group_bytes: int = 1000,
    max_groups_per_model: int = 0,
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
            destination = output_root / f"{task_id}_{lane}_FloatCastGroupToFloat16.onnx"
            row = {field: "" for field in OPTIMIZE_FIELDS}
            row.update({"task_id": task_id, "lane": lane, "source_model_path": str(source)})
            try:
                result = best_float_cast_group_to_float16(
                    str(source),
                    str(destination),
                    min_group_bytes=min_group_bytes,
                    max_groups=max_groups_per_model,
                )
                changed = int(result["candidate_count"]) > 0
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
                        "removed_dead_nodes": int(result["source_node_count"])
                        - int(result["output_node_count"]),
                        "removed_unused_initializers": "0",
                        "deduplicated_initializers": "0",
                        "constant_gather_tables_pruned": "0",
                        "constant_gather_rows_removed": "0",
                        "constant_gather_bytes_removed": "0",
                        "initializer_bytes_delta": "0",
                        "checker_passed": str(changed),
                        "forbidden_ops_passed": str(changed),
                        "static_shapes_passed": str(changed),
                        "equivalence_passed": "not_run",
                        "candidate_valid": str(changed and int(result["estimated_cost_delta"]) < 0),
                        "failure_reason": json.dumps(
                            {
                                "rewritten_cast_outputs": result["rewritten_cast_outputs"],
                                "consumer_op": result["consumer_op"],
                                "candidate_count": result["candidate_count"],
                            },
                            sort_keys=True,
                        )
                        if changed
                        else "no cheaper grouped float32 Cast to float16 rewrite found",
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
    parser.add_argument("--stack-dir", default="outputs/current_6353_30_stack")
    parser.add_argument("--output-dir", default="outputs/candidates/float_cast_group_to_float16")
    parser.add_argument("--report", default="outputs/reports/float_cast_group_to_float16.csv")
    parser.add_argument("--lanes", default="overrides")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--min-group-bytes", type=int, default=1000)
    parser.add_argument("--max-groups-per-model", type=int, default=0)
    args = parser.parse_args()
    build_candidate_report(
        stack_dir=args.stack_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        lanes=_parse_csv_set(args.lanes),
        task_ids=_parse_csv_set(args.task_ids) or None,
        min_group_bytes=args.min_group_bytes,
        max_groups_per_model=args.max_groups_per_model,
    )


if __name__ == "__main__":
    main()
