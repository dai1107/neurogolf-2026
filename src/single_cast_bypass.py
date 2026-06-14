"""Try bypassing one Cast node at a time and keep the cheapest valid result.

This is a broad discovery pass. For each single-output Cast whose output is not
the graph output, it rewires all consumers to the Cast input, removes the Cast,
and keeps the candidate only if ONNX checker, static-shape checks, forbidden-op
checks, and official-static cost all improve.
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

from .cost_estimator import check_forbidden_ops, check_static_shapes
from .hybrid_stack_optimizer import OPTIMIZE_FIELDS, _deduplicate_initializers, _prune_dead_graph
from .inspect_submission import HYBRID_STACK_DIRS
from .official_cost_estimator import estimate_official_static_cost


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


def _safe_fragment(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")[:80] or "cast"


def _bypass_cast_node(model: onnx.ModelProto, cast_index: int) -> str:
    graph = model.graph
    if cast_index < 0 or cast_index >= len(graph.node):
        raise IndexError(cast_index)
    node = graph.node[cast_index]
    if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
        raise ValueError("target node is not a single-output Cast")
    cast_input = node.input[0]
    cast_output = node.output[0]
    if cast_output in {value.name for value in graph.output}:
        raise ValueError("Cast output is graph output")
    if not cast_input or not cast_output:
        raise ValueError("Cast input/output must be named")

    replacement_count = 0
    kept_nodes: list[onnx.NodeProto] = []
    for index, other in enumerate(graph.node):
        if index == cast_index:
            continue
        for input_index, input_name in enumerate(other.input):
            if input_name == cast_output:
                other.input[input_index] = cast_input
                replacement_count += 1
        kept_nodes.append(other)
    if replacement_count == 0:
        raise ValueError("Cast output has no consumers")

    del graph.node[:]
    graph.node.extend(kept_nodes)

    kept_value_info = [value for value in graph.value_info if value.name != cast_output]
    del graph.value_info[:]
    graph.value_info.extend(kept_value_info)
    _prune_dead_graph(model)
    _deduplicate_initializers(model)
    return cast_output


def best_single_cast_bypass(
    source_model: str,
    output_model: str,
    min_cast_output_bytes: int = 0,
    max_cast_candidates: int = 0,
) -> dict[str, Any]:
    """Write the cheapest valid single-Cast-bypass candidate for one model."""
    source_path = Path(source_model)
    output_path = Path(output_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_model_proto = onnx.load(str(source_path))
    before = estimate_official_static_cost(str(source_path))
    source_node_count = int(before["node_count"])
    value_memory = _value_memory_bytes(source_model_proto)

    cast_indices: list[tuple[int, int]] = []
    for index, node in enumerate(source_model_proto.graph.node):
        if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
            continue
        if node.output[0] in {value.name for value in source_model_proto.graph.output}:
            continue
        memory = value_memory.get(node.output[0], 0)
        if memory < min_cast_output_bytes:
            continue
        cast_indices.append((memory, index))
    cast_indices.sort(reverse=True)
    if max_cast_candidates > 0:
        cast_indices = cast_indices[:max_cast_candidates]

    candidates: list[dict[str, Any]] = []
    for _memory, index in cast_indices:
        node = source_model_proto.graph.node[index]
        model = copy.deepcopy(source_model_proto)
        cast_output = node.output[0]
        candidate_path = output_path.with_name(
            f"{output_path.stem}_{index:04d}_{_safe_fragment(cast_output)}{output_path.suffix}"
        )
        try:
            removed_name = _bypass_cast_node(model, index)
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
                        "removed_cast_output": removed_name,
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
            "source_node_count": source_node_count,
            "output_node_count": source_node_count,
            "removed_cast_output": "",
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
        "source_node_count": source_node_count,
        "output_node_count": int(after["node_count"]),
        "removed_cast_output": best["removed_cast_output"],
        "candidate_count": len(candidates),
    }


def build_candidate_report(
    stack_dir: str,
    output_dir: str,
    report_path: str,
    lanes: set[str],
    task_ids: set[str] | None,
    min_cast_output_bytes: int = 0,
    max_cast_candidates: int = 0,
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
            destination = output_root / f"{task_id}_{lane}_SingleCastBypassed.onnx"
            row = {field: "" for field in OPTIMIZE_FIELDS}
            row.update({"task_id": task_id, "lane": lane, "source_model_path": str(source)})
            try:
                result = best_single_cast_bypass(
                    str(source),
                    str(destination),
                    min_cast_output_bytes=min_cast_output_bytes,
                    max_cast_candidates=max_cast_candidates,
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
                                "removed_cast_output": result["removed_cast_output"],
                                "candidate_count": result["candidate_count"],
                            },
                            sort_keys=True,
                        )
                        if changed
                        else "no cheaper single-Cast bypass found",
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
    parser.add_argument("--output-dir", default="outputs/candidates/single_cast_bypass")
    parser.add_argument("--report", default="outputs/reports/single_cast_bypass.csv")
    parser.add_argument("--lanes", default="overrides")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--min-cast-output-bytes", type=int, default=0)
    parser.add_argument("--max-cast-candidates-per-model", type=int, default=0)
    args = parser.parse_args()
    build_candidate_report(
        stack_dir=args.stack_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        lanes=_parse_csv_set(args.lanes),
        task_ids=_parse_csv_set(args.task_ids) or None,
        min_cast_output_bytes=args.min_cast_output_bytes,
        max_cast_candidates=args.max_cast_candidates_per_model,
    )


if __name__ == "__main__":
    main()
