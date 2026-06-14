"""Bypass Cast(input -> fp16) nodes when the resulting graph remains cheaper.

This is an experimental, validation-gated graph rewrite. Many generated models
start with a full-grid fp16 copy of the input. When that copy only feeds ops that
can accept the original float32 input without forcing larger downstream tensors,
removing the Cast can save the counted 1x10x30x30 intermediate.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto

from .cost_estimator import check_forbidden_ops, check_static_shapes
from .hybrid_stack_optimizer import OPTIMIZE_FIELDS, _deduplicate_initializers, _prune_dead_graph
from .inspect_submission import HYBRID_STACK_DIRS
from .official_cost_estimator import estimate_official_static_cost


def _cast_to(node: onnx.NodeProto) -> int | None:
    if node.op_type != "Cast":
        return None
    for attr in node.attribute:
        if attr.name == "to":
            return int(attr.i)
    return None


def _consumer_count(graph: onnx.GraphProto, value_name: str) -> int:
    count = 0
    for node in graph.node:
        count += sum(1 for input_name in node.input if input_name == value_name)
    return count


def bypass_input_casts(source_model: str, output_model: str) -> dict[str, Any]:
    """Write a model with eligible Cast(input -> fp16) outputs rewired to input."""
    source_path = Path(source_model)
    output_path = Path(output_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    before = estimate_official_static_cost(str(source_path))
    model = onnx.load(str(source_path))
    graph = model.graph
    output_names = {value.name for value in graph.output}

    bypassed: list[str] = []
    cast_outputs: set[str] = set()
    for node in list(graph.node):
        if node.op_type != "Cast" or len(node.input) != 1 or len(node.output) != 1:
            continue
        if node.input[0] != "input":
            continue
        if node.output[0] in output_names:
            continue
        if _cast_to(node) != TensorProto.FLOAT16:
            continue
        if _consumer_count(graph, node.output[0]) == 0:
            continue
        cast_outputs.add(node.output[0])
        bypassed.append(node.output[0])

    if not cast_outputs:
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
            "bypassed_casts": 0,
            "bypassed_names": "",
        }

    kept_nodes: list[onnx.NodeProto] = []
    for node in graph.node:
        if node.op_type == "Cast" and len(node.output) == 1 and node.output[0] in cast_outputs:
            continue
        for index, input_name in enumerate(node.input):
            if input_name in cast_outputs:
                node.input[index] = "input"
        kept_nodes.append(node)

    del graph.node[:]
    graph.node.extend(kept_nodes)

    kept_value_info = [value for value in graph.value_info if value.name not in cast_outputs]
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
        "bypassed_casts": len(bypassed),
        "bypassed_names": "|".join(bypassed),
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
            destination = output_root / f"{task_id}_{lane}_InputCastBypassed.onnx"
            row = {field: "" for field in OPTIMIZE_FIELDS}
            row.update({"task_id": task_id, "lane": lane, "source_model_path": str(source)})
            try:
                result = bypass_input_casts(str(source), str(destination))
                changed = int(result["bypassed_casts"]) > 0
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
                            changed
                            and checker_passed
                            and forbidden_passed
                            and static_passed
                            and int(result["estimated_cost_delta"]) < 0
                        ),
                        "failure_reason": json.dumps(
                            {
                                "bypassed_casts": result["bypassed_casts"],
                                "bypassed_names": result["bypassed_names"],
                            },
                            sort_keys=True,
                        )
                        if changed
                        else "no Cast(input -> fp16) nodes found",
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
    parser.add_argument("--output-dir", default="outputs/candidates/input_cast_bypass")
    parser.add_argument("--report", default="outputs/reports/input_cast_bypass.csv")
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
