"""Iteratively rewrite safe Cast(... -> int64) nodes to int32.

The single-step pass intentionally returns one best checker/static candidate.
Some models have many independent int64 index casts, and some checker-valid
casts fail ONNX Runtime because a downstream kernel only accepts int64.  This
tool is stricter: it tries one cast at a time, runs source-vs-candidate ORT
equivalence immediately, accepts only passing rewrites, and then repeats.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto

from .cost_estimator import check_forbidden_ops, check_static_shapes
from .hybrid_stack_optimizer import (
    OPTIMIZE_FIELDS,
    _deduplicate_initializers,
    _prune_dead_graph,
    validate_one_candidate,
)
from .inspect_submission import HYBRID_STACK_DIRS
from .int64_cast_to_int32 import _cast_to, _consumer_ops, _safe_fragment, _set_cast_to, _value_memory_bytes
from .official_cost_estimator import estimate_official_static_cost


STEP_FIELDS = [
    "task_id",
    "lane",
    "step",
    "status",
    "cast_output",
    "source_model_path",
    "candidate_model_path",
    "source_estimated_cost",
    "candidate_estimated_cost",
    "estimated_cost_delta",
    "equivalence_inputs_checked",
    "equivalence_max_abs_diff",
    "failure_reason",
]


def _parse_csv_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _cast_candidates(
    model: onnx.ModelProto,
    min_cast_output_bytes: int,
    banned_outputs: set[str],
) -> list[tuple[int, int, str]]:
    value_memory = _value_memory_bytes(model)
    graph_output_names = {value.name for value in model.graph.output}
    candidates: list[tuple[int, int, str]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Cast" or len(node.output) != 1:
            continue
        cast_output = node.output[0]
        if cast_output in banned_outputs or cast_output in graph_output_names:
            continue
        if _cast_to(node) != TensorProto.INT64:
            continue
        memory = value_memory.get(cast_output, 0)
        if memory < min_cast_output_bytes:
            continue
        consumer_ops = set(_consumer_ops(model.graph, cast_output))
        if consumer_ops and consumer_ops.issubset({"Reshape", "Expand", "Slice", "ConstantOfShape"}):
            continue
        candidates.append((memory, index, cast_output))
    candidates.sort(reverse=True)
    return candidates


def _write_candidate(
    source: onnx.ModelProto,
    node_index: int,
    cast_output: str,
    output_path: Path,
) -> None:
    model = copy.deepcopy(source)
    _set_cast_to(model.graph.node[node_index], TensorProto.INT32)
    kept_value_info = [value for value in model.graph.value_info if value.name != cast_output]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_value_info)
    _prune_dead_graph(model)
    _deduplicate_initializers(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path), full_check=True)
    forbidden = check_forbidden_ops(str(output_path))
    if not forbidden["passed"]:
        raise ValueError(f"forbidden_ops={forbidden['forbidden_ops_found']}")
    static = check_static_shapes(str(output_path))
    if not static["passed"]:
        raise ValueError(f"static_shapes={static['failures'][:3]}")


def optimize_one_model(
    source_model: Path,
    task_path: Path,
    output_model: Path,
    work_dir: Path,
    task_id: str,
    lane: str,
    min_cast_output_bytes: int,
    max_steps: int,
    fuzz_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    work_dir.mkdir(parents=True, exist_ok=True)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    original_cost = estimate_official_static_cost(str(source_model))
    current_path = source_model
    current_cost = original_cost
    accepted_steps: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    banned_outputs: set[str] = set()

    for step in range(1, max_steps + 1):
        model = onnx.load(str(current_path))
        candidates = _cast_candidates(
            model,
            min_cast_output_bytes=min_cast_output_bytes,
            banned_outputs=banned_outputs,
        )
        accepted_this_step = False
        for _memory, node_index, cast_output in candidates:
            candidate_path = work_dir / f"{task_id}_{lane}_step{step:03d}_{_safe_fragment(cast_output)}.onnx"
            source_cost = current_cost
            try:
                _write_candidate(model, node_index, cast_output, candidate_path)
                candidate_cost = estimate_official_static_cost(str(candidate_path))
                delta = int(candidate_cost["official_static_cost"]) - int(source_cost["official_static_cost"])
                if delta >= 0:
                    banned_outputs.add(cast_output)
                    step_rows.append(
                        {
                            "task_id": task_id,
                            "lane": lane,
                            "step": step,
                            "status": "rejected_non_improving",
                            "cast_output": cast_output,
                            "source_model_path": str(current_path),
                            "candidate_model_path": str(candidate_path),
                            "source_estimated_cost": source_cost["official_static_cost"],
                            "candidate_estimated_cost": candidate_cost["official_static_cost"],
                            "estimated_cost_delta": delta,
                            "equivalence_inputs_checked": 0,
                            "equivalence_max_abs_diff": "",
                            "failure_reason": "",
                        }
                    )
                    continue
                validation = validate_one_candidate(
                    source_model=str(current_path),
                    candidate_model=str(candidate_path),
                    task_path=str(task_path),
                    fuzz_count=fuzz_count,
                    seed=int(task_id[-3:]) + step,
                )
                if not validation["candidate_valid"]:
                    banned_outputs.add(cast_output)
                    step_rows.append(
                        {
                            "task_id": task_id,
                            "lane": lane,
                            "step": step,
                            "status": "rejected_equivalence",
                            "cast_output": cast_output,
                            "source_model_path": str(current_path),
                            "candidate_model_path": str(candidate_path),
                            "source_estimated_cost": source_cost["official_static_cost"],
                            "candidate_estimated_cost": candidate_cost["official_static_cost"],
                            "estimated_cost_delta": delta,
                            "equivalence_inputs_checked": validation["equivalence_inputs_checked"],
                            "equivalence_max_abs_diff": validation["equivalence_max_abs_diff"],
                            "failure_reason": validation["failure_reason"],
                        }
                    )
                    continue

                accepted = {
                    "task_id": task_id,
                    "lane": lane,
                    "step": step,
                    "status": "accepted",
                    "cast_output": cast_output,
                    "source_model_path": str(current_path),
                    "candidate_model_path": str(candidate_path),
                    "source_estimated_cost": source_cost["official_static_cost"],
                    "candidate_estimated_cost": candidate_cost["official_static_cost"],
                    "estimated_cost_delta": delta,
                    "equivalence_inputs_checked": validation["equivalence_inputs_checked"],
                    "equivalence_max_abs_diff": validation["equivalence_max_abs_diff"],
                    "failure_reason": "",
                }
                step_rows.append(accepted)
                accepted_steps.append(accepted)
                current_path = candidate_path
                current_cost = candidate_cost
                accepted_this_step = True
                break
            except Exception as exc:
                banned_outputs.add(cast_output)
                step_rows.append(
                    {
                        "task_id": task_id,
                        "lane": lane,
                        "step": step,
                        "status": "rejected_exception",
                        "cast_output": cast_output,
                        "source_model_path": str(current_path),
                        "candidate_model_path": str(candidate_path),
                        "source_estimated_cost": current_cost["official_static_cost"],
                        "candidate_estimated_cost": "",
                        "estimated_cost_delta": "",
                        "equivalence_inputs_checked": 0,
                        "equivalence_max_abs_diff": "",
                        "failure_reason": str(exc),
                    }
                )
        if not accepted_this_step:
            break

    row = {field: "" for field in OPTIMIZE_FIELDS}
    row.update(
        {
            "task_id": task_id,
            "lane": lane,
            "source_model_path": str(source_model),
            "source_estimated_cost": int(original_cost["official_static_cost"]),
            "source_file_size_bytes": int(original_cost["file_size_bytes"]),
            "changed": str(bool(accepted_steps)),
            "removed_dead_nodes": 0,
            "removed_unused_initializers": 0,
            "deduplicated_initializers": 0,
            "constant_gather_tables_pruned": 0,
            "constant_gather_rows_removed": 0,
            "constant_gather_bytes_removed": 0,
            "initializer_bytes_delta": 0,
            "checker_passed": str(bool(accepted_steps)),
            "forbidden_ops_passed": str(bool(accepted_steps)),
            "static_shapes_passed": str(bool(accepted_steps)),
            "equivalence_passed": str(bool(accepted_steps)),
            "candidate_valid": str(bool(accepted_steps)),
        }
    )
    if accepted_steps:
        shutil.copyfile(current_path, output_model)
        final_cost = estimate_official_static_cost(str(output_model))
        row.update(
            {
                "output_model_path": str(output_model),
                "output_estimated_cost": int(final_cost["official_static_cost"]),
                "estimated_cost_delta": int(final_cost["official_static_cost"])
                - int(original_cost["official_static_cost"]),
                "output_file_size_bytes": int(final_cost["file_size_bytes"]),
                "file_size_delta": int(final_cost["file_size_bytes"])
                - int(original_cost["file_size_bytes"]),
                "equivalence_inputs_checked": accepted_steps[-1]["equivalence_inputs_checked"],
                "equivalence_max_abs_diff": accepted_steps[-1]["equivalence_max_abs_diff"],
                "failure_reason": json.dumps(
                    {
                        "accepted_steps": len(accepted_steps),
                        "accepted_cast_outputs": [step["cast_output"] for step in accepted_steps],
                    },
                    sort_keys=True,
                ),
            }
        )
    else:
        row.update(
            {
                "output_model_path": "",
                "output_estimated_cost": int(original_cost["official_static_cost"]),
                "estimated_cost_delta": 0,
                "output_file_size_bytes": int(original_cost["file_size_bytes"]),
                "file_size_delta": 0,
                "failure_reason": "no equivalence-valid int64 Cast to int32 rewrite found",
            }
        )
    return row, step_rows


def build_candidate_report(
    stack_dir: str,
    task_dir: str,
    output_dir: str,
    report_path: str,
    step_report_path: str,
    lanes: set[str],
    task_ids: set[str] | None,
    min_cast_output_bytes: int,
    max_steps: int,
    fuzz_count: int,
) -> dict[str, Any]:
    root = Path(stack_dir)
    output_root = Path(output_dir)
    rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    for lane in sorted(lanes):
        if lane not in HYBRID_STACK_DIRS:
            raise ValueError(f"unknown lane: {lane}")
        for source in sorted((root / lane).glob("task*.onnx")):
            task_id = source.stem
            if task_ids is not None and task_id not in task_ids:
                continue
            row, steps = optimize_one_model(
                source_model=source,
                task_path=Path(task_dir) / f"{task_id}.json",
                output_model=output_root / f"{task_id}_{lane}_IterativeInt64CastToInt32.onnx",
                work_dir=output_root / "_work" / lane / task_id,
                task_id=task_id,
                lane=lane,
                min_cast_output_bytes=min_cast_output_bytes,
                max_steps=max_steps,
                fuzz_count=fuzz_count,
            )
            rows.append(row)
            step_rows.extend(steps)

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPTIMIZE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    step_report = Path(step_report_path)
    step_report.parent.mkdir(parents=True, exist_ok=True)
    with step_report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_FIELDS)
        writer.writeheader()
        writer.writerows(step_rows)

    valid_rows = [row for row in rows if row["candidate_valid"] == "True"]
    summary = {
        "stack_dir": stack_dir,
        "output_dir": output_dir,
        "report_path": report_path,
        "step_report_path": step_report_path,
        "rows": len(rows),
        "valid_candidates": len(valid_rows),
        "accepted_steps": sum(1 for row in step_rows if row["status"] == "accepted"),
        "total_estimated_cost_delta": sum(int(row["estimated_cost_delta"]) for row in valid_rows),
        "improved_tasks": [row["task_id"] for row in valid_rows],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-dir", required=True)
    parser.add_argument("--task-dir", default="task")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--step-report", required=True)
    parser.add_argument("--lanes", default="overrides")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--min-cast-output-bytes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--fuzz-count", type=int, default=20)
    args = parser.parse_args()
    build_candidate_report(
        stack_dir=args.stack_dir,
        task_dir=args.task_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        step_report_path=args.step_report,
        lanes=_parse_csv_set(args.lanes),
        task_ids=_parse_csv_set(args.task_ids) or None,
        min_cast_output_bytes=args.min_cast_output_bytes,
        max_steps=args.max_steps,
        fuzz_count=args.fuzz_count,
    )


if __name__ == "__main__":
    main()
