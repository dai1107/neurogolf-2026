"""Build a compact taxonomy report for currently failed ARC tasks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .arc_io import load_all_tasks
from .pattern_rules import grid_shape


FIELDS = [
    "task_id",
    "status",
    "num_train_cases",
    "input_shapes",
    "output_shapes",
    "shape_relation",
    "color_relation",
    "changed_cells_total",
    "top_failure_reason",
    "reason_family",
]

NEAR_MISS_FIELDS = [
    "task_id",
    "rule_name",
    "near_miss_type",
    "matched_cases",
    "total_cases",
    "first_failed_case",
    "reason",
    "suggested_fix",
]


def _read_summary(summary_path: str) -> dict[str, dict[str, str]]:
    path = Path(summary_path)
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["task_id"]: row for row in csv.DictReader(handle)}


def _shape_relation(input_shapes: set[tuple[int, int]], output_shapes: set[tuple[int, int]]) -> str:
    if len(input_shapes) != 1 or len(output_shapes) != 1:
        return "variable_shapes"
    input_height, input_width = next(iter(input_shapes))
    output_height, output_width = next(iter(output_shapes))
    if (input_height, input_width) == (output_height, output_width):
        return "same_size"
    if output_height <= input_height and output_width <= input_width:
        return "shrinks_or_crop"
    if output_height >= input_height and output_width >= input_width:
        if output_height % input_height == 0 and output_width % input_width == 0:
            return "integer_scale_or_tile"
        return "expands_non_integer"
    return "mixed_resize"


def _color_relation(cases: list[dict[str, list[list[int]]]]) -> str:
    input_colors: set[int] = set()
    output_colors: set[int] = set()
    for case in cases:
        input_colors.update(color for row in case["input"] for color in row)
        output_colors.update(color for row in case["output"] for color in row)
    if input_colors == output_colors:
        return "same_color_set"
    if output_colors <= input_colors:
        return "output_subset_colors"
    if input_colors <= output_colors:
        return "output_superset_colors"
    return "changed_color_set"


def _changed_cells_total(cases: list[dict[str, list[list[int]]]]) -> int:
    total = 0
    for case in cases:
        if grid_shape(case["input"]) != grid_shape(case["output"]):
            continue
        for row_index, row in enumerate(case["input"]):
            for col_index, old_color in enumerate(row):
                if old_color != case["output"][row_index][col_index]:
                    total += 1
    return total


def _top_failure_reason(row: dict[str, str] | None) -> str:
    if row is None:
        return "missing_summary"
    raw = row.get("failure_reasons") or "[]"
    try:
        reasons = json.loads(raw)
    except json.JSONDecodeError:
        return "invalid_failure_json"
    if not reasons:
        return "none"
    counter = Counter(item.get("reason", "unknown") for item in reasons if isinstance(item, dict))
    if not counter:
        return "unknown"
    return counter.most_common(1)[0][0]


def _reason_family(reason: str) -> str:
    if "output_size_differs" in reason or "shape" in reason or "size" in reason:
        return "shape"
    if "color" in reason or "mapping" in reason:
        return "color"
    if "crop" in reason or "subsample" in reason:
        return "substructure"
    if "translation" in reason or "mirror" in reason or "rotation" in reason:
        return "geometry"
    if "scale" in reason or "tile" in reason or "periodic" in reason:
        return "tiling"
    if "panel" in reason:
        return "panel"
    if "local" in reason or "neighborhood" in reason:
        return "local"
    return "other"


def build_failure_taxonomy(
    data_dir: str,
    summary_path: str,
    report_path: str,
    failed_only: bool = True,
) -> list[dict[str, Any]]:
    """Write one taxonomy row per task, filtered to failed tasks by default."""
    tasks = load_all_tasks(data_dir)
    summary_rows = _read_summary(summary_path)
    rows: list[dict[str, Any]] = []
    for task_id, task in tasks.items():
        summary_row = summary_rows.get(task_id)
        status = "unknown" if summary_row is None else summary_row.get("status", "unknown")
        if failed_only and status != "failed":
            continue
        cases = task["train"]
        input_shapes = {grid_shape(case["input"]) for case in cases}
        output_shapes = {grid_shape(case["output"]) for case in cases}
        reason = _top_failure_reason(summary_row)
        rows.append(
            {
                "task_id": task_id,
                "status": status,
                "num_train_cases": len(cases),
                "input_shapes": " ".join(f"{height}x{width}" for height, width in sorted(input_shapes)),
                "output_shapes": " ".join(f"{height}x{width}" for height, width in sorted(output_shapes)),
                "shape_relation": _shape_relation(input_shapes, output_shapes),
                "color_relation": _color_relation(cases),
                "changed_cells_total": _changed_cells_total(cases),
                "top_failure_reason": reason,
                "reason_family": _reason_family(reason),
            }
        )

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _near_miss_type(rule_name: str, reason: str) -> tuple[str, str] | None:
    if "requires_shared_shape" in reason or "requires_one_shared_grid_size" in reason:
        return (
            "blocked_by_shared_shape",
            "Remove cross-case shared-shape requirement; operate on padded 30x30 with active-cell semantics.",
        )
    if "requires_shared_output_shape" in reason or "requires_one_shared_output_size" in reason:
        return (
            "blocked_by_shared_output_shape",
            "Allow per-case output shape when the rule semantics can infer active output area safely.",
        )
    if rule_name == "PeriodicExtensionColorMapRule" and (
        "no periodic extension" in reason or "requires_shared_shapes" in reason
    ):
        return (
            "different_period_per_case",
            "Probe auto-period extension where each case infers its own minimal period.",
        )
    if "probe-only" in reason or "NotImplemented" in reason:
        return (
            "probe_matched_but_builder_missing",
            "Implement an ONNX builder or keep this rule out of first_version_rules.",
        )
    if "no generalized panel layout" in reason or "panel" in reason:
        return (
            "panel_rule_near_miss",
            "Add panel selection/extraction probes beyond binary AND/OR/XOR panel operations.",
        )
    if "local" in reason or "neighborhood" in reason:
        return (
            "local_rule_almost_matches",
            "Inspect whether shape-polymorphic local fill/rewrite or hole fill covers this task.",
        )
    return None


def build_rule_near_miss(
    summary_path: str,
    log_dir: str,
    report_path: str,
) -> list[dict[str, Any]]:
    """Write a compact near-miss report from per-task solver logs."""
    summary_rows = _read_summary(summary_path)
    rows: list[dict[str, Any]] = []
    for task_id, summary_row in sorted(summary_rows.items()):
        if summary_row.get("status") != "failed":
            continue
        log_path = Path(log_dir) / f"{task_id}.json"
        if not log_path.is_file():
            continue
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        total_cases = int(log.get("analysis_summary", {}).get("num_train_cases", 0) or 0)
        for failure in log.get("failure_reasons", []):
            if not isinstance(failure, dict):
                continue
            rule_name = str(failure.get("rule", "unknown"))
            reason = str(failure.get("reason", "unknown"))
            classified = _near_miss_type(rule_name, reason)
            if classified is None:
                continue
            near_miss_type, suggested_fix = classified
            rows.append(
                {
                    "task_id": task_id,
                    "rule_name": rule_name,
                    "near_miss_type": near_miss_type,
                    "matched_cases": "",
                    "total_cases": total_cases,
                    "first_failed_case": "",
                    "reason": reason,
                    "suggested_fix": suggested_fix,
                }
            )
            break

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=NEAR_MISS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="task")
    parser.add_argument("--summary", default="outputs/reports/summary.csv")
    parser.add_argument("--report", default="outputs/reports/failure_taxonomy.csv")
    parser.add_argument("--near-miss-report", default="outputs/reports/rule_near_miss.csv")
    parser.add_argument("--log-dir", default="outputs/logs")
    parser.add_argument("--all-tasks", action="store_true")
    args = parser.parse_args()
    rows = build_failure_taxonomy(
        data_dir=args.data_dir,
        summary_path=args.summary,
        report_path=args.report,
        failed_only=not args.all_tasks,
    )
    near_miss_rows = build_rule_near_miss(
        summary_path=args.summary,
        log_dir=args.log_dir,
        report_path=args.near_miss_report,
    )
    print(f"taxonomy_rows = {len(rows)}")
    print(f"failure_taxonomy = {args.report}")
    print(f"near_miss_rows = {len(near_miss_rows)}")
    print(f"rule_near_miss = {args.near_miss_report}")


if __name__ == "__main__":
    main()
