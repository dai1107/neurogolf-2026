"""Diagnose high-cost tasks in the current validated ONNX model bank."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from .arc_io import load_task


FIELDS = [
    "task_id",
    "current_cost",
    "current_file_size",
    "input_shapes",
    "output_shapes",
    "same_shape",
    "shape_relation",
    "colors",
    "likely_rule_families",
    "recommended_action",
    "expected_cost_after_replacement",
]


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _read_current_report(report_path: str) -> dict[str, dict[str, Any]]:
    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError(f"current report does not exist: {report_path}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task_id = row.get("task_id", "")
            if not task_id:
                continue
            if row.get("valid") and not _is_true(row["valid"]):
                continue
            cost = row.get("estimated_cost", "")
            file_size = row.get("file_size_bytes", "")
            if cost == "" or file_size == "":
                continue
            rows[task_id] = {
                "task_id": task_id,
                "current_cost": int(cost),
                "current_file_size": int(file_size),
            }
    return rows


def _grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _shape_text(shapes: list[tuple[int, int]]) -> str:
    return ";".join(f"{height}x{width}" for height, width in shapes)


def _colors_text(task: dict) -> str:
    colors = sorted(
        {
            color
            for case in task["train"]
            for grid in (case["input"], case["output"])
            for row in grid
            for color in row
        }
    )
    return " ".join(str(color) for color in colors)


def _same_case_shapes(input_shapes: list[tuple[int, int]], output_shapes: list[tuple[int, int]]) -> bool:
    return all(input_shape == output_shape for input_shape, output_shape in zip(input_shapes, output_shapes))


def _shape_relation(input_shapes: list[tuple[int, int]], output_shapes: list[tuple[int, int]]) -> str:
    if _same_case_shapes(input_shapes, output_shapes):
        return "same_shape"
    if all(
        output_height <= input_height and output_width <= input_width
        for (input_height, input_width), (output_height, output_width) in zip(input_shapes, output_shapes)
    ):
        return "shrinks_or_crop"
    if all(
        output_height % input_height == 0 and output_width % input_width == 0
        for (input_height, input_width), (output_height, output_width) in zip(input_shapes, output_shapes)
    ):
        return "integer_scale_or_tile"
    if len(set(input_shapes)) > 1 or len(set(output_shapes)) > 1:
        return "variable_shapes"
    return "other_size_change"


def _has_color_role_change(task: dict) -> bool:
    for case in task["train"]:
        input_colors = {color for row in case["input"] for color in row}
        output_colors = {color for row in case["output"] for color in row}
        if input_colors != output_colors:
            return True
    return False


def _likely_rule_families(task: dict, shape_relation: str) -> list[str]:
    families: list[str] = []
    if shape_relation == "same_shape":
        families.extend(["identity", "color_map", "mirror_or_rotate", "local_neighborhood"])
        if _has_color_role_change(task):
            families.extend(["object_edit", "mask_recolor"])
    elif shape_relation == "shrinks_or_crop":
        families.extend(["fixed_crop", "dynamic_bbox_crop", "frame_or_substructure_extract"])
    elif shape_relation == "integer_scale_or_tile":
        families.extend(["scale_repeat", "tile_repeat", "mirror_concat"])
    elif shape_relation == "variable_shapes":
        families.extend(["dynamic_bbox", "padding_aware_transform", "panel_or_object_selection"])
    else:
        families.extend(["panel_transform", "periodic_extension", "symbolic_geometry"])
    return families


def _recommended_action(shape_relation: str, families: list[str]) -> str:
    if shape_relation == "same_shape":
        return "run formal same-shape rules, then inspect changed cells for compact mask algebra"
    if shape_relation == "shrinks_or_crop":
        return "try crop, dynamic bbox, frame interior, and substructure extraction builders"
    if shape_relation == "integer_scale_or_tile":
        return "try scale/tile rules and verify whether output is a repeated input primitive"
    if shape_relation == "variable_shapes":
        return "prefer dynamic bbox or padding-aware builders; reject static absolute-position guesses"
    return f"inspect for {families[0]} and add a narrow builder only after all train cases agree"


def _expected_cost(current_cost: int, shape_relation: str) -> int:
    targets = {
        "same_shape": 10_000,
        "shrinks_or_crop": 5_000,
        "integer_scale_or_tile": 8_000,
        "variable_shapes": 50_000,
        "other_size_change": 25_000,
    }
    return min(current_cost, targets.get(shape_relation, 25_000))


def _write_analysis_markdown(path: Path, row: dict[str, Any], task: dict) -> None:
    lines = [
        f"# {row['task_id']} High-Cost Diagnosis",
        "",
        f"- Current estimated cost: {row['current_cost']}",
        f"- Current file size: {row['current_file_size']} bytes",
        f"- Input shapes: {row['input_shapes']}",
        f"- Output shapes: {row['output_shapes']}",
        f"- Same shape: {row['same_shape']}",
        f"- Shape relation: {row['shape_relation']}",
        f"- Colors: {row['colors']}",
        f"- Likely rule families: {row['likely_rule_families']}",
        f"- Recommended action: {row['recommended_action']}",
        f"- Expected cost after replacement: {row['expected_cost_after_replacement']}",
        "",
        "## Train Cases",
        "",
    ]
    for index, case in enumerate(task["train"]):
        input_shape = _grid_shape(case["input"])
        output_shape = _grid_shape(case["output"])
        input_colors = sorted({color for train_row in case["input"] for color in train_row})
        output_colors = sorted({color for train_row in case["output"] for color in train_row})
        lines.extend(
            [
                f"### Case {index}",
                "",
                f"- Input shape: {input_shape[0]}x{input_shape[1]}",
                f"- Output shape: {output_shape[0]}x{output_shape[1]}",
                f"- Input colors: {' '.join(str(color) for color in input_colors)}",
                f"- Output colors: {' '.join(str(color) for color in output_colors)}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _select_tasks(
    current_rows: dict[str, dict[str, Any]],
    top_k: int,
    task_ids: list[str] | None,
) -> list[str]:
    if task_ids is not None:
        missing = [task_id for task_id in task_ids if task_id not in current_rows]
        if missing:
            raise ValueError(f"task ids missing from current report: {', '.join(missing)}")
        return task_ids
    return [
        item["task_id"]
        for item in sorted(current_rows.values(), key=lambda row: row["current_cost"], reverse=True)[:top_k]
    ]


def diagnose_high_cost_tasks(
    current_report: str,
    data_dir: str,
    report_path: str,
    analysis_dir: str,
    top_k: int = 30,
    task_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Write CSV and Markdown diagnoses for the highest-cost current models."""
    current_rows = _read_current_report(current_report)
    selected_task_ids = _select_tasks(current_rows, top_k, task_ids)
    data_root = Path(data_dir)
    rows: list[dict[str, Any]] = []
    for task_id in selected_task_ids:
        task = load_task(str(data_root / f"{task_id}.json"))
        input_shapes = [_grid_shape(case["input"]) for case in task["train"]]
        output_shapes = [_grid_shape(case["output"]) for case in task["train"]]
        same_shape = _same_case_shapes(input_shapes, output_shapes)
        relation = _shape_relation(input_shapes, output_shapes)
        families = _likely_rule_families(task, relation)
        current = current_rows[task_id]
        row = {
            "task_id": task_id,
            "current_cost": current["current_cost"],
            "current_file_size": current["current_file_size"],
            "input_shapes": _shape_text(input_shapes),
            "output_shapes": _shape_text(output_shapes),
            "same_shape": same_shape,
            "shape_relation": relation,
            "colors": _colors_text(task),
            "likely_rule_families": "|".join(families),
            "recommended_action": _recommended_action(relation, families),
            "expected_cost_after_replacement": _expected_cost(current["current_cost"], relation),
        }
        rows.append(row)
        _write_analysis_markdown(Path(analysis_dir) / f"{task_id}.md", row, task)

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _parse_task_ids(raw: str) -> list[str] | None:
    task_ids = [item.strip() for item in raw.split(",") if item.strip()]
    return task_ids or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-report", default="outputs/reports/current_model_bank_report.csv")
    parser.add_argument("--data-dir", default="task")
    parser.add_argument("--report", default="outputs/reports/high_cost_task_diagnosis.csv")
    parser.add_argument("--analysis-dir", default="outputs/reports/high_cost_task_analysis")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--task-ids", default="")
    args = parser.parse_args()
    rows = diagnose_high_cost_tasks(
        current_report=args.current_report,
        data_dir=args.data_dir,
        report_path=args.report,
        analysis_dir=args.analysis_dir,
        top_k=args.top_k,
        task_ids=_parse_task_ids(args.task_ids),
    )
    print(f"diagnosed_tasks = {len(rows)}")
    print(f"report_path = {args.report}")
    print(f"analysis_dir = {args.analysis_dir}")


if __name__ == "__main__":
    main()
