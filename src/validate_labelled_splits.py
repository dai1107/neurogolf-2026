"""Validate an ONNX model on every labelled train/test/arc-gen case."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from .arc_io import load_task
from .encoding import DEFAULT_SHAPE, find_zero_confidence_cells, grid_to_onehot, onehot_to_grid
from .validate_onnx_model import find_nonzero_padding_cells


ort.set_default_logger_severity(3)

FIELDS = [
    "split",
    "case_index",
    "passed",
    "num_mismatched_cells",
    "zero_confidence_cells",
    "nonzero_padding_cells",
    "first_mismatch",
]


def _first_mismatch_text(mismatches: list[dict[str, int]]) -> str:
    if not mismatches:
        return ""
    return json.dumps(mismatches[0], sort_keys=True)


def _grid_size(grid: list[list[int]]) -> tuple[int, int]:
    if not grid:
        raise ValueError("grid must contain at least one row")
    width = len(grid[0])
    if width == 0:
        raise ValueError("grid rows must contain at least one cell")
    for row_index, row in enumerate(grid):
        if len(row) != width:
            raise ValueError(
                f"grid must be rectangular: row 0 has width {width}, "
                f"row {row_index} has width {len(row)}"
            )
    return len(grid), width


def _session_io(session: ort.InferenceSession) -> tuple[str, str]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise ValueError(f"model must have exactly one input, got {len(inputs)}")
    if len(outputs) != 1:
        raise ValueError(f"model must have exactly one output, got {len(outputs)}")
    return inputs[0].name, outputs[0].name


def _validate_case_with_session(
    session: ort.InferenceSession,
    input_name: str,
    output_name: str,
    input_grid: list[list[int]],
    expected_grid: list[list[int]],
) -> dict[str, object]:
    expected_height, expected_width = _grid_size(expected_grid)
    input_tensor = grid_to_onehot(input_grid)
    result = session.run([output_name], {input_name: input_tensor.astype(np.float32, copy=False)})
    output_tensor = result[0]
    if output_tensor.shape != DEFAULT_SHAPE:
        raise ValueError(f"output tensor shape must be {DEFAULT_SHAPE}, got {output_tensor.shape}")
    if not np.isfinite(output_tensor).all():
        raise ValueError("output tensor contains NaN or Inf")
    actual_grid = onehot_to_grid(output_tensor, expected_height, expected_width)

    mismatches: list[dict[str, int]] = []
    for row_index, row in enumerate(expected_grid):
        for col_index, expected_color in enumerate(row):
            actual_color = actual_grid[row_index][col_index]
            if actual_color != expected_color:
                mismatches.append(
                    {
                        "row": row_index,
                        "col": col_index,
                        "expected": expected_color,
                        "actual": actual_color,
                    }
                )

    zero_confidence_cells = find_zero_confidence_cells(
        output_tensor,
        expected_height,
        expected_width,
    )
    return {
        "passed": not mismatches,
        "num_mismatched_cells": len(mismatches),
        "mismatches": mismatches,
        "zero_confidence_cells": zero_confidence_cells,
        "nonzero_padding_cells": find_nonzero_padding_cells(
            output_tensor,
            expected_height,
            expected_width,
        ),
    }


def validate_labelled_splits(
    model_path: str,
    task_path: str,
    report_path: str,
) -> dict[str, Any]:
    """Write per-case strict validation results for labelled task splits."""
    task = load_task(task_path)
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name, output_name = _session_io(session)
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}

    for split in ("train", "test", "arc-gen"):
        cases = task.get(split, [])
        if not isinstance(cases, list):
            continue
        for case_index, case in enumerate(cases):
            if "output" not in case:
                continue
            result = _validate_case_with_session(
                session,
                input_name,
                output_name,
                case["input"],
                case["output"],
            )
            zero_confidence = result["zero_confidence_cells"]
            nonzero_padding = result["nonzero_padding_cells"]
            passed = (
                bool(result["passed"])
                and not zero_confidence
                and not nonzero_padding
            )
            rows.append(
                {
                    "split": split,
                    "case_index": case_index,
                    "passed": passed,
                    "num_mismatched_cells": result["num_mismatched_cells"],
                    "zero_confidence_cells": len(zero_confidence),
                    "nonzero_padding_cells": len(nonzero_padding),
                    "first_mismatch": _first_mismatch_text(result["mismatches"]),
                }
            )
            counts = split_counts.setdefault(split, {"passed": 0, "total": 0})
            counts["total"] += 1
            if passed:
                counts["passed"] += 1

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    passed_total = sum(1 for row in rows if row["passed"])
    summary = {
        "model_path": model_path,
        "task_path": task_path,
        "report_path": str(report),
        "passed": passed_total == total,
        "passed_cases": passed_total,
        "total_cases": total,
        "split_counts": split_counts,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    validate_labelled_splits(args.model, args.task, args.report)


if __name__ == "__main__":
    main()
