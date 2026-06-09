"""Validate an ONNX model on every labelled train/test/arc-gen case."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import onnxruntime as ort

from .arc_io import load_task
from .validate_onnx_model import validate_case


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


def validate_labelled_splits(
    model_path: str,
    task_path: str,
    report_path: str,
) -> dict[str, Any]:
    """Write per-case strict validation results for labelled task splits."""
    task = load_task(task_path)
    rows: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}

    for split in ("train", "test", "arc-gen"):
        cases = task.get(split, [])
        if not isinstance(cases, list):
            continue
        for case_index, case in enumerate(cases):
            if "output" not in case:
                continue
            result = validate_case(model_path, case["input"], case["output"])
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
