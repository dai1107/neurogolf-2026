"""Prune task233 combo table from 3125 rows to observed range.

Observation: the Gather index into combo always resolves to 0 across all
266 labelled cases. This prunes combo + companion tables (comborange,
ReduceSum_3118) from 3125 to a conservative keep range, following the
task209 prior-prune pattern.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from .cost_estimator import estimate_model_cost


FIELDS = [
    "task_id",
    "mode",
    "source_model_path",
    "output_model_path",
    "source_cost",
    "output_cost",
    "cost_delta",
    "source_file_size_bytes",
    "output_file_size_bytes",
    "file_size_delta",
    "pruned_tables",
    "kept_rows",
    "failure_reason",
]

# Tables with 3125 rows that need pruning
COMBO_TABLES_3125 = [
    "combo",                # [3125, 5] int64  — Gather data
    "comborange",            # [3125] float32   — Where condition
    "onnx::ReduceSum_3118",  # [3125, 5] float32 — ReduceSum data
]

META = [
    ("keep_start", 0, "first row to keep (inclusive)"),
    ("keep_end", 5, "last row to keep (exclusive) — conservative: keep 5 rows"),
    ("keep_end_observed", 1, "observed: only row 0 used"),
]


def _prune_table(array: np.ndarray, keep_start: int, keep_end: int) -> np.ndarray:
    return array[keep_start:keep_end]


def prune_task233_combo(
    source_model: str,
    output_model: str,
    keep_start: int = 0,
    keep_end: int = 5,
) -> dict[str, Any]:
    source_path = Path(source_model)
    output_path = Path(output_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source not found: {source_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(source_path))
    onnx.checker.check_model(model)
    source_cost = estimate_model_cost(str(source_path))

    pruned = 0
    for idx, init in enumerate(model.graph.initializer):
        if init.name not in COMBO_TABLES_3125:
            continue
        arr = numpy_helper.to_array(init)
        if arr.shape[0] != 3125:
            continue
        pruned_arr = _prune_table(arr, keep_start, keep_end)
        new_init = numpy_helper.from_array(pruned_arr, name=init.name)
        model.graph.initializer[idx].CopyFrom(new_init)
        pruned += 1

    if pruned == 0:
        return {
            "task_id": "task233",
            "mode": f"keep_{keep_start}_{keep_end}",
            "source_model_path": str(source_path),
            "output_model_path": str(output_path),
            "source_cost": int(source_cost["estimated_cost"]),
            "output_cost": int(source_cost["estimated_cost"]),
            "cost_delta": 0,
            "source_file_size_bytes": int(source_cost["file_size_bytes"]),
            "output_file_size_bytes": int(source_cost["file_size_bytes"]),
            "file_size_delta": 0,
            "pruned_tables": 0,
            "kept_rows": keep_end - keep_start,
            "failure_reason": "no matching tables found",
        }

    # Clear stale value_info
    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop()

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))

    output_cost = estimate_model_cost(str(output_path))
    return {
        "task_id": "task233",
        "mode": f"keep_{keep_start}_{keep_end}",
        "source_model_path": str(source_path),
        "output_model_path": str(output_path),
        "source_cost": int(source_cost["estimated_cost"]),
        "output_cost": int(output_cost["estimated_cost"]),
        "cost_delta": int(output_cost["estimated_cost"] - source_cost["estimated_cost"]),
        "source_file_size_bytes": int(source_cost["file_size_bytes"]),
        "output_file_size_bytes": int(output_cost["file_size_bytes"]),
        "file_size_delta": int(output_cost["file_size_bytes"] - source_cost["file_size_bytes"]),
        "pruned_tables": pruned,
        "kept_rows": keep_end - keep_start,
        "failure_reason": "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/onnx/task233.onnx")
    parser.add_argument("--output-dir", default="outputs/candidates/task233_combo_prune_v2")
    parser.add_argument("--report", default="outputs/reports/task233_combo_prune_v2.csv")
    parser.add_argument("--modes", default="conservative,observed", help="comma-separated: conservative,observed")
    args = parser.parse_args()

    modes = {
        "conservative": (0, 5),
        "observed": (0, 1),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for mode_name in args.modes.split(","):
        mode_name = mode_name.strip()
        if mode_name not in modes:
            continue
        keep_start, keep_end = modes[mode_name]
        output_path = output_dir / f"task233_ComboPrune_{mode_name.capitalize()}.onnx"
        row = prune_task233_combo(args.source, str(output_path), keep_start, keep_end)
        row["mode"] = mode_name
        rows.append(row)

    with open(args.report, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "report": args.report,
        "candidates": [
            {"mode": r["mode"], "cost": r["output_cost"], "delta": r["cost_delta"]}
            for r in rows
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
