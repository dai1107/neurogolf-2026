"""Decompile high-cost ONNX models — understand what tables/structures they contain.

Generates: outputs/reports/model_decompilation.csv

For each model with cost > threshold, analyzes every large initializer:
  - Shape, dtype, size
  - Structural classification (permutation, row-bank, interval, mask, template, etc.)
  - Whether it could be replaced by formula, Gather, or small Conv
  - Consumer/downstream op chain
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


DEFAULT_REPORT = "outputs/reports/model_decompilation.csv"
FIELDS = [
    "task_id",
    "model_cost",
    "init_name",
    "shape",
    "dtype",
    "num_elements",
    "nbytes",
    "init_class",
    "is_permutation",
    "is_row_bank",
    "is_interval_table",
    "is_mask_bank",
    "is_template_bank",
    "is_transform_bank",
    "is_one_hot",
    "one_hot_ratio",
    "num_unique",
    "value_range",
    "sparsity",
    "consumer_ops",
    "downstream_chain",
    "can_be_gather",
    "can_be_formula",
    "can_be_small_conv",
    "decompilation_notes",
]


def _classify_initializer(arr: np.ndarray, consumers: list[str], downstream: list[str]) -> dict[str, Any]:
    """Classify what this initializer represents structurally."""
    result: dict[str, Any] = {
        "init_class": "unknown",
        "is_permutation": False,
        "is_row_bank": False,
        "is_interval_table": False,
        "is_mask_bank": False,
        "is_template_bank": False,
        "is_transform_bank": False,
        "is_one_hot": False,
        "one_hot_ratio": 0.0,
        "num_unique": len(np.unique(arr)),
        "value_range": f"[{arr.min()},{arr.max()}]",
        "sparsity": float(np.mean(arr == 0)) if arr.size > 0 else 0.0,
        "can_be_gather": False,
        "can_be_formula": False,
        "can_be_small_conv": False,
    }

    if arr.size == 0:
        return result

    # One-hot detection
    if np.issubdtype(arr.dtype, np.floating) and arr.ndim >= 2:
        flat = arr.reshape(-1, arr.shape[-1])
        row_sums = flat.sum(axis=1)
        oh_ratio = float(np.mean(np.isclose(row_sums, 1.0)))
        result["one_hot_ratio"] = round(oh_ratio, 4)
        if oh_ratio > 0.99:
            result["is_one_hot"] = True
            result["init_class"] = "one_hot_selector"
            result["can_be_gather"] = True
            return result

        is_binary = bool(np.all((arr == 0.0) | (arr == 1.0)))
        if is_binary and oh_ratio > 0.5:
            result["init_class"] = "sparse_one_hot"
            result["can_be_gather"] = True
            return result

    # Integer tables
    if np.issubdtype(arr.dtype, np.integer):
        num_u = result["num_unique"]
        total = arr.size

        # Permutation-like: square last dims, many rows
        if arr.ndim >= 2 and arr.shape[-1] == arr.shape[-2] and num_u == arr.shape[-1]:
            result["init_class"] = "permutation_table"
            result["is_permutation"] = True
            result["can_be_gather"] = True
            return result

        # Interval / arange table
        if arr.ndim <= 2:
            # Check if values form arithmetic progressions
            if arr.ndim == 1:
                diffs = np.diff(arr.astype(np.int64))
                if len(diffs) > 0 and len(np.unique(diffs)) <= 2:
                    result["init_class"] = "interval_sequence"
                    result["is_interval_table"] = True
                    result["can_be_formula"] = True
                    return result
            elif arr.ndim == 2:
                col_diffs = np.diff(arr.astype(np.int64), axis=1)
                if col_diffs.size > 0 and len(np.unique(col_diffs)) <= 3:
                    result["init_class"] = "interval_grid"
                    result["is_interval_table"] = True
                    result["can_be_formula"] = True
                    return result

        # Row bank: many rows, small per-row values
        if arr.ndim == 2 and arr.shape[0] >= 10:
            rows_unique_ratio = num_u / max(1, arr.shape[0])
            if rows_unique_ratio < 0.3:
                result["init_class"] = "row_bank_lookup"
                result["is_row_bank"] = True
                return result

        # Mask bank: small value range
        if arr.ndim >= 1 and np.issubdtype(arr.dtype, np.integer):
            mn, mx = int(arr.min()), int(arr.max())
            if mx - mn <= 16 and total > 100:
                result["init_class"] = "small_int_table"
                result["is_mask_bank"] = True
                return result

        # Template bank: repeating patterns
        if arr.ndim >= 2 and num_u < total * 0.1:
            result["init_class"] = "template_bank"
            result["is_template_bank"] = True
            return result

        # Transform bank: wide range of values, possibly coordinates
        if arr.ndim == 2 and arr.shape[1] in (2, 3, 4) and mx > arr.shape[0]:
            result["init_class"] = "coordinate_bank"
            result["is_transform_bank"] = True
            return result

    # Float tables
    if np.issubdtype(arr.dtype, np.floating):
        is_binary = bool(np.all((arr == 0.0) | (arr == 1.0)))
        if is_binary:
            nz_per_row = np.count_nonzero(arr.reshape(arr.shape[0], -1), axis=1) if arr.ndim >= 2 else np.array([np.count_nonzero(arr)])
            nz_max = int(nz_per_row.max())
            if nz_max <= 4:
                result["init_class"] = "sparse_binary_mask"
                result["is_mask_bank"] = True
                result["can_be_gather"] = True
                return result
            else:
                result["init_class"] = "dense_binary_bank"
                result["is_mask_bank"] = True
                return result

    return result


def _consumer_info(model: onnx.ModelProto, init_name: str) -> tuple[list[str], str]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    direct_ops = list(set(n.op_type for n in consumers.get(init_name, [])))

    # Track downstream chain
    visited: set[str] = set()
    frontier = {init_name}
    chain_ops: list[str] = []
    for _ in range(4):
        next_f: set[str] = set()
        for name in frontier:
            if name in visited:
                continue
            visited.add(name)
            for node in consumers.get(name, []):
                chain_ops.append(node.op_type)
                next_f.update(node.output)
        frontier = next_f
        if not frontier:
            break

    chain_str = "→".join(chain_ops[:10])
    return direct_ops, chain_str


def decompile_models(
    model_dir: str = "outputs/onnx",
    cost_report_path: str = "outputs/reports/current_model_bank_report.csv",
    report_path: str = DEFAULT_REPORT,
    min_cost: int = 5000,
) -> dict[str, Any]:
    import csv as _csv

    cost_data: dict[str, int] = {}
    with Path(cost_report_path).open("r", newline="", encoding="utf-8") as handle:
        for row in _csv.DictReader(handle):
            if row.get("valid", "True").strip().lower() == "true":
                cost_data[row["task_id"].strip()] = int(row.get("estimated_cost") or 0)

    root = Path(model_dir)
    rows: list[dict[str, Any]] = []

    for model_path in sorted(root.glob("task*.onnx")):
        tid = model_path.stem
        cost = cost_data.get(tid, 0)
        if cost < min_cost:
            continue

        try:
            model = onnx.load(str(model_path))
        except Exception:
            continue

        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            if arr.nbytes < 1000:
                continue

            consumer_ops, chain = _consumer_info(model, init.name)
            cls_info = _classify_initializer(arr, consumer_ops, [])

            row = {
                "task_id": tid,
                "model_cost": cost,
                "init_name": init.name,
                "shape": "x".join(str(d) for d in arr.shape),
                "dtype": str(arr.dtype),
                "num_elements": int(arr.size),
                "nbytes": int(arr.nbytes),
                "init_class": cls_info["init_class"],
                "is_permutation": cls_info["is_permutation"],
                "is_row_bank": cls_info["is_row_bank"],
                "is_interval_table": cls_info["is_interval_table"],
                "is_mask_bank": cls_info["is_mask_bank"],
                "is_template_bank": cls_info["is_template_bank"],
                "is_transform_bank": cls_info["is_transform_bank"],
                "is_one_hot": cls_info["is_one_hot"],
                "one_hot_ratio": cls_info["one_hot_ratio"],
                "num_unique": cls_info["num_unique"],
                "value_range": cls_info["value_range"],
                "sparsity": round(cls_info["sparsity"], 4),
                "consumer_ops": ";".join(consumer_ops),
                "downstream_chain": chain,
                "can_be_gather": cls_info["can_be_gather"],
                "can_be_formula": cls_info["can_be_formula"],
                "can_be_small_conv": cls_info["can_be_small_conv"],
                "decompilation_notes": "",
            }
            rows.append(row)

    rows.sort(key=lambda r: (-r["nbytes"], r["task_id"]))

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    by_class: dict[str, int] = {}
    gather_candidates = 0
    formula_candidates = 0
    for r in rows:
        cls = r["init_class"]
        by_class[cls] = by_class.get(cls, 0) + 1
        if r["can_be_gather"]:
            gather_candidates += 1
        if r["can_be_formula"]:
            formula_candidates += 1

    summary = {
        "report_path": str(report),
        "models_analyzed": len(set(r["task_id"] for r in rows)),
        "initializers_analyzed": len(rows),
        "by_class": by_class,
        "gather_replaceable": gather_candidates,
        "formula_replaceable": formula_candidates,
        "top_by_nbytes": [
            {k: r[k] for k in ("task_id", "init_name", "shape", "nbytes", "init_class", "can_be_gather")}
            for r in rows[:20]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/onnx")
    parser.add_argument("--cost-report", default="outputs/reports/current_model_bank_report.csv")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--min-cost", type=int, default=5000)
    args = parser.parse_args()
    decompile_models(
        model_dir=args.model_dir,
        cost_report_path=args.cost_report,
        report_path=args.report,
        min_cost=args.min_cost,
    )


if __name__ == "__main__":
    main()
