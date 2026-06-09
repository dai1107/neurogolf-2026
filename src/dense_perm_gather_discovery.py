"""Discover dense one-hot MatMul tables that may be rewriteable as Gather."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


DEFAULT_REPORT = "outputs/reports/dense_perm_gather_discovery.csv"
DEFAULT_MIN_ELEMENTS = 4096
FIELDS = [
    "task_id",
    "model_path",
    "initializer_name",
    "shape",
    "dtype",
    "num_elements",
    "nbytes",
    "one_hot_axis",
    "rank",
    "unique_values",
    "matmul_consumer_count",
    "consumer_nodes",
    "candidate_kind",
    "priority_score",
]


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape) if shape else "scalar"


def _task_ids_from_report(report_path: str, limit: int | None) -> list[str]:
    import csv as _csv

    with Path(report_path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(_csv.DictReader(handle))
    rows.sort(key=lambda row: int(row.get("estimated_cost") or 0), reverse=True)
    if limit is not None:
        rows = rows[:limit]
    return [str(row["task_id"]) for row in rows]


def _is_float_one_hot_last_axis(array: np.ndarray) -> bool:
    if array.ndim < 2 or not np.issubdtype(array.dtype, np.floating):
        return False
    if array.size == 0:
        return False
    if not np.isfinite(array).all():
        return False
    if not np.all((array == 0.0) | (array == 1.0)):
        return False
    row_sums = np.sum(array, axis=-1)
    return bool(np.all(row_sums == 1.0))


def _constant_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and node.output:
                values[node.output[0]] = numpy_helper.to_array(onnx.helper.get_attribute_value(attr))
    return values


def discover_model(model_path: Path, task_id: str, min_elements: int) -> list[dict[str, Any]]:
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node)

    rows: list[dict[str, Any]] = []
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        if array.size < min_elements or not _is_float_one_hot_last_axis(array):
            continue
        matmul_consumers = [
            node
            for node in consumers.get(initializer.name, [])
            if node.op_type == "MatMul"
        ]
        if not matmul_consumers:
            continue
        unique_values = ",".join(str(value) for value in np.unique(array).tolist())
        candidate_kind = "one_hot_matmul"
        if array.shape[-1] == array.shape[-2]:
            candidate_kind = "permutation_matmul"
        rows.append(
            {
                "task_id": task_id,
                "model_path": str(model_path),
                "initializer_name": initializer.name,
                "shape": _shape_text(tuple(int(dim) for dim in array.shape)),
                "dtype": str(array.dtype),
                "num_elements": int(array.size),
                "nbytes": int(array.nbytes),
                "one_hot_axis": -1,
                "rank": int(array.ndim),
                "unique_values": unique_values,
                "matmul_consumer_count": len(matmul_consumers),
                "consumer_nodes": "; ".join(
                    f"{node.name or '<unnamed>'}->{','.join(node.output)}"
                    for node in matmul_consumers
                ),
                "candidate_kind": candidate_kind,
                "priority_score": int(array.size * max(1, len(matmul_consumers))),
            }
        )
    return rows


def discover_dense_perm_gather_candidates(
    model_dir: str = "outputs/onnx",
    report_path: str = DEFAULT_REPORT,
    task_ids: list[str] | None = None,
    top_from_report: str = "outputs/reports/current_model_bank_report.csv",
    top_limit: int | None = 80,
    min_elements: int = DEFAULT_MIN_ELEMENTS,
) -> dict[str, Any]:
    """Scan selected models for one-hot MatMul initializer tables."""
    if task_ids is None:
        task_ids = _task_ids_from_report(top_from_report, top_limit)
    root = Path(model_dir)
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for task_id in task_ids:
        model_path = root / f"{task_id}.onnx"
        if not model_path.is_file():
            missing.append(task_id)
            continue
        rows.extend(discover_model(model_path, task_id, min_elements))
    rows.sort(
        key=lambda row: (
            -int(row["priority_score"]),
            row["task_id"],
            row["initializer_name"],
        )
    )
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model_dir": str(root),
        "report_path": str(report),
        "task_count": len(task_ids),
        "missing_tasks": missing,
        "candidate_count": len(rows),
        "top_candidates": rows[:10],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_task_ids(raw: str) -> list[str] | None:
    task_ids = [item.strip() for item in raw.split(",") if item.strip()]
    return task_ids or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/onnx")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--top-from-report", default="outputs/reports/current_model_bank_report.csv")
    parser.add_argument("--top-limit", type=int, default=80)
    parser.add_argument("--min-elements", type=int, default=DEFAULT_MIN_ELEMENTS)
    args = parser.parse_args()
    discover_dense_perm_gather_candidates(
        model_dir=args.model_dir,
        report_path=args.report,
        task_ids=_parse_task_ids(args.task_ids),
        top_from_report=args.top_from_report,
        top_limit=args.top_limit,
        min_elements=args.min_elements,
    )


if __name__ == "__main__":
    main()
