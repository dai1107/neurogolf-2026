"""Discover exact gather rewrites across all ONNX models.

Follows 优化策略.md 主线A: scan all models for dense one-hot/permutation/selector
tables that can be replaced with Gather index tables + minimal cast/slice nodes.

Patterns detected:
  Pattern A: Dense float one-hot matrix -> MatMul
  Pattern B: Dense float one-hot matrix -> Mul+ReduceSum
  Pattern C: Dense float binary selector -> Conv
  Pattern D: Oversized int32/int64 index tables -> smaller dtype
  Pattern E: Many similar float tables sharing an implicit index
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


DEFAULT_REPORT = "outputs/reports/exact_gather_rewrites_discovery.csv"
DEFAULT_MIN_ELEMENTS = 256
FIELDS = [
    "task_id",
    "model_path",
    "initializer_name",
    "shape",
    "dtype",
    "num_elements",
    "nbytes",
    "pattern",
    "one_hot_axis",
    "row_count",
    "col_count",
    "nonzero_per_row_max",
    "nonzero_per_row_min",
    "row_uniformity",
    "unique_values_count",
    "consumer_ops",
    "downstream_ops",
    "priority_score",
    "rewrite_suggestion",
]


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape) if shape else "scalar"


def _is_one_hot_last_axis(array: np.ndarray) -> tuple[bool, float]:
    """Check if array has exactly one 1.0 per row (last-axis rows)."""
    if array.ndim < 2 or array.size == 0:
        return False, 0.0
    if not np.issubdtype(array.dtype, np.floating):
        return False, 0.0
    if not np.all(np.isfinite(array)):
        return False, 0.0
    is_binary = np.all((array == 0.0) | (array == 1.0))
    if not is_binary:
        return False, 0.0
    flat = array.reshape(-1, array.shape[-1])
    row_sums = flat.sum(axis=1)
    one_hot_ratio = float(np.mean(np.isclose(row_sums, 1.0)))
    return one_hot_ratio > 0.99, one_hot_ratio


def _row_stats(array: np.ndarray) -> dict[str, Any]:
    """Compute per-row nonzero statistics."""
    if array.ndim < 2 or array.size == 0:
        return {"nonzero_per_row_max": 0, "nonzero_per_row_min": 0, "row_uniformity": 0.0}
    flat = array.reshape(array.shape[0], -1)
    row_nz = np.count_nonzero(flat, axis=1)
    return {
        "nonzero_per_row_max": int(row_nz.max()),
        "nonzero_per_row_min": int(row_nz.min()),
        "row_uniformity": float(1.0 - np.std(row_nz) / max(1, np.mean(row_nz))),
    }


def _is_index_table(array: np.ndarray) -> bool:
    """Check if int array looks like a gather index table."""
    if not np.issubdtype(array.dtype, np.integer):
        return False
    if array.size < 256:
        return False
    return True


def _downstream_ops(
    model: onnx.ModelProto,
    start_names: list[str],
    max_depth: int = 3,
) -> list[str]:
    """Collect downstream op types from the named values."""
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    visited: set[str] = set()
    frontier = set(start_names)
    ops: list[str] = []
    for _ in range(max_depth):
        next_frontier: set[str] = set()
        for name in frontier:
            if name in visited:
                continue
            visited.add(name)
            for consumer in consumers.get(name, []):
                ops.append(consumer.op_type)
                next_frontier.update(consumer.output)
        frontier = next_frontier
        if not frontier:
            break
    return ops


def discover_model(model_path: Path, task_id: str, min_elements: int) -> list[dict[str, Any]]:
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    consumers: dict[str, list[onnx.NodeProto]] = {}
    node_by_input: dict[str, list[str]] = {}
    for node in model.graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)
            node_by_input.setdefault(inp, []).append(node.op_type)

    rows: list[dict[str, Any]] = []
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        if array.size < min_elements:
            continue

        consumer_ops = list(set(
            node.op_type for node in consumers.get(initializer.name, [])
        ))
        dops = _downstream_ops(model, [initializer.name])
        downstream = list(set(dops)) if isinstance(dops, list) else []

        # Pattern A/B: Float one-hot per row
        is_oh, oh_ratio = _is_one_hot_last_axis(array)
        if is_oh:
            matmul_consumers = [n for n in consumers.get(initializer.name, []) if n.op_type == "MatMul"]
            mul_consumers = [n for n in consumers.get(initializer.name, []) if n.op_type == "Mul"]
            reduce_consumers = "ReduceSum" in downstream or "ReduceMean" in downstream

            pattern = "one_hot_matrix"
            suggestion = "Gather(index_of_1) + Reshape"
            if matmul_consumers:
                pattern = "one_hot_matmul"
                suggestion = "GatherIndexTable + Gather"
            if mul_consumers and reduce_consumers:
                pattern = "one_hot_mul_reduce"
                suggestion = "GatherIndexTable + Gather"

            rows.append(_make_row(
                task_id, model_path, initializer, array, pattern,
                oh_ratio, consumer_ops, downstream, suggestion,
                int(array.size * 2),
            ))
            continue

        # Pattern C: Large int index table that can be dtype-reduced
        if _is_index_table(array) and array.nbytes > 4096:
            suggestion = _dtype_suggestion(array)
            if suggestion:
                rows.append(_make_row(
                    task_id, model_path, initializer, array, "int_index_table",
                    0.0, consumer_ops, downstream, suggestion,
                    int(array.nbytes // 2),
                ))

        # Pattern D: Binary float table (non-uniform rows)
        if np.issubdtype(array.dtype, np.floating) and array.size >= 4096:
            is_binary = np.all((array == 0.0) | (array == 1.0))
            if is_binary:
                stats = _row_stats(array)
                nz_max = stats["nonzero_per_row_max"]
                nz_min = stats["nonzero_per_row_min"]
                suggestion = ""
                priority = 0

                if nz_max <= 4 and nz_min >= 1:
                    suggestion = "SparseGather: collect nonzero col indices as int index table"
                    priority = int(array.size * 100 // max(1, nz_max))
                elif stats["row_uniformity"] > 0.95:
                    suggestion = "RowCollapse: uniform row pattern, dedup rows"
                    priority = int(array.size // 2)

                if suggestion:
                    rows.append(_make_row(
                        task_id, model_path, initializer, array,
                        "binary_float_table",
                        0.0, consumer_ops, downstream, suggestion, priority,
                    ))

    rows.sort(key=lambda r: (-r["priority_score"], r["task_id"], r["initializer_name"]))
    return rows


def _dtype_suggestion(array: np.ndarray) -> str | None:
    """Suggest dtype compression for int tables."""
    if np.issubdtype(array.dtype, np.integer):
        minimum, maximum = int(array.min()), int(array.max())
        itemsize = array.dtype.itemsize
        if minimum >= 0 and maximum <= 255 and itemsize > 1:
            return f"uint8 (from {array.dtype}, range [{minimum},{maximum}])"
        if minimum >= 0 and maximum <= 65535 and itemsize > 2:
            return f"uint16 (from {array.dtype}, range [{minimum},{maximum}])"
        if minimum >= -128 and maximum <= 127 and itemsize > 1:
            return f"int8 (from {array.dtype}, range [{minimum},{maximum}])"
        if minimum >= -32768 and maximum <= 32767:
            if itemsize > 2:
                return f"int16 (from {array.dtype}, range [{minimum},{maximum}])"
            if itemsize == 2 and str(array.dtype) != "int16":
                return f"int16 more canonical (from {array.dtype})"
    return None


def _make_row(
    task_id: str,
    model_path: Path,
    initializer: onnx.TensorProto,
    array: np.ndarray,
    pattern: str,
    oh_ratio: float,
    consumer_ops: list[str],
    downstream: list[str],
    suggestion: str,
    priority: int,
) -> dict[str, Any]:
    stats = _row_stats(array)
    flat = array.reshape(-1, array.shape[-1]) if array.ndim >= 2 else array.reshape(1, -1)
    return {
        "task_id": task_id,
        "model_path": str(model_path),
        "initializer_name": initializer.name,
        "shape": _shape_text(tuple(int(dim) for dim in array.shape)),
        "dtype": str(array.dtype),
        "num_elements": int(array.size),
        "nbytes": int(array.nbytes),
        "pattern": pattern,
        "one_hot_axis": -1,
        "row_count": int(flat.shape[0]),
        "col_count": int(flat.shape[1]),
        "nonzero_per_row_max": int(stats["nonzero_per_row_max"]),
        "nonzero_per_row_min": int(stats["nonzero_per_row_min"]),
        "row_uniformity": f"{stats['row_uniformity']:.4f}",
        "unique_values_count": len(np.unique(array)),
        "consumer_ops": ";".join(sorted(consumer_ops)) if consumer_ops else "",
        "downstream_ops": ";".join(sorted(downstream)) if downstream else "",
        "priority_score": priority,
        "rewrite_suggestion": suggestion,
    }


def discover_all(
    model_dir: str = "outputs/onnx",
    report_path: str = DEFAULT_REPORT,
    task_ids: list[str] | None = None,
    min_elements: int = DEFAULT_MIN_ELEMENTS,
) -> dict[str, Any]:
    root = Path(model_dir)
    if task_ids is None:
        task_ids = sorted(p.stem for p in root.glob("task*.onnx"))

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for task_id in task_ids:
        model_path = root / f"{task_id}.onnx"
        if not model_path.is_file():
            missing.append(task_id)
            continue
        rows.extend(discover_model(model_path, task_id, min_elements))

    rows.sort(key=lambda r: (-r["priority_score"], r["task_id"], r["initializer_name"]))

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_pattern: dict[str, int] = {}
    for r in rows:
        by_pattern[r["pattern"]] = by_pattern.get(r["pattern"], 0) + 1

    summary = {
        "report_path": str(report),
        "task_count": len(task_ids),
        "missing_tasks": missing,
        "candidate_count": len(rows),
        "by_pattern": by_pattern,
        "top_candidates": [
            {k: r[k] for k in ("task_id", "initializer_name", "pattern", "nbytes", "rewrite_suggestion")}
            for r in rows[:15]
        ],
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
    parser.add_argument("--min-elements", type=int, default=DEFAULT_MIN_ELEMENTS)
    args = parser.parse_args()
    discover_all(
        model_dir=args.model_dir,
        report_path=args.report,
        task_ids=_parse_task_ids(args.task_ids),
        min_elements=args.min_elements,
    )


if __name__ == "__main__":
    main()
