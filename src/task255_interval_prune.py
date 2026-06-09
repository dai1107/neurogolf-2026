"""Build task255 interval-table pruning candidates.

The current task255 graph enumerates all contiguous intervals in a 30-cell line:
30 * 31 / 2 = 465 rows.  It scores pairs of intervals with ArgMax.  This module
keeps a conservative subset of those interval rows, rewrites row-index tables,
and updates the ArgMax div/mod constants from the source row count to the new row
count.
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

from .cost_estimator import estimate_model_cost


DEFAULT_SOURCE = "outputs/onnx/task255.onnx"
DEFAULT_OBSERVED_REPORT = "outputs/reports/task255_selected_interval_observed.csv"
DEFAULT_OUTPUT_DIR = "outputs/candidates/task255_interval_prune"
DEFAULT_REPORT = "outputs/reports/task255_interval_prune.csv"

MODE_CHOICES = ("conservative", "medium", "observed", "safe_drop")
ROW_TABLES = {"I0", "I1", "ILEN", "MEMB", "AT0", "AT1", "rng", "up_idx", "dn_idx"}
INDEX_TABLES = {"up_idx", "dn_idx"}
CANONICAL_ROW_COUNT = 465
CANONICAL_LINE_WIDTH = 30
SAFE_DROP_ROWS = frozenset({31, 34, 57, 60, 61, 63, 85, 88, 89, 91, 448, 453, 460})

REPORT_FIELDS = [
    "value_name",
    "value_kind",
    "old_shape",
    "new_shape",
    "old_num_elements",
    "new_num_elements",
    "action",
]


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape) if shape else "scalar"


def _initializer_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }


def _read_observed_rows(path: str) -> set[int]:
    observed_path = Path(path)
    if not observed_path.is_file():
        raise FileNotFoundError(f"observed interval report does not exist: {path}")

    observed: set[int] = set()
    with observed_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ("first_a", "first_b", "second_a", "second_b"):
                if key not in row:
                    raise ValueError(f"observed report is missing column {key}: {path}")
                observed.add(int(row[key]))
    if not observed:
        raise ValueError(f"observed report has no selected interval rows: {path}")
    return observed


def _canonical_interval_for_row(row_index: int, width: int = CANONICAL_LINE_WIDTH) -> tuple[int, int]:
    """Return the original 30-cell contiguous interval encoded by a row index."""
    if row_index < 0 or row_index >= (width * (width + 1)) // 2:
        raise ValueError(f"canonical interval row is out of range: {row_index}")
    remaining = width
    offset = row_index
    start = 0
    while offset >= remaining:
        offset -= remaining
        start += 1
        remaining -= 1
    return start, start + offset


def _map_observed_rows_to_source(
    observed_rows: set[int],
    arrays: dict[str, np.ndarray],
) -> set[int]:
    """Map observed canonical row ids onto the source model's current row ids."""
    i0 = arrays["I0"]
    i1 = arrays["I1"]
    row_count = int(i0.shape[0])
    signature_to_row: dict[tuple[int, int], int] = {}
    duplicate_signatures: list[tuple[int, int]] = []
    for row_index in range(row_count):
        signature = (int(i0[row_index]), int(i1[row_index]))
        if signature in signature_to_row:
            duplicate_signatures.append(signature)
        signature_to_row[signature] = row_index
    if duplicate_signatures:
        raise ValueError(
            f"source interval table has duplicate interval signatures: {duplicate_signatures[:5]}"
        )

    mapped: set[int] = set()
    missing: list[dict[str, Any]] = []
    for observed in sorted(observed_rows):
        signature = _canonical_interval_for_row(observed)
        source_row = signature_to_row.get(signature)
        if source_row is None:
            missing.append({"observed_row": observed, "interval": signature})
            continue
        mapped.add(source_row)
    if missing:
        raise ValueError(f"observed interval rows are absent from source model: {missing[:5]}")
    return mapped


def _close_under_index_tables(
    keep: set[int],
    up_idx: np.ndarray,
    dn_idx: np.ndarray,
) -> set[int]:
    """Add interval rows referenced by each kept row's up/down index tables."""
    closed = set(keep)
    changed = True
    while changed:
        changed = False
        for row_index in list(closed):
            for referenced in (int(up_idx[row_index]), int(dn_idx[row_index])):
                if referenced not in closed:
                    closed.add(referenced)
                    changed = True
    return closed


def _validate_index_references_are_kept(
    keep: set[int],
    up_idx: np.ndarray,
    dn_idx: np.ndarray,
) -> None:
    """Reject a keep set that would leave an index-table reference dangling."""
    dropped_references: list[dict[str, int]] = []
    for row_index in sorted(keep):
        for table_name, table in (("up_idx", up_idx), ("dn_idx", dn_idx)):
            referenced = int(table[row_index])
            if referenced not in keep:
                dropped_references.append(
                    {
                        "row": row_index,
                        "table": table_name,
                        "referenced": referenced,
                    }
                )
    if dropped_references:
        preview = dropped_references[:5]
        raise ValueError(f"pruning leaves dangling index references: {preview}")


def _base_keep_indices(
    mode: str,
    arrays: dict[str, np.ndarray],
    observed_rows: set[int],
) -> set[int]:
    i0 = arrays["I0"]
    i1 = arrays["I1"]
    ilen = arrays["ILEN"]
    row_count = int(i0.shape[0])

    observed_lengths = {int(ilen[index]) for index in observed_rows}
    observed_starts = {int(i0[index]) for index in observed_rows}
    observed_ends = {int(i1[index]) for index in observed_rows}
    if mode == "observed":
        return set(observed_rows)
    if mode == "safe_drop":
        return set(range(row_count)) - set(SAFE_DROP_ROWS)
    if mode == "medium":
        return {
            index
            for index in range(row_count)
            if int(ilen[index]) in observed_lengths
        }
    if mode == "conservative":
        return {
            index
            for index in range(row_count)
            if (
                int(ilen[index]) in observed_lengths
                or int(i0[index]) in observed_starts
                or int(i1[index]) in observed_ends
            )
        }
    raise ValueError(f"unknown task255 interval pruning mode: {mode}")


def _keep_indices_for_mode(
    mode: str,
    arrays: dict[str, np.ndarray],
    observed_rows: set[int],
) -> np.ndarray:
    keep = _base_keep_indices(mode, arrays, observed_rows)
    if mode == "safe_drop":
        _validate_index_references_are_kept(keep, arrays["up_idx"], arrays["dn_idx"])
    else:
        keep = _close_under_index_tables(keep, arrays["up_idx"], arrays["dn_idx"])
    if not observed_rows <= keep:
        missing = sorted(observed_rows - keep)[:10]
        raise ValueError(f"{mode} pruning lost observed interval rows: {missing}")
    return np.asarray(sorted(keep), dtype=np.int64)


def _replace_initializer(
    initializer: onnx.TensorProto,
    new_array: np.ndarray,
    action: str,
    rows: list[dict[str, Any]],
) -> None:
    old_array = numpy_helper.to_array(initializer)
    initializer.CopyFrom(numpy_helper.from_array(new_array, name=initializer.name))
    rows.append(
        {
            "value_name": initializer.name,
            "value_kind": "initializer",
            "old_shape": _shape_text(tuple(int(dim) for dim in old_array.shape)),
            "new_shape": _shape_text(tuple(int(dim) for dim in new_array.shape)),
            "old_num_elements": int(old_array.size),
            "new_num_elements": int(new_array.size),
            "action": action,
        }
    )


def _replace_constant_row_count_scalar(
    node: onnx.NodeProto,
    attr: onnx.AttributeProto,
    old_row_count: int,
    new_row_count: int,
    rows: list[dict[str, Any]],
) -> bool:
    old_array = numpy_helper.to_array(onnx.helper.get_attribute_value(attr))
    if old_array.shape != () or int(old_array) != old_row_count:
        return False
    new_array = np.asarray(new_row_count, dtype=old_array.dtype)
    attr.CopyFrom(
        onnx.helper.make_attribute(
            "value",
            numpy_helper.from_array(new_array),
        )
    )
    rows.append(
        {
            "value_name": node.output[0] if node.output else node.name,
            "value_kind": "constant_node",
            "old_shape": _shape_text(tuple(int(dim) for dim in old_array.shape)),
            "new_shape": _shape_text(tuple(int(dim) for dim in new_array.shape)),
            "old_num_elements": int(old_array.size),
            "new_num_elements": int(new_array.size),
            "action": "row_count_scalar_updated",
        }
    )
    return True


def prune_task255_intervals(
    source_model: str = DEFAULT_SOURCE,
    output_model: str | None = None,
    report_path: str = DEFAULT_REPORT,
    observed_report: str = DEFAULT_OBSERVED_REPORT,
    mode: str = "conservative",
) -> dict[str, Any]:
    """Prune task255 interval rows and write one candidate ONNX model."""
    if mode not in MODE_CHOICES:
        raise ValueError(f"mode must be one of {', '.join(MODE_CHOICES)}, got {mode}")

    source_path = Path(source_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")

    model = onnx.load(str(source_path))
    arrays = _initializer_arrays(model)
    missing = sorted(ROW_TABLES - set(arrays))
    if missing:
        raise ValueError(f"source model is missing task255 row tables: {missing}")
    row_count = int(arrays["I0"].shape[0])
    if mode == "safe_drop" and row_count != CANONICAL_ROW_COUNT:
        raise ValueError(f"safe_drop expects original task255 row count, got {row_count}")

    canonical_observed_rows = _read_observed_rows(observed_report)
    observed_rows = _map_observed_rows_to_source(canonical_observed_rows, arrays)
    keep_indices = _keep_indices_for_mode(mode, arrays, observed_rows)
    new_row_count = int(keep_indices.size)
    old_to_new = {int(old): int(new) for new, old in enumerate(keep_indices.tolist())}

    rows: list[dict[str, Any]] = []
    updated_initializers = 0
    for initializer in model.graph.initializer:
        if initializer.name not in ROW_TABLES:
            continue
        array = numpy_helper.to_array(initializer)
        if array.ndim == 0 or int(array.shape[0]) != row_count:
            raise ValueError(f"unexpected shape for {initializer.name}: {array.shape}")

        if initializer.name == "rng":
            new_array = np.arange(new_row_count, dtype=array.dtype)
            action = "renumbered"
        elif initializer.name in INDEX_TABLES:
            remapped = [old_to_new[int(array[old_index])] for old_index in keep_indices.tolist()]
            new_array = np.asarray(remapped, dtype=array.dtype)
            action = "remapped"
        else:
            new_array = array[keep_indices]
            action = "sliced"
        _replace_initializer(initializer, new_array, action, rows)
        updated_initializers += 1

    updated_constant_nodes = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and _replace_constant_row_count_scalar(
                node, attr, row_count, new_row_count, rows
            ):
                updated_constant_nodes += 1

    del model.graph.value_info[:]
    onnx.checker.check_model(model)

    if output_model is None:
        output_model = str(
            Path(DEFAULT_OUTPUT_DIR) / f"task255_IntervalPrune{mode.title()}.onnx"
        )
    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path))

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    source_cost = estimate_model_cost(str(source_path))
    output_cost = estimate_model_cost(str(output_path))
    summary = {
        "source_model": str(source_path),
        "output_model": str(output_path),
        "report_path": str(report),
        "observed_report": observed_report,
        "mode": mode,
        "original_rows": row_count,
        "kept_rows": new_row_count,
        "canonical_observed_rows": len(canonical_observed_rows),
        "observed_rows": len(observed_rows),
        "updated_initializers": updated_initializers,
        "updated_constant_nodes": updated_constant_nodes,
        "source_estimated_cost": source_cost["estimated_cost"],
        "output_estimated_cost": output_cost["estimated_cost"],
        "source_file_size_bytes": source_cost["file_size_bytes"],
        "output_file_size_bytes": output_cost["file_size_bytes"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--observed-report", default=DEFAULT_OBSERVED_REPORT)
    parser.add_argument("--mode", default="conservative", choices=MODE_CHOICES)
    args = parser.parse_args()
    prune_task255_intervals(
        source_model=args.source,
        output_model=args.output,
        report_path=args.report,
        observed_report=args.observed_report,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
