"""Prune task209 prior tables to a labelled-safe index range."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .cost_estimator import estimate_model_cost


COL_PRIORS = ("ic_prior_2", "ic_prior_3", "ic_prior_4")
ROW_PRIORS = ("ir_prior_2", "ir_prior_3", "ir_prior_4")
COL_INDEX = "squeeze_231"
ROW_INDEX = "squeeze_233"

FIELDS = [
    "task_id",
    "mode",
    "source_model_path",
    "output_model_path",
    "row_start",
    "row_end",
    "col_start",
    "col_end",
    "updated_initializers",
    "source_cost",
    "output_cost",
    "cost_delta",
    "source_file_size_bytes",
    "output_file_size_bytes",
    "file_size_delta",
]


def _range_for_mode(mode: str) -> tuple[int, int, int, int]:
    """Return row_start, row_end, col_start, col_end for a pruning mode."""
    if mode == "conservative":
        # Labelled coverage is rows 6..16 and cols 6..20. Keep a one-cell row
        # and left-column margin where possible, while preserving full col max.
        return 5, 18, 5, 21
    if mode == "observed":
        return 6, 17, 6, 21
    raise ValueError(f"unsupported mode: {mode}")


def _make_i64_scalar(value: int, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.int64), name=name)


def _replace_initializer(
    initializer: onnx.TensorProto,
    new_array: np.ndarray,
) -> None:
    initializer.CopyFrom(numpy_helper.from_array(new_array, name=initializer.name))


def _insert_index_sub_nodes(
    model: onnx.ModelProto,
    row_start: int,
    col_start: int,
) -> tuple[str, str, list[onnx.TensorProto]]:
    col_adjusted = "task209_col_prior_idx_adjusted"
    row_adjusted = "task209_row_prior_idx_adjusted"
    col_offset = "task209_col_prior_offset"
    row_offset = "task209_row_prior_offset"
    new_initializers = [
        _make_i64_scalar(col_start, col_offset),
        _make_i64_scalar(row_start, row_offset),
    ]
    new_nodes = [
        helper.make_node(
            "Sub",
            [COL_INDEX, col_offset],
            [col_adjusted],
            name="Task209ColPriorIndexAdjust",
        ),
        helper.make_node(
            "Sub",
            [ROW_INDEX, row_offset],
            [row_adjusted],
            name="Task209RowPriorIndexAdjust",
        ),
    ]

    inserted = False
    rewritten_nodes: list[onnx.NodeProto] = []
    for node in model.graph.node:
        is_prior_gather = node.op_type == "Gather" and node.input and node.input[0] in {*COL_PRIORS, *ROW_PRIORS}
        if is_prior_gather and not inserted:
            rewritten_nodes.extend(new_nodes)
            inserted = True
        if is_prior_gather:
            if node.input[0] in COL_PRIORS:
                node.input[1] = col_adjusted
            elif node.input[0] in ROW_PRIORS:
                node.input[1] = row_adjusted
        rewritten_nodes.append(node)

    if not inserted:
        raise ValueError("no prior Gather nodes found")
    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)
    return row_adjusted, col_adjusted, new_initializers


def prune_task209_prior_ranges(
    source_model: str,
    output_model: str,
    mode: str = "conservative",
) -> dict[str, Any]:
    """Slice task209 row/col prior tables and adjust their Gather indices."""
    source_path = Path(source_model)
    output_path = Path(output_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_start, row_end, col_start, col_end = _range_for_mode(mode)
    model = onnx.load(str(source_path))
    onnx.checker.check_model(model)
    source_cost = estimate_model_cost(str(source_path))

    updated = 0
    expected_names = {*COL_PRIORS, *ROW_PRIORS}
    seen: set[str] = set()
    for initializer in model.graph.initializer:
        if initializer.name not in expected_names:
            continue
        array = numpy_helper.to_array(initializer)
        if array.ndim != 3 or array.shape[0] != 21:
            raise ValueError(f"unexpected prior shape for {initializer.name}: {array.shape}")
        if initializer.name in COL_PRIORS:
            new_array = array[col_start:col_end]
        else:
            new_array = array[row_start:row_end]
        _replace_initializer(initializer, new_array)
        updated += 1
        seen.add(initializer.name)

    missing = sorted(expected_names - seen)
    if missing:
        raise ValueError(f"missing prior initializers: {missing}")
    _, _, new_initializers = _insert_index_sub_nodes(model, row_start=row_start, col_start=col_start)
    model.graph.initializer.extend(new_initializers)
    del model.graph.value_info[:]

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path))
    output_cost = estimate_model_cost(str(output_path))
    return {
        "mode": mode,
        "source_model_path": str(source_path),
        "output_model_path": str(output_path),
        "row_start": row_start,
        "row_end": row_end,
        "col_start": col_start,
        "col_end": col_end,
        "updated_initializers": updated,
        "source_cost": int(source_cost["estimated_cost"]),
        "output_cost": int(output_cost["estimated_cost"]),
        "cost_delta": int(output_cost["estimated_cost"]) - int(source_cost["estimated_cost"]),
        "source_file_size_bytes": int(source_cost["file_size_bytes"]),
        "output_file_size_bytes": int(output_cost["file_size_bytes"]),
        "file_size_delta": int(output_cost["file_size_bytes"]) - int(source_cost["file_size_bytes"]),
    }


def build_task209_candidates(
    model_dir: str,
    output_dir: str,
    report_path: str,
    modes: list[str],
) -> dict[str, Any]:
    """Build task209 prior-range candidates and write a CSV report."""
    source = Path(model_dir) / "task209.onnx"
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for mode in modes:
        name = "PriorRangeConservative" if mode == "conservative" else "PriorRangeObserved"
        destination = output_root / f"task209_{name}.onnx"
        row = prune_task209_prior_ranges(str(source), str(destination), mode=mode)
        rows.append({"task_id": "task209", **row})

    with Path(report_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "task_id": "task209",
        "modes": modes,
        "report_path": report_path,
        "output_dir": output_dir,
        "candidate_count": len(rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_modes(raw: str) -> list[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    return modes or ["conservative"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/onnx")
    parser.add_argument("--output-dir", default="outputs/candidates/task209_prior_range_prune")
    parser.add_argument("--report", default="outputs/reports/task209_prior_range_prune.csv")
    parser.add_argument("--modes", default="conservative,observed")
    args = parser.parse_args()
    build_task209_candidates(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        modes=_parse_modes(args.modes),
    )


if __name__ == "__main__":
    main()
