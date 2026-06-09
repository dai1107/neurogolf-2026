"""Build a task233 ONNX candidate by pruning template-combo rows.

The existing task233 graph enumerates all 5^5 template assignments.  The
board-hole paste probe supports a stricter semantic assumption: each template
is used at most once.  This module keeps only combo rows with unique template
indices and slices every row-aligned initializer in lockstep.
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


DEFAULT_SOURCE = "outputs/onnx/task233.onnx"
DEFAULT_OUTPUT = (
    "outputs/candidates/task233_board_hole_paste/"
    "task233_BoardHolePasteComboPruned.onnx"
)
DEFAULT_REPORT = "outputs/reports/task233_combo_prune_report.csv"
DEFAULT_MODE = "at_most_two_distinct"
MODE_CHOICES = (
    "permutation",
    "at_most_two_distinct",
    "one_nonzero",
    "observed_labelled",
)

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


def _row_indices_for_mode(combo: np.ndarray, mode: str) -> np.ndarray:
    if combo.ndim != 2:
        raise ValueError(f"combo must be rank 2, got shape {combo.shape}")
    if mode == "permutation":
        keep = [
            index
            for index, row in enumerate(combo)
            if len(set(row.tolist())) == combo.shape[1]
        ]
    elif mode == "at_most_two_distinct":
        keep = [
            index
            for index, row in enumerate(combo)
            if len(set(row.tolist())) <= 2
        ]
    elif mode == "one_nonzero":
        keep = [
            index
            for index, row in enumerate(combo)
            if int(np.count_nonzero(row)) <= 1
        ]
    elif mode == "observed_labelled":
        observed = {(0, 0, 0, 0, 0), (0, 1, 0, 0, 0)}
        keep = [
            index
            for index, row in enumerate(combo)
            if tuple(int(value) for value in row.tolist()) in observed
        ]
    else:
        raise ValueError(f"unknown pruning mode: {mode}")
    if not keep:
        raise ValueError(f"{mode} pruning would remove every combo row")
    return np.asarray(keep, dtype=np.int64)


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


def _replace_constant_value(
    node: onnx.NodeProto,
    attr: onnx.AttributeProto,
    new_array: np.ndarray,
    action: str,
    rows: list[dict[str, Any]],
) -> None:
    old_array = numpy_helper.to_array(onnx.helper.get_attribute_value(attr))
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
            "action": action,
        }
    )


def prune_task233_combo(
    source_model: str = DEFAULT_SOURCE,
    output_model: str = DEFAULT_OUTPUT,
    report_path: str = DEFAULT_REPORT,
    mode: str = DEFAULT_MODE,
) -> dict[str, Any]:
    """Prune row-aligned task233 combo tables and write a candidate model."""
    if mode not in MODE_CHOICES:
        raise ValueError(f"mode must be one of {', '.join(MODE_CHOICES)}, got {mode}")
    source_path = Path(source_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")

    model = onnx.load(str(source_path))
    arrays = _initializer_arrays(model)
    if "combo" not in arrays:
        raise ValueError("source model is missing the combo initializer")
    if "comborange" not in arrays:
        raise ValueError("source model is missing the comborange initializer")

    combo = arrays["combo"]
    if combo.shape != (3125, 5):
        raise ValueError(f"unexpected combo shape: {combo.shape}")
    row_count = combo.shape[0]
    keep_indices = _row_indices_for_mode(combo, mode)
    kept_count = int(keep_indices.size)

    rows: list[dict[str, Any]] = []
    updated_initializers = 0
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        if initializer.name == "/Where_96_output_0":
            _replace_initializer(
                initializer,
                np.asarray([kept_count, 1], dtype=array.dtype),
                "row_shape_updated",
                rows,
            )
            updated_initializers += 1
            continue
        if initializer.name == "/Constant_575_output_0":
            _replace_initializer(
                initializer,
                np.asarray([kept_count], dtype=array.dtype),
                "row_shape_updated",
                rows,
            )
            updated_initializers += 1
            continue
        if array.ndim == 0 or array.shape[0] != row_count:
            continue

        if initializer.name == "comborange":
            new_array = np.arange(kept_count, dtype=array.dtype)
            action = "renumbered"
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
            if attr.name != "value":
                continue
            array = numpy_helper.to_array(onnx.helper.get_attribute_value(attr))
            if array.shape == (row_count, 1) and np.array_equal(
                array.reshape(-1),
                np.arange(row_count, dtype=array.dtype),
            ):
                new_array = np.arange(kept_count, dtype=array.dtype).reshape(kept_count, 1)
                _replace_constant_value(node, attr, new_array, "renumbered", rows)
                updated_constant_nodes += 1
            elif array.shape == (row_count, combo.shape[1]) and np.count_nonzero(array) == 0:
                new_array = np.zeros((kept_count, combo.shape[1]), dtype=array.dtype)
                _replace_constant_value(node, attr, new_array, "zero_table_resized", rows)
                updated_constant_nodes += 1

    if "combo" not in {row["value_name"] for row in rows}:
        raise ValueError("combo was not pruned")
    if "comborange" not in {row["value_name"] for row in rows}:
        raise ValueError("comborange was not pruned")

    del model.graph.value_info[:]
    onnx.checker.check_model(model)

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

    summary = {
        "source_model": str(source_path),
        "output_model": str(output_path),
        "report_path": str(report),
        "mode": mode,
        "original_combo_rows": row_count,
        "kept_combo_rows": kept_count,
        "updated_initializers": updated_initializers,
        "updated_constant_nodes": updated_constant_nodes,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=MODE_CHOICES)
    args = parser.parse_args()
    prune_task233_combo(
        source_model=args.source,
        output_model=args.output,
        report_path=args.report,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
