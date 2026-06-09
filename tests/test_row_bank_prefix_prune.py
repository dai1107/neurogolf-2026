from __future__ import annotations

import csv

import numpy as np

from src.row_bank_prefix_prune import (
    _bank_ids,
    _prefix_for_mode,
    _read_observed,
)


def test_bank_ids_finds_matching_row_col_pairs() -> None:
    arrays = {
        "row_bank_0": np.arange(4),
        "col_bank_0": np.arange(4),
        "row_bank_2": np.arange(3),
        "other": np.arange(3),
    }

    assert _bank_ids(arrays) == [0]


def test_prefix_for_mode_is_ordered_by_risk() -> None:
    observed = {2, 10}

    conservative = _prefix_for_mode(100, observed, "conservative")
    medium = _prefix_for_mode(100, observed, "medium")
    observed_prefix = _prefix_for_mode(100, observed, "observed")

    assert conservative >= medium >= observed_prefix
    assert observed_prefix == 11
    assert conservative == 90
    assert medium == 75


def test_read_observed_groups_selected_indices_by_bank(tmp_path) -> None:
    report = tmp_path / "observed.csv"
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "split",
                "case_index",
                "bank_id",
                "selected_index",
                "selected_row",
                "selected_col",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "split": "train",
                "case_index": 0,
                "bank_id": 1,
                "selected_index": 4,
                "selected_row": 2,
                "selected_col": 1,
            }
        )
        writer.writerow(
            {
                "split": "test",
                "case_index": 0,
                "bank_id": 1,
                "selected_index": 5,
                "selected_row": 2,
                "selected_col": 2,
            }
        )

    assert _read_observed(str(report)) == {1: {4, 5}}
