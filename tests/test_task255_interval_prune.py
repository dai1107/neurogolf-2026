from __future__ import annotations

import numpy as np

from src.task255_interval_prune import (
    SAFE_DROP_ROWS,
    _canonical_interval_for_row,
    _close_under_index_tables,
    _map_observed_rows_to_source,
    _validate_index_references_are_kept,
)


def test_close_under_index_tables_adds_referenced_rows() -> None:
    keep = {4}
    up_idx = np.asarray([0, 0, 1, 2, 3], dtype=np.int64)
    dn_idx = np.asarray([1, 2, 3, 4, 4], dtype=np.int64)

    assert _close_under_index_tables(keep, up_idx, dn_idx) == {0, 1, 2, 3, 4}


def test_validate_index_references_rejects_dangling_reference() -> None:
    keep = {0, 2}
    up_idx = np.asarray([0, 0, 2], dtype=np.int64)
    dn_idx = np.asarray([1, 2, 2], dtype=np.int64)

    try:
        _validate_index_references_are_kept(keep, up_idx, dn_idx)
    except ValueError as exc:
        assert "dangling index references" in str(exc)
    else:
        raise AssertionError("expected dangling index reference rejection")


def test_safe_drop_rows_are_explicitly_documented() -> None:
    assert SAFE_DROP_ROWS == frozenset(
        {31, 34, 57, 60, 61, 63, 85, 88, 89, 91, 448, 453, 460}
    )


def test_canonical_interval_for_row_uses_contiguous_interval_order() -> None:
    assert _canonical_interval_for_row(0) == (0, 0)
    assert _canonical_interval_for_row(29) == (0, 29)
    assert _canonical_interval_for_row(30) == (1, 1)
    assert _canonical_interval_for_row(464) == (29, 29)


def test_map_observed_rows_to_source_handles_prepruned_rows() -> None:
    arrays = {
        "I0": np.asarray([0, 0, 1, 2], dtype=np.float16),
        "I1": np.asarray([0, 2, 1, 2], dtype=np.float16),
    }

    assert _map_observed_rows_to_source({0, 2, 30, 59}, arrays) == {0, 1, 2, 3}
