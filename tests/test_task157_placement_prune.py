from __future__ import annotations

import numpy as np

from src.task157_placement_prune import (
    ROW_COUNT,
    _keep_indices_for_mode,
    _slice_row_count_axes,
)


def test_slice_row_count_axes_slices_matching_axis() -> None:
    array = np.arange(ROW_COUNT, dtype=np.float32).reshape(1, ROW_COUNT)
    keep = np.asarray([0, 2, 4], dtype=np.int64)

    sliced = _slice_row_count_axes(array, keep)

    assert sliced is not None
    assert sliced.shape == (1, 3)
    assert sliced.tolist() == [[0.0, 2.0, 4.0]]


def test_component_mode_keeps_rows_sharing_observed_components() -> None:
    expand_idx = np.asarray([index % 4 for index in range(ROW_COUNT)], dtype=np.int64)

    keep = _keep_indices_for_mode(
        "conservative",
        {"expand_idx_983": expand_idx},
        observed_rows={1, 2},
        row_list=None,
    )

    assert {int(expand_idx[index]) for index in keep.tolist()} == {1, 2}
    assert {1, 2} <= set(keep.tolist())


def test_medium_mode_keeps_observed_local_offsets_across_observed_components() -> None:
    expand_idx = np.repeat(np.asarray([0, 1, 2], dtype=np.int64), 4)

    keep = _keep_indices_for_mode(
        "medium",
        {"expand_idx_983": expand_idx},
        observed_rows={1, 6},
        row_list=None,
    )

    assert set(keep.tolist()) == {1, 2, 5, 6}


def test_drop_list_rejects_observed_rows(tmp_path) -> None:
    row_list = tmp_path / "drop_rows.txt"
    row_list.write_text("2\n", encoding="utf-8")
    expand_idx = np.zeros((ROW_COUNT,), dtype=np.int64)

    try:
        _keep_indices_for_mode(
            "drop-list",
            {"expand_idx_983": expand_idx},
            observed_rows={2},
            row_list=str(row_list),
        )
    except ValueError as exc:
        assert "drops observed placement rows" in str(exc)
    else:
        raise AssertionError("expected observed-row drop rejection")
