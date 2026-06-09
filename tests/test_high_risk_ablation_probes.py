from __future__ import annotations

from src.high_risk_ablation_probes import (
    PROBES,
    _complete_line_patterns,
    _copy_marker_matched_source_objects_to_sparse_panel,
)


def test_horizontal_zero_run_probe_is_not_registered() -> None:
    assert "horizontal_zero_runs_by_marker_length" not in PROBES
    assert PROBES["line_pattern_completion"][2] == "no"


def test_line_pattern_completion_is_probe_only_for_blank_runs() -> None:
    grid = [
        [2, 2, 0, 0],
        [0, 0, 5, 5],
        [5, 5, 5, 5],
        [0, 5, 5, 5],
    ]

    assert _complete_line_patterns(grid) == [
        [2, 2, 2, 2],
        [2, 2, 5, 5],
        [5, 5, 5, 5],
        [0, 5, 5, 5],
    ]


def test_panel_transfer_rejects_marker_only_source_component() -> None:
    grid = [
        [7, 7, 7, 7],
        [7, 2, 7, 7],
        [0, 0, 0, 0],
        [0, 2, 0, 0],
    ]

    assert _copy_marker_matched_source_objects_to_sparse_panel(grid) == grid
