from __future__ import annotations

import pytest

from src.task209_prior_range_prune import _range_for_mode


def test_range_for_mode_keeps_margin_for_conservative() -> None:
    assert _range_for_mode("conservative") == (5, 18, 5, 21)


def test_range_for_mode_uses_observed_bounds() -> None:
    assert _range_for_mode("observed") == (6, 17, 6, 21)


def test_range_for_mode_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported mode"):
        _range_for_mode("bad")
