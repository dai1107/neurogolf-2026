from __future__ import annotations

import pytest

from src.task101_template_radius_prune import _radius_value_name


def test_radius_value_name_uses_two_digit_tensor_name() -> None:
    assert _radius_value_name(2) == "R_02"
    assert _radius_value_name(15) == "R_15"


def test_radius_value_name_rejects_out_of_range_radius() -> None:
    with pytest.raises(ValueError, match="radius must be in 1..15"):
        _radius_value_name(0)
