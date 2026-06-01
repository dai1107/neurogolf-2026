from __future__ import annotations

import onnx

from src.build_single_conv_model import build_single_layer_conv2d_model
from src.cost_estimator import check_forbidden_ops, estimate_model_cost
from src.encoding import DEFAULT_SHAPE, grid_to_onehot, onehot_to_grid
from src.validate_onnx_model import run_model, validate_case


def test_grid_encoding_roundtrip_and_padding() -> None:
    grid = [
        [1, 2],
        [3, 0],
    ]

    tensor = grid_to_onehot(grid)

    assert tensor.shape == DEFAULT_SHAPE
    assert tensor.dtype.name == "float32"
    assert tensor[0, 0, 1, 1] == 1.0
    assert tensor[:, :, 2:, :].sum() == 0.0
    assert tensor[:, :, :, 2:].sum() == 0.0
    assert onehot_to_grid(tensor, 2, 2) == grid


def test_task000_model_can_be_created_checked_and_run(tmp_path) -> None:
    model_path = tmp_path / "task000.onnx"

    build_single_layer_conv2d_model(str(model_path))

    assert model_path.is_file()
    onnx.checker.check_model(str(model_path))
    assert check_forbidden_ops(str(model_path)) == {
        "passed": True,
        "forbidden_ops_found": [],
    }

    cost = estimate_model_cost(str(model_path))
    assert cost["num_parameters"] == 900
    assert cost["initializer_memory_bytes"] == 3600
    assert cost["file_size_ok"] is True

    input_tensor = grid_to_onehot([[1, 5], [5, 5]])
    output_tensor = run_model(str(model_path), input_tensor)
    assert output_tensor.shape == DEFAULT_SHAPE

    result = validate_case(
        str(model_path),
        input_grid=[[1, 5], [5, 5]],
        expected_grid=[[1, 5], [5, 0]],
    )
    assert result["passed"] is True
    assert result["num_mismatched_cells"] == 0
