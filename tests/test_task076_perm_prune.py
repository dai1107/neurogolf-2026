from __future__ import annotations

import json
from pathlib import Path

import onnx

from src.cost_estimator import check_forbidden_ops, check_static_shapes, estimate_model_cost
from src.validate_onnx_model import validate_cases


def _task076() -> dict:
    return json.loads(Path("task/task076.json").read_text(encoding="utf-8"))


def test_task076_perm_gather_exact_promoted_model_validates_train() -> None:
    task = _task076()
    model_path = Path("outputs/onnx/task076.onnx")
    model = onnx.load(str(model_path))
    initializer_names = {initializer.name for initializer in model.graph.initializer}

    assert any("PermGatherIdx" in name for name in initializer_names), f"PermGatherIdx not found in {initializer_names}"
    assert "perm_flat" not in initializer_names
    assert validate_cases(str(model_path), task["train"])["passed"]
    assert check_forbidden_ops(str(model_path))["passed"]
    assert check_static_shapes(str(model_path))["passed"]
    assert estimate_model_cost(str(model_path))["file_size_ok"]
