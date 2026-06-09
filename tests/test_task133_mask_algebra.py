from __future__ import annotations

import json
from pathlib import Path

from src.cost_estimator import check_forbidden_ops, check_static_shapes, estimate_model_cost
from src.task133_mask_algebra import (
    build_task133_mask_algebra_model,
    probe_task,
    task133_mask_algebra_transform,
)
from src.validate_onnx_model import validate_cases


def _task133() -> dict:
    return json.loads(Path("task/task133.json").read_text(encoding="utf-8"))


def test_task133_mask_algebra_probe_matches_labelled_splits() -> None:
    task = _task133()
    row = probe_task(task)

    assert row["train_pass"] == "4/4"
    assert row["test_pass"] == "1/1"
    assert row["arc_gen_pass"] == "262/262"
    assert row["num_conditions"] <= 5
    for split in ("train", "test", "arc-gen"):
        for case in task.get(split, []):
            if "output" in case:
                assert task133_mask_algebra_transform(case["input"]) == case["output"]


def test_task133_mask_algebra_builder_validates_train(tmp_path) -> None:
    task = _task133()
    model_path = tmp_path / "task133.onnx"

    build_task133_mask_algebra_model(str(model_path))

    assert validate_cases(str(model_path), task["train"])["passed"]
    assert check_forbidden_ops(str(model_path))["passed"]
    assert check_static_shapes(str(model_path))["passed"]
    assert estimate_model_cost(str(model_path))["file_size_ok"]
