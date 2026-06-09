from __future__ import annotations

import json
from pathlib import Path

from src.cost_estimator import check_forbidden_ops, check_static_shapes, estimate_model_cost
from src.task076_template_matcher import (
    build_task076_template_model,
    extract_template_rules,
    probe_task,
    task076_template_transform,
)
from src.validate_onnx_model import validate_cases


def _task076() -> dict:
    return json.loads(Path("task/task076.json").read_text(encoding="utf-8"))


def test_task076_template_probe_matches_labelled_splits() -> None:
    task = _task076()
    row = probe_task(task)

    assert row["train_pass"] == "3/3"
    assert row["test_pass"] == "1/1"
    assert row["arc_gen_pass"] == "262/262"
    assert row["num_rules"] > 0
    for split in ("train", "test", "arc-gen"):
        for case in task.get(split, []):
            if "output" in case:
                assert task076_template_transform(case["input"]) == case["output"]


def test_task076_template_builder_validates_train(tmp_path) -> None:
    task = _task076()
    model_path = tmp_path / "task076.onnx"

    build_task076_template_model(task, str(model_path), mode="conservative")

    assert extract_template_rules(task)
    assert validate_cases(str(model_path), task["train"])["passed"]
    assert check_forbidden_ops(str(model_path))["passed"]
    assert check_static_shapes(str(model_path))["passed"]
    assert estimate_model_cost(str(model_path))["file_size_ok"]
