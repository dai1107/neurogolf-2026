from __future__ import annotations

from pathlib import Path

import onnx
import pytest

from src.cost_estimator import check_static_shapes
from src.evaluate_onnx_candidate import evaluate
from src.repair_archive_padding import repair_negative_conv_pads, repair_task277_static_pads


def test_task277_static_pad_repair_validates_archive_model(tmp_path) -> None:
    source = Path("archive/task277.onnx")
    task = Path("task/task277.json")
    if not source.is_file() or not task.is_file():
        pytest.skip("archive task277 fixture is not available")

    repaired = tmp_path / "task277.onnx"
    repair_task277_static_pads(source, repaired)

    assert check_static_shapes(str(repaired)) == {"passed": True, "failures": []}
    report = evaluate(str(repaired), str(task))
    assert report["valid"] is True


def test_negative_conv_pad_repair_validates_archive_model(tmp_path) -> None:
    source = Path("archive/task042.onnx")
    task = Path("task/task042.json")
    if not source.is_file() or not task.is_file():
        pytest.skip("archive task042 fixture is not available")

    repaired = tmp_path / "task042.onnx"
    rewritten = repair_negative_conv_pads(source, repaired)

    assert rewritten > 0
    model = onnx.load(str(repaired))
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        for attribute in node.attribute:
            if attribute.name == "pads":
                assert all(value >= 0 for value in attribute.ints)
    assert check_static_shapes(str(repaired)) == {"passed": True, "failures": []}
    report = evaluate(str(repaired), str(task))
    assert report["valid"] is True
