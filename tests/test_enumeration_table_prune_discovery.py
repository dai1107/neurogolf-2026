from __future__ import annotations

import csv

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.enumeration_table_prune_discovery import discover_enumeration_tables


def _write_model(path) -> None:
    table = np.arange(40, dtype=np.int64)
    combo = np.stack([table % 3, table % 5], axis=1).astype(np.int64)
    weights = np.ones((40, 2), dtype=np.float32)
    small = np.asarray([1, 2, 3], dtype=np.int64)

    graph = helper.make_graph(
        nodes=[
            helper.make_node("Identity", ["input"], ["output"]),
        ],
        name="enumeration_discovery_test",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])],
        initializer=[
            numpy_helper.from_array(table, name="row_range"),
            numpy_helper.from_array(combo, name="combo"),
            numpy_helper.from_array(weights, name="weights"),
            numpy_helper.from_array(small, name="small"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def test_discover_enumeration_tables_reports_shared_large_first_dim(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _write_model(model_dir / "task001.onnx")
    report = tmp_path / "reports" / "discovery.csv"

    summary = discover_enumeration_tables(
        model_dir=str(model_dir),
        task_ids=["task001"],
        report_path=str(report),
        min_first_dim=16,
        min_shared_values=3,
    )

    rows = list(csv.DictReader(report.open("r", newline="", encoding="utf-8")))
    assert summary["candidate_group_count"] == 1
    assert rows[0]["task_id"] == "task001"
    assert rows[0]["row_count"] == "40"
    assert rows[0]["shared_value_count"] == "3"
    assert rows[0]["has_arange_vector"] == "True"
    assert rows[0]["has_small_integer_table"] == "True"
