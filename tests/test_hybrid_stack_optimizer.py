from __future__ import annotations

import csv
import zipfile

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.hybrid_stack_optimizer import (
    OPTIMIZE_FIELDS,
    build_lane_ablations,
    build_merged_submission,
    optimize_model_equivalent,
)
from src.inspect_submission import inspect_submission
from src.onnx_builders import build_color_map_model, build_identity_model


def _save_model(path, graph) -> None:
    model = helper.make_model(
        graph,
        producer_name="test",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, str(path))


def test_optimize_model_equivalent_prunes_dead_nodes_and_initializers(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Add", ["input", "Live"], ["output"], name="live_add"),
            helper.make_node("Add", ["input", "Dead"], ["dead"], name="dead_add"),
        ],
        name="dead_graph",
        inputs=[x],
        outputs=[y],
        initializer=[
            numpy_helper.from_array(np.asarray([[1.0]], dtype=np.float32), name="Live"),
            numpy_helper.from_array(np.asarray([[2.0]], dtype=np.float32), name="Dead"),
        ],
    )
    _save_model(input_path, graph)

    report = optimize_model_equivalent(str(input_path), str(output_path), passes=("dead",))
    output = onnx.load(str(output_path))

    assert report["removed_dead_nodes"] == 1
    assert report["removed_unused_initializers"] == 1
    assert [node.name for node in output.graph.node] == ["live_add"]
    assert [initializer.name for initializer in output.graph.initializer] == ["Live"]
    onnx.checker.check_model(output)


def test_optimize_model_equivalent_deduplicates_identical_initializers(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    one = np.asarray([[1.0]], dtype=np.float32)
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Add", ["input", "A"], ["middle"], name="a"),
            helper.make_node("Add", ["middle", "B"], ["output"], name="b"),
        ],
        name="dedup_graph",
        inputs=[x],
        outputs=[y],
        initializer=[
            numpy_helper.from_array(one, name="A"),
            numpy_helper.from_array(one, name="B"),
        ],
    )
    _save_model(input_path, graph)

    report = optimize_model_equivalent(str(input_path), str(output_path), passes=("dedup",))
    output = onnx.load(str(output_path))

    assert report["deduplicated_initializers"] == 1
    assert [initializer.name for initializer in output.graph.initializer] == ["A"]
    assert output.graph.node[1].input[1] == "A"
    onnx.checker.check_model(output)


def test_optimize_model_equivalent_prunes_constant_gather_table(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [3, 2])
    table = np.arange(10, dtype=np.float32).reshape(5, 2)
    indices = np.asarray([2, 4, 2], dtype=np.int64)
    graph = helper.make_graph(
        nodes=[helper.make_node("Gather", ["Table", "Idx"], ["output"], name="gather", axis=0)],
        name="const_gather",
        inputs=[],
        outputs=[y],
        initializer=[
            numpy_helper.from_array(table, name="Table"),
            numpy_helper.from_array(indices, name="Idx"),
        ],
    )
    _save_model(input_path, graph)

    report = optimize_model_equivalent(
        str(input_path),
        str(output_path),
        passes=("const-gather",),
    )
    output = onnx.load(str(output_path))
    inits = {initializer.name: numpy_helper.to_array(initializer) for initializer in output.graph.initializer}

    assert report["constant_gather_tables_pruned"] == 1
    assert report["constant_gather_rows_removed"] == 3
    assert inits["Table"].shape == (2, 2)
    np.testing.assert_array_equal(inits["Table"], table[[2, 4]])
    np.testing.assert_array_equal(inits["Idx"], np.asarray([0, 1, 0], dtype=np.int64))
    onnx.checker.check_model(output)


def _write_hybrid_zip(zip_path, stack_dir, task_ids: list[str]) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        for lane in ("base_submission", "overrides"):
            for task_id in task_ids:
                archive.write(stack_dir / lane / f"{task_id}.onnx", f"{lane}/{task_id}.onnx")


def test_build_lane_ablations_replaces_only_selected_lane(tmp_path) -> None:
    stack_dir = tmp_path / "stack"
    candidate_dir = tmp_path / "candidates"
    report_path = tmp_path / "reports" / "candidates.csv"
    ablation_report = tmp_path / "reports" / "ablations.csv"
    ablation_dir = tmp_path / "ablations"
    for path in (
        stack_dir / "base_submission",
        stack_dir / "overrides",
        candidate_dir / "base_submission",
        report_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)

    build_identity_model(str(stack_dir / "base_submission" / "task001.onnx"))
    build_identity_model(str(stack_dir / "overrides" / "task001.onnx"))
    build_identity_model(str(stack_dir / "base_submission" / "task002.onnx"))
    build_identity_model(str(stack_dir / "overrides" / "task002.onnx"))
    build_color_map_model({1: 2}, str(candidate_dir / "base_submission" / "task001.onnx"))

    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPTIMIZE_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "task001",
                "lane": "base_submission",
                "output_model_path": str(candidate_dir / "base_submission" / "task001.onnx"),
                "file_size_delta": "-10",
                "initializer_bytes_delta": "-10",
                "candidate_valid": True,
            }
        )

    base_zip = tmp_path / "base.zip"
    _write_hybrid_zip(base_zip, stack_dir, ["task001", "task002"])
    summary = build_lane_ablations(
        base_zip=str(base_zip),
        candidate_report=str(report_path),
        output_dir=str(ablation_dir),
        report_path=str(ablation_report),
        max_candidates=10,
    )

    assert summary["valid_zip_count"] == 1
    candidate_zip = ablation_dir / "task001_base_EquivOptimized" / "submission.zip"
    assert inspect_submission(str(candidate_zip), layout="hybrid_stack")["num_models"] == 4
    with zipfile.ZipFile(base_zip) as base, zipfile.ZipFile(candidate_zip) as candidate:
        assert candidate.read("base_submission/task001.onnx") != base.read("base_submission/task001.onnx")
        assert candidate.read("overrides/task001.onnx") == base.read("overrides/task001.onnx")
        assert candidate.read("base_submission/task002.onnx") == base.read("base_submission/task002.onnx")


def test_build_merged_submission_applies_all_selected_replacements(tmp_path) -> None:
    stack_dir = tmp_path / "stack"
    candidate_dir = tmp_path / "candidates"
    report_path = tmp_path / "reports" / "candidates.csv"
    merge_report = tmp_path / "reports" / "merge.csv"
    output_zip = tmp_path / "merged" / "submission.zip"
    for path in (
        stack_dir / "base_submission",
        stack_dir / "overrides",
        candidate_dir / "base_submission",
        candidate_dir / "overrides",
        report_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)

    build_identity_model(str(stack_dir / "base_submission" / "task001.onnx"))
    build_identity_model(str(stack_dir / "overrides" / "task001.onnx"))
    build_identity_model(str(stack_dir / "base_submission" / "task002.onnx"))
    build_identity_model(str(stack_dir / "overrides" / "task002.onnx"))
    build_color_map_model({1: 2}, str(candidate_dir / "base_submission" / "task001.onnx"))
    build_color_map_model({2: 3}, str(candidate_dir / "overrides" / "task002.onnx"))

    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPTIMIZE_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "task001",
                "lane": "base_submission",
                "output_model_path": str(candidate_dir / "base_submission" / "task001.onnx"),
                "estimated_cost_delta": "-20",
                "file_size_delta": "-10",
                "initializer_bytes_delta": "-10",
                "candidate_valid": True,
            }
        )
        writer.writerow(
            {
                "task_id": "task002",
                "lane": "overrides",
                "output_model_path": str(candidate_dir / "overrides" / "task002.onnx"),
                "estimated_cost_delta": "-30",
                "file_size_delta": "-15",
                "initializer_bytes_delta": "-15",
                "candidate_valid": True,
            }
        )

    base_zip = tmp_path / "base.zip"
    _write_hybrid_zip(base_zip, stack_dir, ["task001", "task002"])
    summary = build_merged_submission(
        base_zip=str(base_zip),
        candidate_report=str(report_path),
        output_zip=str(output_zip),
        report_path=str(merge_report),
    )

    assert summary["merged_replacements"] == 2
    assert summary["total_estimated_cost_delta"] == -50
    assert inspect_submission(str(output_zip), layout="hybrid_stack")["num_models"] == 4
    with zipfile.ZipFile(base_zip) as base, zipfile.ZipFile(output_zip) as merged:
        assert merged.read("base_submission/task001.onnx") != base.read("base_submission/task001.onnx")
        assert merged.read("overrides/task002.onnx") != base.read("overrides/task002.onnx")
        assert merged.read("overrides/task001.onnx") == base.read("overrides/task001.onnx")
        assert merged.read("base_submission/task002.onnx") == base.read("base_submission/task002.onnx")
