from __future__ import annotations

import csv
import zipfile

from src.inspect_submission import inspect_submission
from src.onnx_builders import build_color_map_model, build_identity_model
from src.select_best_submission import SubmissionCandidate, select_best_submission


def _write_flat_zip(zip_path, source_dir, task_ids: list[str]) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        for task_id in task_ids:
            archive.write(source_dir / f"{task_id}.onnx", f"{task_id}.onnx")


def _write_hybrid_zip(zip_path, source_dir, task_ids: list[str]) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        for folder in ("base_submission", "overrides"):
            for task_id in task_ids:
                archive.write(source_dir / f"{task_id}.onnx", f"{folder}/{task_id}.onnx")


def test_inspect_submission_accepts_hybrid_stack_layout(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    build_identity_model(str(model_dir / "task001.onnx"))
    build_color_map_model({1: 2}, str(model_dir / "task002.onnx"))

    zip_path = tmp_path / "submission.zip"
    _write_hybrid_zip(zip_path, model_dir, ["task001", "task002"])

    assert inspect_submission(str(zip_path)) == {
        "passed": True,
        "num_models": 4,
        "layout": "hybrid_stack",
        "num_task_ids": 2,
    }


def test_select_best_submission_uses_known_online_score_and_extracts_stack(tmp_path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    build_identity_model(str(model_dir / "task001.onnx"))
    build_color_map_model({1: 2}, str(model_dir / "task002.onnx"))

    old_zip = tmp_path / "old.zip"
    new_zip = tmp_path / "new.zip"
    output_zip = tmp_path / "selected" / "submission.zip"
    report = tmp_path / "reports" / "take_best.csv"
    extract_dir = tmp_path / "selected_stack"
    _write_flat_zip(old_zip, model_dir, ["task001", "task002"])
    _write_hybrid_zip(new_zip, model_dir, ["task001", "task002"])

    summary = select_best_submission(
        candidates=[
            SubmissionCandidate("old6275", str(old_zip), 6275.09),
            SubmissionCandidate("new6348", str(new_zip), 6348.56),
        ],
        output_zip=str(output_zip),
        report_path=str(report),
        extract_dir=str(extract_dir),
    )

    assert summary["selected_label"] == "new6348"
    assert summary["selected_layout"] == "hybrid_stack"
    assert inspect_submission(str(output_zip))["layout"] == "hybrid_stack"
    assert (extract_dir / "base_submission" / "task001.onnx").is_file()
    assert (extract_dir / "overrides" / "task002.onnx").is_file()

    with report.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["label"] for row in rows] == ["old6275", "new6348"]
    assert [row["selected"] for row in rows] == ["False", "True"]
