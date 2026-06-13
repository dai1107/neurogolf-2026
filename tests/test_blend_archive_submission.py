from __future__ import annotations

import json
import zipfile

from src.blend_archive_submission import blend_archive_submission
from src.inspect_submission import inspect_submission
from src.onnx_builders import build_color_map_model, build_identity_model


def _write_task(path, color: int) -> None:
    path.write_text(
        json.dumps({"train": [{"input": [[color]], "output": [[color]]}]}),
        encoding="utf-8",
    )


def test_blend_archive_submission_trusted_mode_selects_lowest_cost(tmp_path) -> None:
    data_dir = tmp_path / "tasks"
    archive_dir = tmp_path / "archive"
    current_dir = tmp_path / "current"
    blended_dir = tmp_path / "blended"
    report = tmp_path / "reports" / "blend.csv"
    zip_path = tmp_path / "submission.zip"
    data_dir.mkdir()
    archive_dir.mkdir()
    current_dir.mkdir()

    _write_task(data_dir / "task001.json", 1)
    _write_task(data_dir / "task002.json", 2)

    build_color_map_model({1: 2}, str(archive_dir / "task001.onnx"))
    build_identity_model(str(current_dir / "task001.onnx"))
    build_identity_model(str(archive_dir / "task002.onnx"))
    build_color_map_model({2: 3}, str(current_dir / "task002.onnx"))

    summary = blend_archive_submission(
        data_dir=str(data_dir),
        archive_dir=str(archive_dir),
        current_dir=str(current_dir),
        blended_dir=str(blended_dir),
        report_path=str(report),
        zip_path=str(zip_path),
        validation_mode="trusted",
    )

    assert summary["selected_tasks"] == 2
    assert summary["missing_tasks"] == 0
    assert summary["source_counts"] == {"archive": 1, "current": 1}
    assert summary["validation_mode"] == "trusted"
    assert inspect_submission(str(zip_path)) == {"passed": True, "num_models": 2}
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["task001.onnx", "task002.onnx"]

    forced_zip_path = tmp_path / "forced_submission.zip"
    forced_summary = blend_archive_submission(
        data_dir=str(data_dir),
        archive_dir=str(archive_dir),
        current_dir=str(current_dir),
        blended_dir=str(tmp_path / "forced_blended"),
        report_path=str(tmp_path / "reports" / "forced_blend.csv"),
        zip_path=str(forced_zip_path),
        force_archive_task_ids={"task001"},
        validation_mode="trusted",
    )

    assert forced_summary["source_counts"] == {"archive": 2}
    assert inspect_submission(str(forced_zip_path)) == {"passed": True, "num_models": 2}
