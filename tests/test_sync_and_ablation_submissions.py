from __future__ import annotations

import hashlib
import json
import zipfile

import pytest

from src.build_ablation_submissions import build_ablation_submissions
from src.onnx_builders import build_color_map_model, build_identity_model
from src.sync_model_bank_from_submission import sync_model_bank_from_submission


def _write_task(path, color: int) -> None:
    path.write_text(
        json.dumps({"train": [{"input": [[color]], "output": [[color]]}]}),
        encoding="utf-8",
    )


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sync_model_bank_from_submission_extracts_valid_zip(tmp_path) -> None:
    data_dir = tmp_path / "tasks"
    source_dir = tmp_path / "source"
    model_dir = tmp_path / "models"
    zip_path = tmp_path / "submission.zip"
    data_dir.mkdir()
    source_dir.mkdir()

    _write_task(data_dir / "task001.json", 1)
    _write_task(data_dir / "task002.json", 2)
    build_identity_model(str(source_dir / "task001.onnx"))
    build_identity_model(str(source_dir / "task002.onnx"))

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(source_dir / "task001.onnx", "task001.onnx")
        archive.write(source_dir / "task002.onnx", "task002.onnx")

    summary = sync_model_bank_from_submission(
        zip_path=str(zip_path),
        data_dir=str(data_dir),
        model_dir=str(model_dir),
    )

    assert summary["copied_models"] == 2
    assert _sha256(model_dir / "task001.onnx") == _sha256(source_dir / "task001.onnx")
    assert _sha256(model_dir / "task002.onnx") == _sha256(source_dir / "task002.onnx")


def test_sync_model_bank_from_submission_rejects_missing_task(tmp_path) -> None:
    data_dir = tmp_path / "tasks"
    source_dir = tmp_path / "source"
    model_dir = tmp_path / "models"
    zip_path = tmp_path / "submission.zip"
    data_dir.mkdir()
    source_dir.mkdir()

    _write_task(data_dir / "task001.json", 1)
    _write_task(data_dir / "task002.json", 2)
    build_identity_model(str(source_dir / "task001.onnx"))

    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.write(source_dir / "task001.onnx", "task001.onnx")

    with pytest.raises(ValueError, match="missing"):
        sync_model_bank_from_submission(
            zip_path=str(zip_path),
            data_dir=str(data_dir),
            model_dir=str(model_dir),
        )

    assert not model_dir.exists()


def test_build_ablation_submissions_replaces_one_entry(tmp_path) -> None:
    base_dir = tmp_path / "base"
    candidate_dir = tmp_path / "candidates"
    output_dir = tmp_path / "ablation"
    report = tmp_path / "reports" / "ablation.csv"
    base_zip = tmp_path / "submission.zip"
    base_dir.mkdir()
    candidate_dir.mkdir()

    build_identity_model(str(base_dir / "task001.onnx"))
    build_identity_model(str(base_dir / "task002.onnx"))
    build_color_map_model({1: 2}, str(candidate_dir / "task001_ColorMapRule.onnx"))

    with zipfile.ZipFile(base_zip, "w") as archive:
        archive.write(base_dir / "task001.onnx", "task001.onnx")
        archive.write(base_dir / "task002.onnx", "task002.onnx")

    summary = build_ablation_submissions(
        base_zip=str(base_zip),
        candidate_dir=str(candidate_dir),
        output_dir=str(output_dir),
        report_path=str(report),
    )

    candidate_zip = output_dir / "task001_ColorMapRule.zip"
    assert summary["candidate_count"] == 1
    assert summary["valid_zip_count"] == 1
    assert candidate_zip.is_file()

    with zipfile.ZipFile(base_zip) as base, zipfile.ZipFile(candidate_zip) as candidate:
        assert sorted(candidate.namelist()) == ["task001.onnx", "task002.onnx"]
        assert hashlib.sha256(candidate.read("task001.onnx")).digest() != hashlib.sha256(
            base.read("task001.onnx")
        ).digest()
        assert hashlib.sha256(candidate.read("task002.onnx")).digest() == hashlib.sha256(
            base.read("task002.onnx")
        ).digest()
