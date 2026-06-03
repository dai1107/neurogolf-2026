from __future__ import annotations

import json
import zipfile

import pytest

from src.build_current_model_submission import build_current_model_submission
from src.inspect_submission import inspect_submission
from src.onnx_builders import build_identity_model


def _write_task(path, train) -> None:
    path.write_text(json.dumps({"train": train}), encoding="utf-8")


def test_build_current_model_submission_validates_local_model_bank(tmp_path) -> None:
    data_dir = tmp_path / "tasks"
    model_dir = tmp_path / "models"
    validated_dir = tmp_path / "validated"
    report = tmp_path / "reports" / "current.csv"
    zip_path = tmp_path / "submission.zip"
    data_dir.mkdir()
    model_dir.mkdir()

    _write_task(
        data_dir / "task001.json",
        [{"input": [[1, 0], [2, 3]], "output": [[1, 0], [2, 3]]}],
    )
    build_identity_model(str(model_dir / "task001.onnx"))

    summary = build_current_model_submission(
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        validated_dir=str(validated_dir),
        report_path=str(report),
        zip_path=str(zip_path),
    )

    assert summary["total_tasks"] == 1
    assert summary["selected_tasks"] == 1
    assert summary["missing_or_invalid_tasks"] == 0
    assert (validated_dir / "task001.onnx").is_file()
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["task001.onnx"]
    assert inspect_submission(str(zip_path)) == {"passed": True, "num_models": 1}


def test_build_current_model_submission_rejects_incomplete_bank(tmp_path) -> None:
    data_dir = tmp_path / "tasks"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    model_dir.mkdir()

    _write_task(
        data_dir / "task001.json",
        [{"input": [[1]], "output": [[1]]}],
    )
    _write_task(
        data_dir / "task002.json",
        [{"input": [[2]], "output": [[2]]}],
    )
    build_identity_model(str(model_dir / "task001.onnx"))

    with pytest.raises(ValueError, match="1/2 valid"):
        build_current_model_submission(
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            validated_dir=str(tmp_path / "validated"),
            report_path=str(tmp_path / "reports" / "current.csv"),
            zip_path=str(tmp_path / "submission.zip"),
        )


def test_build_current_model_submission_rejects_validated_dir_equal_to_model_dir(tmp_path) -> None:
    data_dir = tmp_path / "tasks"
    model_dir = tmp_path / "models"
    data_dir.mkdir()
    model_dir.mkdir()

    _write_task(
        data_dir / "task001.json",
        [{"input": [[1]], "output": [[1]]}],
    )
    build_identity_model(str(model_dir / "task001.onnx"))

    with pytest.raises(ValueError, match="validated_dir must be different"):
        build_current_model_submission(
            data_dir=str(data_dir),
            model_dir=str(model_dir),
            validated_dir=str(model_dir),
            report_path=str(tmp_path / "reports" / "current.csv"),
            zip_path=str(tmp_path / "submission.zip"),
        )
