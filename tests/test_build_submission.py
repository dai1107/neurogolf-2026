from __future__ import annotations

import csv
import json
import zipfile

from src.build_submission import build_submission
from src.inspect_submission import inspect_submission


def _write_task(path, train) -> None:
    path.write_text(json.dumps({"train": train}), encoding="utf-8")


def test_build_submission_includes_only_validated_models(tmp_path) -> None:
    data_dir = tmp_path / "tasks"
    out_dir = tmp_path / "onnx"
    candidate_dir = tmp_path / "candidates"
    log_dir = tmp_path / "logs"
    report = tmp_path / "reports" / "summary.csv"
    zip_path = tmp_path / "submission.zip"
    data_dir.mkdir()

    _write_task(
        data_dir / "task001.json",
        [{"input": [[1, 0], [2, 3]], "output": [[1, 0], [2, 3]]}],
    )
    _write_task(
        data_dir / "task002.json",
        [{"input": [[1, 1]], "output": [[2, 3]]}],
    )

    summary = build_submission(
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        candidate_dir=str(candidate_dir),
        log_dir=str(log_dir),
        report=str(report),
        zip_path=str(zip_path),
    )

    assert summary["total_tasks"] == 2
    assert summary["solved_tasks"] == 1
    assert zip_path.is_file()
    assert report.is_file()
    assert (log_dir / "task001.json").is_file()
    assert (log_dir / "task002.json").is_file()

    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["task001.onnx"]
    assert inspect_submission(str(zip_path)) == {"passed": True, "num_models": 1}

    with report.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["solved", "failed"]
