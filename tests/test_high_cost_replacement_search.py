from __future__ import annotations

import csv
import json

from src.diagnose_high_cost_tasks import diagnose_high_cost_tasks
from src.onnx_builders import build_color_map_model
from src.pattern_rules import IdentityRule
from src.search_symbolic_replacements import run_replacement_search


def _write_task(path, train) -> None:
    path.write_text(json.dumps({"train": train}), encoding="utf-8")


def _write_current_report(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task_id",
                "valid",
                "model_path",
                "estimated_cost",
                "file_size_bytes",
                "failure_reason",
                "selected_for_zip",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_diagnose_high_cost_tasks_writes_csv_and_markdown(tmp_path) -> None:
    data_dir = tmp_path / "task"
    data_dir.mkdir()
    _write_task(
        data_dir / "task001.json",
        [{"input": [[1, 0], [2, 3]], "output": [[1, 0], [2, 3]]}],
    )
    current_report = tmp_path / "reports" / "current.csv"
    _write_current_report(
        current_report,
        [
            {
                "task_id": "task001",
                "valid": "True",
                "model_path": "outputs/onnx/task001.onnx",
                "estimated_cost": "999",
                "file_size_bytes": "1234",
                "failure_reason": "",
                "selected_for_zip": "True",
            }
        ],
    )

    report = tmp_path / "reports" / "diagnosis.csv"
    analysis_dir = tmp_path / "reports" / "analysis"
    rows = diagnose_high_cost_tasks(
        current_report=str(current_report),
        data_dir=str(data_dir),
        report_path=str(report),
        analysis_dir=str(analysis_dir),
        top_k=1,
    )

    assert rows[0]["task_id"] == "task001"
    assert rows[0]["shape_relation"] == "same_shape"
    assert "color_map" in rows[0]["likely_rule_families"]
    assert report.is_file()
    assert (analysis_dir / "task001.md").is_file()


def test_replacement_search_marks_and_copies_cheaper_valid_candidate(tmp_path) -> None:
    data_dir = tmp_path / "task"
    model_dir = tmp_path / "models"
    candidate_dir = tmp_path / "candidates"
    data_dir.mkdir()
    model_dir.mkdir()
    _write_task(
        data_dir / "task001.json",
        [{"input": [[1, 0], [2, 3]], "output": [[1, 0], [2, 3]]}],
    )
    build_color_map_model({0: 0, 1: 1, 2: 2, 3: 3}, str(model_dir / "task001.onnx"))
    current_report = tmp_path / "reports" / "current.csv"
    _write_current_report(
        current_report,
        [
            {
                "task_id": "task001",
                "valid": "True",
                "model_path": str(model_dir / "task001.onnx"),
                "estimated_cost": "999",
                "file_size_bytes": "1234",
                "failure_reason": "",
                "selected_for_zip": "True",
            }
        ],
    )

    search_report = tmp_path / "reports" / "search.csv"
    summary = run_replacement_search(
        data_dir=str(data_dir),
        current_model_dir=str(model_dir),
        current_report=str(current_report),
        candidate_dir=str(candidate_dir),
        report_path=str(search_report),
        top_k=1,
        replace=True,
        timeout_seconds=30,
        rules=[IdentityRule()],
    )

    assert summary["replacement_count"] == 1
    assert summary["replacements"][0]["task_id"] == "task001"
    rows = list(csv.DictReader(search_report.open("r", newline="", encoding="utf-8")))
    assert rows[0]["replace_recommended"] == "True"
    assert rows[0]["selected_replacement"] == "True"
    assert (model_dir / "task001.onnx").read_bytes() == (candidate_dir / "task001_IdentityRule.onnx").read_bytes()
