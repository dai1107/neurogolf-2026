from __future__ import annotations

import csv
import zipfile

from src.build_pairwise_local_reference_ablations import build_pairwise_local_reference_ablations
from src.onnx_builders import build_color_map_model, build_identity_model


def test_build_pairwise_local_reference_ablations_replaces_one_task(tmp_path) -> None:
    base_dir = tmp_path / "base"
    replacement_dir = tmp_path / "replacement"
    output_dir = tmp_path / "ablation"
    reports_dir = tmp_path / "reports"
    base_zip = tmp_path / "ref.zip"
    selection_report = reports_dir / "selection.csv"
    report = reports_dir / "pairwise.csv"
    base_dir.mkdir()
    replacement_dir.mkdir()
    reports_dir.mkdir()

    build_identity_model(str(base_dir / "task001.onnx"))
    build_identity_model(str(base_dir / "task002.onnx"))
    build_color_map_model({1: 2}, str(replacement_dir / "task001.onnx"))
    build_identity_model(str(replacement_dir / "task002.onnx"))

    with zipfile.ZipFile(base_zip, "w") as archive:
        archive.write(base_dir / "task001.onnx", "task001.onnx")
        archive.write(base_dir / "task002.onnx", "task002.onnx")

    with selection_report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task_id", "selected_source", "archive_estimated_cost", "current_estimated_cost"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "task001",
                "selected_source": "current",
                "archive_estimated_cost": "10",
                "current_estimated_cost": "5",
            }
        )
        writer.writerow(
            {
                "task_id": "task002",
                "selected_source": "current",
                "archive_estimated_cost": "20",
                "current_estimated_cost": "20",
            }
        )

    summary = build_pairwise_local_reference_ablations(
        base_zip=str(base_zip),
        replacement_dir=str(replacement_dir),
        output_dir=str(output_dir),
        report_path=str(report),
        selection_report=str(selection_report),
        selection="selected-current",
        replacement_label="LocalCandidate",
    )

    assert summary["selected_task_count"] == 2
    assert summary["valid_zip_count"] == 1
    assert summary["skipped_identical_count"] == 1

    assert summary["replacement_label"] == "LocalCandidate"
    upload_zip = output_dir / "task001_LocalCandidate" / "submission.zip"
    assert upload_zip.is_file()
    with zipfile.ZipFile(base_zip) as base, zipfile.ZipFile(upload_zip) as candidate:
        assert sorted(candidate.namelist()) == ["task001.onnx", "task002.onnx"]
        assert candidate.read("task001.onnx") != base.read("task001.onnx")
        assert candidate.read("task002.onnx") == base.read("task002.onnx")
