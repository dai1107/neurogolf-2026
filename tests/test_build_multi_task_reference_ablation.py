from __future__ import annotations

import csv
import zipfile

from src.build_multi_task_reference_ablation import build_multi_task_reference_ablation
from src.onnx_builders import build_color_map_model, build_identity_model


def test_build_multi_task_reference_ablation_replaces_selected_ordinals(tmp_path) -> None:
    base_dir = tmp_path / "base"
    replacement_dir = tmp_path / "replacement"
    output_zip = tmp_path / "round" / "candidate.zip"
    upload_zip = tmp_path / "round" / "candidate" / "submission.zip"
    selection_report = tmp_path / "reports" / "selection.csv"
    report = tmp_path / "reports" / "multi.csv"
    base_zip = tmp_path / "base.zip"
    base_dir.mkdir()
    replacement_dir.mkdir()
    selection_report.parent.mkdir()

    for task_id in ["task001", "task002", "task003"]:
        build_identity_model(str(base_dir / f"{task_id}.onnx"))
        build_color_map_model({1: 2}, str(replacement_dir / f"{task_id}.onnx"))

    with zipfile.ZipFile(base_zip, "w") as archive:
        for task_id in ["task001", "task002", "task003"]:
            archive.write(base_dir / f"{task_id}.onnx", f"{task_id}.onnx")

    with selection_report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "replacement_model_path"])
        writer.writeheader()
        writer.writerow({"task_id": "task001", "replacement_model_path": str(replacement_dir / "task001.onnx")})
        writer.writerow({"task_id": "task002", "replacement_model_path": str(replacement_dir / "task002.onnx")})
        writer.writerow({"task_id": "task003", "replacement_model_path": str(replacement_dir / "task003.onnx")})

    summary = build_multi_task_reference_ablation(
        base_zip=str(base_zip),
        replacement_dir=str(replacement_dir),
        selection_report=str(selection_report),
        ordinals=[1, 3],
        output_zip=str(output_zip),
        upload_path=str(upload_zip),
        report_path=str(report),
    )

    assert summary["selected_tasks"] == ["task001", "task003"]
    assert summary["selected_count"] == 2
    assert output_zip.is_file()
    assert upload_zip.is_file()
    with zipfile.ZipFile(base_zip) as base, zipfile.ZipFile(upload_zip) as candidate:
        assert candidate.read("task001.onnx") != base.read("task001.onnx")
        assert candidate.read("task002.onnx") == base.read("task002.onnx")
        assert candidate.read("task003.onnx") != base.read("task003.onnx")
