from __future__ import annotations

import csv
import zipfile

from src.build_6348_6275_cost_experiments import build_local_cost_candidate, build_one_task_ablations
from src.inspect_submission import inspect_submission
from src.onnx_builders import build_color_map_model, build_identity_model


def _write_task(path, color: int) -> None:
    path.write_text(
        f'{{"train":[{{"input":[[{color}]],"output":[[{color}]]}}]}}',
        encoding="utf-8",
    )


def _write_hybrid_zip(zip_path, stack_dir, task_ids: list[str]) -> None:
    with zipfile.ZipFile(zip_path, "w") as archive:
        for folder in ("base_submission", "overrides"):
            for task_id in task_ids:
                archive.write(stack_dir / folder / f"{task_id}.onnx", f"{folder}/{task_id}.onnx")


def test_build_local_cost_candidate_and_one_task_ablations(tmp_path) -> None:
    data_dir = tmp_path / "task"
    ref6275 = tmp_path / "ref6275"
    ref6348 = tmp_path / "ref6348"
    for path in (data_dir, ref6275, ref6348 / "base_submission", ref6348 / "overrides"):
        path.mkdir(parents=True)

    _write_task(data_dir / "task001.json", 1)
    _write_task(data_dir / "task002.json", 2)

    build_identity_model(str(ref6275 / "task001.onnx"))
    build_color_map_model({2: 3}, str(ref6275 / "task002.onnx"))
    build_color_map_model({1: 2}, str(ref6348 / "base_submission" / "task001.onnx"))
    build_color_map_model({1: 3}, str(ref6348 / "overrides" / "task001.onnx"))
    build_color_map_model({2: 3}, str(ref6348 / "base_submission" / "task002.onnx"))
    build_identity_model(str(ref6348 / "overrides" / "task002.onnx"))

    local_zip = tmp_path / "local_cost" / "submission.zip"
    selection_report = tmp_path / "reports" / "selection.csv"
    summary = build_local_cost_candidate(
        data_dir=str(data_dir),
        ref6275_dir=str(ref6275),
        ref6348_stack_dir=str(ref6348),
        output_zip=str(local_zip),
        report_path=str(selection_report),
    )

    assert summary["selected_tasks"] == 2
    assert summary["source_counts"] == {"ref6275": 1, "ref6348_overrides": 1}
    assert inspect_submission(str(local_zip), layout="flat") == {"passed": True, "num_models": 2}

    with selection_report.open(newline="", encoding="utf-8") as handle:
        rows = {row["task_id"]: row for row in csv.DictReader(handle)}
    assert rows["task001"]["selected_source"] == "ref6275"
    assert rows["task002"]["selected_source"] == "ref6348_overrides"

    base_zip = tmp_path / "6348.zip"
    _write_hybrid_zip(base_zip, ref6348, ["task001", "task002"])
    ablation_report = tmp_path / "reports" / "ablations.csv"
    ablation_dir = tmp_path / "ablations"
    ablation_summary = build_one_task_ablations(
        base_6348_zip=str(base_zip),
        ref6275_dir=str(ref6275),
        selection_report=str(selection_report),
        output_dir=str(ablation_dir),
        report_path=str(ablation_report),
    )

    assert ablation_summary["selected_task_count"] == 1
    assert ablation_summary["valid_zip_count"] == 1
    candidate_zip = ablation_dir / "task001_6275Over6348BothLanes.zip"
    assert inspect_submission(str(candidate_zip), layout="hybrid_stack")["num_models"] == 4

    with zipfile.ZipFile(candidate_zip) as archive:
        base_data = archive.read("base_submission/task001.onnx")
        overrides_data = archive.read("overrides/task001.onnx")
        assert base_data == overrides_data == (ref6275 / "task001.onnx").read_bytes()
        assert archive.read("overrides/task002.onnx") == (ref6348 / "overrides" / "task002.onnx").read_bytes()
