"""Validate an external ONNX archive and blend it with local solved models."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import onnxruntime as ort

from .arc_io import load_all_tasks


ort.set_default_logger_severity(3)


FIELDS = [
    "task_id",
    "selected_source",
    "selected_model_path",
    "selected_estimated_cost",
    "selected_file_size_bytes",
    "archive_valid",
    "archive_estimated_cost",
    "archive_file_size_bytes",
    "archive_failure_reason",
    "current_valid",
    "current_estimated_cost",
    "current_file_size_bytes",
    "current_failure_reason",
]


def evaluate_model(model_path: Path, task_path: Path, timeout_seconds: int) -> dict[str, Any]:
    """Return strict local validation and cost details for one model in a subprocess."""
    if not model_path.is_file():
        return {"valid": False, "failure_reason": "missing_model"}
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.evaluate_onnx_candidate",
                "--model",
                str(model_path),
                "--task",
                str(task_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"valid": False, "model_path": str(model_path), "failure_reason": "evaluation_timeout"}
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().replace("\n", " ")
        return {
            "valid": False,
            "model_path": str(model_path),
            "failure_reason": f"evaluation_subprocess_failed: returncode={completed.returncode} {detail[:300]}",
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {
            "valid": False,
            "model_path": str(model_path),
            "failure_reason": f"evaluation_json_decode_failed: {exc}",
        }


def _choose_candidate(task_id: str, archive: dict[str, Any], current: dict[str, Any]) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    if archive.get("valid"):
        candidates.append({"source": "archive", **archive})
    if current.get("valid"):
        candidates.append({"source": "current", **current})
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            int(item["estimated_cost"]),
            int(item["file_size_bytes"]),
            0 if item["source"] == "current" else 1,
            task_id,
        ),
    )


def blend_archive_submission(
    data_dir: str,
    archive_dir: str,
    current_dir: str,
    blended_dir: str,
    report_path: str,
    zip_path: str,
    only_task_ids: set[str] | None = None,
    exclude_task_ids: set[str] | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Validate archive/current candidates, choose lowest-cost valid model, and zip them."""
    tasks = load_all_tasks(data_dir)
    data_root = Path(data_dir)
    archive_root = Path(archive_dir)
    current_root = Path(current_dir)
    blended_root = Path(blended_dir)
    blended_root.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(zip_path).parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for task_id, task in tasks.items():
        if exclude_task_ids is not None and task_id in exclude_task_ids:
            rows.append(
                {
                    "task_id": task_id,
                    "selected_source": "",
                    "selected_model_path": "",
                    "selected_estimated_cost": "",
                    "selected_file_size_bytes": "",
                    "archive_valid": False,
                    "archive_estimated_cost": "",
                    "archive_file_size_bytes": "",
                    "archive_failure_reason": "excluded_by_user",
                    "current_valid": False,
                    "current_estimated_cost": "",
                    "current_file_size_bytes": "",
                    "current_failure_reason": "excluded_by_user",
                }
            )
            continue
        if only_task_ids is not None and task_id not in only_task_ids:
            continue
        try:
            task_path = data_root / f"{task_id}.json"
            archive_report = evaluate_model(archive_root / f"{task_id}.onnx", task_path, timeout_seconds)
            current_report = evaluate_model(current_root / f"{task_id}.onnx", task_path, timeout_seconds)
            best = _choose_candidate(task_id, archive_report, current_report)
        except BaseException as exc:  # Keep one bad external model from aborting the audit.
            archive_report = {"valid": False, "failure_reason": f"task_level_exception: {type(exc).__name__}: {exc}"}
            current_report = {"valid": False, "failure_reason": "not_evaluated_after_task_exception"}
            best = None
        if best is not None:
            destination = blended_root / f"{task_id}.onnx"
            shutil.copyfile(best["model_path"], destination)
            selected.append({"task_id": task_id, "path": str(destination), "source": best["source"]})
            selected_source = best["source"]
            selected_path = str(destination)
            selected_cost = best["estimated_cost"]
            selected_file_size = best["file_size_bytes"]
        else:
            selected_source = ""
            selected_path = ""
            selected_cost = ""
            selected_file_size = ""
        rows.append(
            {
                "task_id": task_id,
                "selected_source": selected_source,
                "selected_model_path": selected_path,
                "selected_estimated_cost": selected_cost,
                "selected_file_size_bytes": selected_file_size,
                "archive_valid": bool(archive_report.get("valid")),
                "archive_estimated_cost": archive_report.get("estimated_cost", ""),
                "archive_file_size_bytes": archive_report.get("file_size_bytes", ""),
                "archive_failure_reason": archive_report.get("failure_reason", ""),
                "current_valid": bool(current_report.get("valid")),
                "current_estimated_cost": current_report.get("estimated_cost", ""),
                "current_file_size_bytes": current_report.get("file_size_bytes", ""),
                "current_failure_reason": current_report.get("failure_reason", ""),
            }
        )

    with Path(report_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in selected:
            archive.write(item["path"], arcname=f"{item['task_id']}.onnx")

    source_counts: dict[str, int] = {}
    for item in selected:
        source_counts[item["source"]] = source_counts.get(item["source"], 0) + 1
    summary = {
        "total_tasks": len(tasks),
        "selected_tasks": len(selected),
        "missing_tasks": len(tasks) - len(selected),
        "source_counts": dict(sorted(source_counts.items())),
        "report_path": report_path,
        "zip_path": zip_path,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="task")
    parser.add_argument("--archive-dir", default="archive")
    parser.add_argument("--current-dir", default="outputs/onnx")
    parser.add_argument("--blended-dir", default="outputs/blended_onnx")
    parser.add_argument("--report", default="outputs/reports/archive_blend_report.csv")
    parser.add_argument("--zip", dest="zip_path", default="outputs/submission.zip")
    parser.add_argument("--task-ids", default="", help="Optional comma-separated task ids such as task042,task043")
    parser.add_argument("--exclude-task-ids", default="", help="Optional comma-separated task ids to omit from the zip")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()
    only_task_ids = {item.strip() for item in args.task_ids.split(",") if item.strip()} or None
    exclude_task_ids = {item.strip() for item in args.exclude_task_ids.split(",") if item.strip()} or None
    blend_archive_submission(
        data_dir=args.data_dir,
        archive_dir=args.archive_dir,
        current_dir=args.current_dir,
        blended_dir=args.blended_dir,
        report_path=args.report,
        zip_path=args.zip_path,
        only_task_ids=only_task_ids,
        exclude_task_ids=exclude_task_ids,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == "__main__":
    main()
