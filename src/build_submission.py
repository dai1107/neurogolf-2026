"""Build the first-version submission.zip from validated task ONNX models."""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .arc_io import load_all_tasks
from .solve_task import solve_task


SUMMARY_FIELDS = [
    "task_id",
    "status",
    "best_rule",
    "model_path",
    "num_candidates",
    "num_valid_candidates",
    "estimated_cost",
    "estimated_score",
    "file_size_bytes",
    "failure_reasons",
]


def _resolve_data_dir(data_dir: str) -> str:
    requested = Path(data_dir)
    if requested.is_dir():
        return data_dir
    fallback = Path("task")
    if data_dir.replace("\\", "/") == "data/arc-agi/training" and fallback.is_dir():
        print("WARNING: data/arc-agi/training not found; using local task/ directory")
        return str(fallback)
    raise FileNotFoundError(f"task directory does not exist: {data_dir}")


def _write_summary(report_path: str, results: list[dict[str, Any]]) -> None:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            row = {field: result.get(field) for field in SUMMARY_FIELDS}
            row["failure_reasons"] = json.dumps(
                row["failure_reasons"],
                ensure_ascii=False,
            )
            writer.writerow(row)


def _write_zip(zip_path: str, solved_results: list[dict[str, Any]]) -> None:
    path = Path(zip_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for result in solved_results:
            model_path = Path(result["model_path"])
            task_id = result["task_id"]
            if task_id == "task000":
                continue
            archive.write(model_path, arcname=f"{task_id}.onnx")


def build_submission(
    data_dir: str,
    out_dir: str,
    candidate_dir: str,
    log_dir: str,
    report: str,
    zip_path: str,
) -> dict[str, Any]:
    """Solve all tasks and build a zip containing only validated ONNX models."""
    resolved_data_dir = _resolve_data_dir(data_dir)
    tasks = load_all_tasks(resolved_data_dir)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(candidate_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(report).parent.mkdir(parents=True, exist_ok=True)

    results = []
    for index, (task_id, task) in enumerate(tasks.items(), start=1):
        result = solve_task(task_id, task, out_dir, candidate_dir, log_dir)
        results.append(result)
        if index % 25 == 0 or result["status"] == "solved":
            print(f"processed {index}/{len(tasks)}: {task_id} {result['status']}")

    solved_results = [result for result in results if result["status"] == "solved"]
    _write_summary(report, results)
    _write_zip(zip_path, solved_results)

    if not solved_results:
        print("WARNING: no solved tasks included in submission.zip")

    breakdown = Counter(result["best_rule"] for result in solved_results)
    summary = {
        "total_tasks": len(tasks),
        "solved_tasks": len(solved_results),
        "failed_tasks": len(tasks) - len(solved_results),
        "submission_path": zip_path,
        "summary_path": report,
        "solved_breakdown": dict(sorted(breakdown.items())),
    }
    print(f"total_tasks = {summary['total_tasks']}")
    print(f"solved_tasks = {summary['solved_tasks']}")
    print(f"failed_tasks = {summary['failed_tasks']}")
    print(f"submission_path = {zip_path}")
    print(f"summary_path = {report}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="task")
    parser.add_argument("--out-dir", default="outputs/onnx")
    parser.add_argument("--candidate-dir", default="outputs/candidates")
    parser.add_argument("--log-dir", default="outputs/logs")
    parser.add_argument("--report", default="outputs/reports/summary.csv")
    parser.add_argument("--zip", dest="zip_path", default="outputs/submission.zip")
    args = parser.parse_args()
    build_submission(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        candidate_dir=args.candidate_dir,
        log_dir=args.log_dir,
        report=args.report,
        zip_path=args.zip_path,
    )


if __name__ == "__main__":
    main()
