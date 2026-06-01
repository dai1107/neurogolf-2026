"""Run pure-Python probes for second-round rules before ONNX generation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .arc_io import load_all_tasks
from .pattern_rules import third_round_probe_rules


PROBE_FIELDS = [
    "rule_name",
    "scanned_tasks",
    "matched_count",
    "matched_tasks",
    "exception_count",
]


def _load_solved_task_ids(summary_path: str) -> set[str]:
    path = Path(summary_path)
    if not path.is_file():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            row["task_id"]
            for row in csv.DictReader(handle)
            if row.get("status") == "solved"
        }


def probe_rules(
    data_dir: str,
    report_path: str,
    summary_path: str = "outputs/reports/summary.csv",
    failed_only: bool = True,
    exclude_task_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    """Probe tasks with pure-Python rule matchers and write aggregate counts."""
    tasks = load_all_tasks(data_dir)
    if exclude_task_ids is not None:
        solved_task_ids = exclude_task_ids
    else:
        solved_task_ids = _load_solved_task_ids(summary_path) if failed_only else set()
    scanned_tasks = {
        task_id: task
        for task_id, task in tasks.items()
        if task_id not in solved_task_ids
    }

    rows: list[dict[str, object]] = []
    for rule in third_round_probe_rules():
        matched_tasks: list[str] = []
        exception_count = 0
        for task_id, task in scanned_tasks.items():
            try:
                match = rule.match(task)
            except Exception:
                exception_count += 1
                continue
            if match.matched and match.confidence == "MATCH":
                matched_tasks.append(task_id)
        rows.append(
            {
                "rule_name": rule.name,
                "scanned_tasks": len(scanned_tasks),
                "matched_count": len(matched_tasks),
                "matched_tasks": " ".join(matched_tasks),
                "exception_count": exception_count,
            }
        )

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROBE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="task")
    parser.add_argument("--report", default="outputs/reports/probe_summary.csv")
    parser.add_argument("--summary", default="outputs/reports/summary.csv")
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--exclude-task-ids", default="")
    args = parser.parse_args()
    exclude_task_ids = {
        task_id.strip()
        for task_id in args.exclude_task_ids.split(",")
        if task_id.strip()
    } or None
    rows = probe_rules(
        data_dir=args.data_dir,
        report_path=args.report,
        summary_path=args.summary,
        failed_only=not args.all_tasks,
        exclude_task_ids=exclude_task_ids,
    )
    for row in rows:
        print(
            f"{row['rule_name']}: {row['matched_count']}/"
            f"{row['scanned_tasks']} matched"
        )
    print(f"probe_summary = {args.report}")


if __name__ == "__main__":
    main()
