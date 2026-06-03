"""Search formal symbolic rules for cheaper replacements of high-cost models."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Sequence

from .arc_io import load_task
from .blend_archive_submission import evaluate_model
from .pattern_rules import BaseRule, first_version_rules


FIELDS = [
    "task_id",
    "current_cost",
    "current_file_size",
    "rule_name",
    "match_status",
    "match_confidence",
    "match_reason",
    "builder_available",
    "candidate_model_path",
    "validation_passed",
    "candidate_cost",
    "candidate_file_size",
    "failure_reason",
    "replace_recommended",
    "selected_replacement",
]


def _is_true(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _read_current_report(report_path: str) -> dict[str, dict[str, Any]]:
    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError(f"current report does not exist: {report_path}")
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task_id = row.get("task_id", "")
            if not task_id:
                continue
            if row.get("valid") and not _is_true(row["valid"]):
                continue
            cost = row.get("estimated_cost", "")
            file_size = row.get("file_size_bytes", "")
            if cost == "" or file_size == "":
                continue
            rows[task_id] = {
                "task_id": task_id,
                "current_cost": int(cost),
                "current_file_size": int(file_size),
            }
    return rows


def _select_targets(
    current_rows: dict[str, dict[str, Any]],
    top_k: int,
    task_ids: list[str] | None,
) -> list[str]:
    if task_ids is not None:
        missing = [task_id for task_id in task_ids if task_id not in current_rows]
        if missing:
            raise ValueError(f"task ids missing from current report: {', '.join(missing)}")
        return task_ids[:top_k]
    return [
        item["task_id"]
        for item in sorted(current_rows.values(), key=lambda row: row["current_cost"], reverse=True)[:top_k]
    ]


def _empty_row(
    task_id: str,
    current: dict[str, Any],
    rule_name: str,
    match_status: str,
    match_confidence: str,
    match_reason: str,
    builder_available: bool | str = "",
    failure_reason: str = "",
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "current_cost": current["current_cost"],
        "current_file_size": current["current_file_size"],
        "rule_name": rule_name,
        "match_status": match_status,
        "match_confidence": match_confidence,
        "match_reason": match_reason,
        "builder_available": builder_available,
        "candidate_model_path": "",
        "validation_passed": "",
        "candidate_cost": "",
        "candidate_file_size": "",
        "failure_reason": failure_reason,
        "replace_recommended": False,
        "selected_replacement": False,
    }


def _candidate_path(candidate_root: Path, task_id: str, rule_name: str) -> Path:
    return candidate_root / f"{task_id}_{rule_name}.onnx"


def _mark_best_replacement(rows: list[dict[str, Any]], task_start_index: int) -> dict[str, Any] | None:
    candidates: list[tuple[int, int, int]] = []
    for index in range(task_start_index, len(rows)):
        row = rows[index]
        if not row["replace_recommended"]:
            continue
        candidates.append((int(row["candidate_cost"]), int(row["candidate_file_size"]), index))
    if not candidates:
        return None
    _, _, best_index = min(candidates)
    rows[best_index]["selected_replacement"] = True
    return rows[best_index]


def run_replacement_search(
    data_dir: str,
    current_model_dir: str,
    current_report: str,
    candidate_dir: str,
    report_path: str,
    top_k: int = 7,
    task_ids: list[str] | None = None,
    replace: bool = False,
    timeout_seconds: int = 120,
    rules: Sequence[BaseRule] | None = None,
) -> dict[str, Any]:
    """Build and validate formal-rule candidates, optionally replacing cheaper ones."""
    current_rows = _read_current_report(current_report)
    targets = _select_targets(current_rows, top_k, task_ids)
    data_root = Path(data_dir)
    current_root = Path(current_model_dir)
    candidate_root = Path(candidate_dir)
    candidate_root.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    formal_rules = list(first_version_rules() if rules is None else rules)

    rows: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    for task_id in targets:
        task = load_task(str(data_root / f"{task_id}.json"))
        current = current_rows[task_id]
        task_start_index = len(rows)
        for rule in formal_rules:
            try:
                match = rule.match(task)
            except Exception as exc:
                rows.append(
                    _empty_row(
                        task_id,
                        current,
                        rule.name,
                        "EXCEPTION",
                        "",
                        "",
                        failure_reason=f"match_exception: {exc}",
                    )
                )
                continue

            if not match.matched:
                rows.append(
                    _empty_row(
                        task_id,
                        current,
                        rule.name,
                        "REJECT",
                        match.confidence,
                        match.reason,
                        builder_available=False,
                    )
                )
                continue

            builder_available = bool(match.metadata.get("builder_available", True))
            if match.confidence != "MATCH" or not builder_available:
                rows.append(
                    _empty_row(
                        task_id,
                        current,
                        rule.name,
                        "MATCH_BLOCKED",
                        match.confidence,
                        match.reason,
                        builder_available=builder_available,
                        failure_reason=match.metadata.get("blocked_reason", "builder_unavailable"),
                    )
                )
                continue

            path = _candidate_path(candidate_root, task_id, rule.name)
            try:
                rule.build(task_id, task, str(path), match.metadata)
                evaluation = evaluate_model(path, data_root / f"{task_id}.json", timeout_seconds)
            except Exception as exc:
                rows.append(
                    {
                        **_empty_row(
                            task_id,
                            current,
                            rule.name,
                            "MATCH",
                            match.confidence,
                            match.reason,
                            builder_available=True,
                            failure_reason=f"build_or_evaluation_exception: {exc}",
                        ),
                        "candidate_model_path": str(path),
                    }
                )
                continue

            validation_passed = bool(evaluation.get("valid"))
            candidate_cost = evaluation.get("estimated_cost", "")
            candidate_file_size = evaluation.get("file_size_bytes", "")
            replace_recommended = (
                validation_passed
                and candidate_cost != ""
                and int(candidate_cost) < int(current["current_cost"])
            )
            rows.append(
                {
                    "task_id": task_id,
                    "current_cost": current["current_cost"],
                    "current_file_size": current["current_file_size"],
                    "rule_name": rule.name,
                    "match_status": "MATCH",
                    "match_confidence": match.confidence,
                    "match_reason": match.reason,
                    "builder_available": True,
                    "candidate_model_path": str(path),
                    "validation_passed": validation_passed,
                    "candidate_cost": candidate_cost,
                    "candidate_file_size": candidate_file_size,
                    "failure_reason": evaluation.get("failure_reason", ""),
                    "replace_recommended": replace_recommended,
                    "selected_replacement": False,
                }
            )

        best = _mark_best_replacement(rows, task_start_index)
        if best is not None:
            delta = int(current["current_cost"]) - int(best["candidate_cost"])
            replacements.append(
                {
                    "task_id": task_id,
                    "rule_name": best["rule_name"],
                    "old_cost": int(current["current_cost"]),
                    "new_cost": int(best["candidate_cost"]),
                    "delta": delta,
                    "candidate_model_path": best["candidate_model_path"],
                }
            )
            if replace:
                destination = current_root / f"{task_id}.onnx"
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(best["candidate_model_path"], destination)

    with Path(report_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "target_tasks": targets,
        "searched_rules": len(formal_rules),
        "candidate_rows": len(rows),
        "replacement_count": len(replacements),
        "replacements": replacements,
        "report_path": report_path,
        "candidate_dir": candidate_dir,
        "replaced_outputs": replace,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_task_ids(raw: str) -> list[str] | None:
    task_ids = [item.strip() for item in raw.split(",") if item.strip()]
    return task_ids or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="task")
    parser.add_argument("--current-model-dir", default="outputs/onnx")
    parser.add_argument("--current-report", default="outputs/reports/current_model_bank_report.csv")
    parser.add_argument("--candidate-dir", default="outputs/candidates/replacements")
    parser.add_argument("--report", default="outputs/reports/replacement_search_report.csv")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        run_replacement_search(
            data_dir=args.data_dir,
            current_model_dir=args.current_model_dir,
            current_report=args.current_report,
            candidate_dir=args.candidate_dir,
            report_path=args.report,
            top_k=args.top_k,
            task_ids=_parse_task_ids(args.task_ids),
            replace=args.replace,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
