"""Discover probe-only optimization candidates for currently failed tasks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .arc_io import load_all_tasks
from .pattern_rules import (
    BaseRule,
    ComposedRuleSearch,
    DynamicBBoxCropRule,
    FrameInteriorRule,
    ObjectEditRule,
    ObjectSelectionRule,
    PanelSelectByColorRule,
    PanelSemanticRule,
    SubstructureExtractRule,
)


FIELDS = [
    "task_id",
    "candidate_rule",
    "python_transform_passed",
    "onnx_builder_available",
    "onnx_validation_passed",
    "blocked_reason",
    "estimated_codegen_difficulty",
    "expected_gain_bucket",
]


PROBE_ONLY_RULES = {
    "PanelSemanticRule",
    "DynamicBBoxCropRule",
    "FrameInteriorRule",
    "ObjectEditRule",
    "ComposedRuleSearch",
}


def discovery_rules() -> list[BaseRule]:
    """Return broad discovery rules without adding probe-only rules to submission solving."""
    return [
        PanelSemanticRule(),
        DynamicBBoxCropRule(),
        FrameInteriorRule(),
        ObjectEditRule(),
        ComposedRuleSearch(),
        PanelSelectByColorRule(),
        SubstructureExtractRule(),
        ObjectSelectionRule(),
    ]


def _read_summary(summary_path: str) -> dict[str, dict[str, str]]:
    path = Path(summary_path)
    if not path.is_file():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {row["task_id"]: row for row in csv.DictReader(handle)}


def _read_solver_log(log_dir: str, task_id: str) -> dict[str, Any]:
    path = Path(log_dir) / f"{task_id}.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _candidate_validation_from_log(log: dict[str, Any], rule_name: str) -> bool:
    for candidate in log.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        if candidate.get("rule_name") == rule_name and candidate.get("passed"):
            return True
    return False


def _builder_available(rule_name: str, metadata: dict[str, Any]) -> bool:
    if rule_name in PROBE_ONLY_RULES:
        return False
    return bool(metadata.get("builder_available", True))


def _blocked_reason(rule_name: str, metadata: dict[str, Any], validation_passed: bool) -> str:
    explicit = metadata.get("blocked_reason")
    if explicit:
        return str(explicit)
    if rule_name in PROBE_ONLY_RULES:
        return "builder_missing"
    if not validation_passed:
        return "matched_formal_rule_but_no_validated_candidate_in_solver_log"
    return ""


def _difficulty(rule_name: str, metadata: dict[str, Any]) -> str:
    mode = str(metadata.get("mode", ""))
    if rule_name in {"ObjectEditRule", "FrameInteriorRule"}:
        return "high"
    if rule_name in {"PanelSemanticRule", "DynamicBBoxCropRule", "ComposedRuleSearch"}:
        return "medium"
    if mode.startswith("bbox") or "dynamic" in mode:
        return "medium"
    return "low"


def _gain(rule_name: str, metadata: dict[str, Any]) -> str:
    if rule_name in {"PanelSemanticRule", "DynamicBBoxCropRule", "ComposedRuleSearch"}:
        return "high"
    if rule_name in {"FrameInteriorRule", "ObjectEditRule"}:
        return "medium"
    if metadata.get("builder_available", True):
        return "low"
    return "medium"


def build_candidate_discovery_report(
    data_dir: str,
    summary_path: str,
    log_dir: str,
    report_path: str,
    failed_only: bool = True,
) -> list[dict[str, Any]]:
    """Write probe candidate rows for tasks whose train pairs match broad strategies."""
    tasks = load_all_tasks(data_dir)
    summary_rows = _read_summary(summary_path)
    rows: list[dict[str, Any]] = []
    rules = discovery_rules()

    for task_id, task in tasks.items():
        summary_row = summary_rows.get(task_id)
        status = "unknown" if summary_row is None else summary_row.get("status", "unknown")
        if failed_only and status != "failed":
            continue
        log = _read_solver_log(log_dir, task_id)
        for rule in rules:
            match = rule.match(task)
            if not match.matched:
                continue
            metadata = match.metadata
            builder_available = _builder_available(rule.name, metadata)
            validation_passed = bool(builder_available and _candidate_validation_from_log(log, rule.name))
            rows.append(
                {
                    "task_id": task_id,
                    "candidate_rule": rule.name,
                    "python_transform_passed": True,
                    "onnx_builder_available": builder_available,
                    "onnx_validation_passed": validation_passed,
                    "blocked_reason": _blocked_reason(rule.name, metadata, validation_passed),
                    "estimated_codegen_difficulty": _difficulty(rule.name, metadata),
                    "expected_gain_bucket": _gain(rule.name, metadata),
                }
            )

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="task")
    parser.add_argument("--summary", default="outputs/reports/summary.csv")
    parser.add_argument("--log-dir", default="outputs/logs")
    parser.add_argument("--report", default="outputs/reports/candidate_discovery_report.csv")
    parser.add_argument("--all-tasks", action="store_true")
    args = parser.parse_args()
    rows = build_candidate_discovery_report(
        data_dir=args.data_dir,
        summary_path=args.summary,
        log_dir=args.log_dir,
        report_path=args.report,
        failed_only=not args.all_tasks,
    )
    print(f"candidate_rows = {len(rows)}")
    print(f"report_path = {args.report}")


if __name__ == "__main__":
    main()
