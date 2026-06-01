"""Solve one task by generating and validating conservative ONNX candidates."""

from __future__ import annotations

import json
import shutil
import traceback
from pathlib import Path
from typing import Any

import onnx

from .cost_estimator import check_forbidden_ops, check_static_shapes, estimate_model_cost
from .pattern_rules import BaseRule, first_version_rules
from .task_analyzer import analyze_task
from .validate_onnx_model import validate_task


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _candidate_report(model_path: str, task: dict) -> dict[str, Any]:
    report: dict[str, Any] = {}
    try:
        onnx.checker.check_model(model_path)
        report["onnx_checker_ok"] = True
    except Exception as exc:
        report["onnx_checker_ok"] = False
        report["failure_reason"] = f"onnx_checker_failed: {exc}"
        return report

    forbidden = check_forbidden_ops(model_path)
    static_shapes = check_static_shapes(model_path)
    cost = estimate_model_cost(model_path)
    validation = validate_task(model_path, task)
    report.update(cost)
    report.update(
        {
            "forbidden_ops_ok": bool(forbidden["passed"]),
            "forbidden_ops_found": forbidden["forbidden_ops_found"],
            "static_shapes_ok": bool(static_shapes["passed"]),
            "static_shape_failures": static_shapes["failures"],
            "validation_passed": bool(validation["passed"]),
            "validation": validation,
        }
    )
    failure_reasons = []
    if not forbidden["passed"]:
        failure_reasons.append("forbidden_ops")
    if not static_shapes["passed"]:
        failure_reasons.append("dynamic_or_invalid_shapes")
    if not cost["file_size_ok"]:
        failure_reasons.append("file_size_exceeds_limit")
    if not validation["passed"]:
        failure_reasons.append("train_validation_failed")
    report["passed"] = not failure_reasons
    report["failure_reasons"] = failure_reasons
    return report


def solve_task(
    task_id: str,
    task: dict,
    out_dir: str,
    candidate_dir: str,
    log_dir: str,
    rules: list[BaseRule] | None = None,
) -> dict[str, Any]:
    """Generate, validate, and select the lowest-cost candidate for one task."""
    rules = first_version_rules() if rules is None else rules
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(candidate_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    matches = []
    candidates = []
    valid_candidates = []
    failure_reasons: list[Any] = []

    for rule in rules:
        try:
            match = rule.match(task)
            matches.append(match.__dict__)
            if not match.matched:
                failure_reasons.append({"rule": rule.name, "reason": match.reason})
                continue
            if match.metadata.get("builder_available") is False:
                failure_reasons.append(
                    {
                        "rule": rule.name,
                        "reason": match.metadata.get("blocked_reason", "builder_unavailable"),
                    }
                )
                continue

            candidate_path = str(Path(candidate_dir) / f"{task_id}_{rule.name}.onnx")
            candidate = rule.build(task_id, task, candidate_path, match.metadata)
            report = _candidate_report(candidate.model_path, task)
            candidate_entry = {
                "task_id": task_id,
                "rule_name": rule.name,
                "priority": rule.priority,
                "model_path": candidate.model_path,
                "metadata": candidate.metadata,
                **report,
            }
            candidates.append(candidate_entry)
            if report.get("passed"):
                valid_candidates.append(candidate_entry)
            else:
                failure_reasons.append(
                    {
                        "rule": rule.name,
                        "reason": ",".join(report.get("failure_reasons", ["candidate_failed"])),
                    }
                )
        except Exception as exc:
            failure_reasons.append(
                {
                    "rule": rule.name,
                    "reason": f"exception: {exc}",
                    "traceback": traceback.format_exc(limit=5),
                }
            )

    if valid_candidates:
        best = min(
            valid_candidates,
            key=lambda item: (
                item["estimated_cost"],
                item["file_size_bytes"],
                item["priority"],
            ),
        )
        final_path = Path(out_dir) / f"{task_id}.onnx"
        shutil.copyfile(best["model_path"], final_path)
        result = {
            "task_id": task_id,
            "status": "solved",
            "best_rule": best["rule_name"],
            "model_path": str(final_path),
            "num_candidates": len(candidates),
            "num_valid_candidates": len(valid_candidates),
            "estimated_cost": best["estimated_cost"],
            "estimated_score": best["estimated_score"],
            "file_size_bytes": best["file_size_bytes"],
            "failure_reasons": failure_reasons,
        }
    else:
        result = {
            "task_id": task_id,
            "status": "failed",
            "best_rule": None,
            "model_path": None,
            "num_candidates": len(candidates),
            "num_valid_candidates": 0,
            "estimated_cost": None,
            "estimated_score": None,
            "file_size_bytes": None,
            "failure_reasons": failure_reasons,
        }

    log = {
        **result,
        "analysis_summary": analyze_task(task),
        "matches": matches,
        "candidates": candidates,
    }
    log_path = Path(log_dir) / f"{task_id}.json"
    log_path.write_text(
        json.dumps(log, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    return result
