"""Evaluate one ONNX candidate in an isolated process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort

from .arc_io import load_task
from .cost_estimator import check_forbidden_ops, check_static_shapes, estimate_model_cost
from .validate_onnx_model import validate_task


ort.set_default_logger_severity(3)


def _short_validation_failure(validation: dict[str, Any]) -> str:
    failed = validation.get("failed_cases", [])
    if not failed:
        return ""
    first = failed[0]
    reason = first.get("reason", "validation_failed")
    mismatches = first.get("mismatches") or []
    if mismatches:
        mismatch = mismatches[0]
        return (
            f"{reason}: case={first.get('case_index')} "
            f"row={mismatch.get('row')} col={mismatch.get('col')} "
            f"expected={mismatch.get('expected')} actual={mismatch.get('actual')}"
        )
    return f"{reason}: case={first.get('case_index')}"


def evaluate(model_path: str, task_path: str) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        return {"valid": False, "failure_reason": "missing_model"}

    report: dict[str, Any] = {"model_path": str(path)}
    try:
        onnx.checker.check_model(str(path))
        forbidden = check_forbidden_ops(str(path))
        static_shapes = check_static_shapes(str(path))
        cost = estimate_model_cost(str(path))
        validation = validate_task(str(path), load_task(task_path))
    except Exception as exc:
        return {**report, "valid": False, "failure_reason": f"evaluation_exception: {exc}"}

    failure_reasons: list[str] = []
    if not forbidden["passed"]:
        failure_reasons.append(f"forbidden_ops={forbidden['forbidden_ops_found']}")
    if not static_shapes["passed"]:
        failure_reasons.append(f"static_shapes={static_shapes['failures'][:3]}")
    if not cost["file_size_ok"]:
        failure_reasons.append("file_size_exceeds_limit")
    if not validation["passed"]:
        failure_reasons.append(_short_validation_failure(validation) or "train_validation_failed")

    return {
        **report,
        **cost,
        "valid": not failure_reasons,
        "failure_reason": "; ".join(failure_reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.model, args.task), ensure_ascii=False))


if __name__ == "__main__":
    main()
