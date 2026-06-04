"""Build submission.zip from a validated local ONNX model bank."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .arc_io import load_all_tasks
from .blend_archive_submission import evaluate_model
from .cost_estimator import check_forbidden_ops, estimate_model_cost


FIELDS = [
    "task_id",
    "valid",
    "model_path",
    "estimated_cost",
    "file_size_bytes",
    "failure_reason",
    "selected_for_zip",
]


def build_current_model_submission(
    data_dir: str,
    model_dir: str,
    validated_dir: str,
    report_path: str,
    zip_path: str,
    allow_partial: bool = False,
    timeout_seconds: int = 120,
    validation_mode: str = "strict",
) -> dict[str, Any]:
    """Validate local per-task ONNX models and package only passing models."""
    if validation_mode not in {"strict", "trusted"}:
        raise ValueError(f"validation_mode must be 'strict' or 'trusted', got {validation_mode}")
    tasks = load_all_tasks(data_dir)
    data_root = Path(data_dir)
    model_root = Path(model_dir)
    validated_root = Path(validated_dir)
    validated_root.mkdir(parents=True, exist_ok=True)
    if validated_root.resolve() == model_root.resolve():
        raise ValueError("validated_dir must be different from model_dir")
    for stale_model in validated_root.glob("task*.onnx"):
        stale_model.unlink()
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(zip_path).parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    selected: list[dict[str, str]] = []
    for task_id in tasks:
        source_path = model_root / f"{task_id}.onnx"
        task_path = data_root / f"{task_id}.json"
        if validation_mode == "strict":
            report = evaluate_model(source_path, task_path, timeout_seconds)
        else:
            report = evaluate_trusted_model(source_path)
        valid = bool(report.get("valid"))
        destination = validated_root / f"{task_id}.onnx"
        if valid:
            shutil.copyfile(report["model_path"], destination)
            selected.append({"task_id": task_id, "path": str(destination)})
        rows.append(
            {
                "task_id": task_id,
                "valid": valid,
                "model_path": report.get("model_path", str(source_path)),
                "estimated_cost": report.get("estimated_cost", ""),
                "file_size_bytes": report.get("file_size_bytes", ""),
                "failure_reason": report.get("failure_reason", ""),
                "selected_for_zip": valid,
            }
        )

    with Path(report_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    missing_or_invalid = [row["task_id"] for row in rows if not row["valid"]]
    if missing_or_invalid and not allow_partial:
        preview = ", ".join(missing_or_invalid[:20])
        raise ValueError(
            "current model bank validation failed: "
            f"{len(selected)}/{len(tasks)} valid; missing_or_invalid={preview}"
        )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in selected:
            archive.write(item["path"], arcname=f"{item['task_id']}.onnx")

    total_cost = sum(int(row["estimated_cost"]) for row in rows if row["estimated_cost"] != "")
    total_file_size = sum(int(row["file_size_bytes"]) for row in rows if row["file_size_bytes"] != "")
    summary = {
        "total_tasks": len(tasks),
        "selected_tasks": len(selected),
        "missing_or_invalid_tasks": len(missing_or_invalid),
        "estimated_cost_total": total_cost,
        "onnx_file_size_total": total_file_size,
        "report_path": report_path,
        "zip_path": zip_path,
        "validation_mode": validation_mode,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def evaluate_trusted_model(model_path: Path) -> dict[str, Any]:
    """Validate a model for trusted online-baseline packaging.

    This deliberately mirrors the submission-structure gate instead of strict
    local train validation. It is intended only for reproducing a known online
    baseline whose models may fail local padding/static-shape heuristics.
    """
    if not model_path.is_file():
        return {"valid": False, "model_path": str(model_path), "failure_reason": "missing_model"}

    report: dict[str, Any] = {"model_path": str(model_path)}
    try:
        cost = estimate_model_cost(str(model_path))
        forbidden = check_forbidden_ops(str(model_path))
    except Exception as exc:
        return {**report, "valid": False, "failure_reason": f"trusted_evaluation_exception: {exc}"}

    failure_reasons: list[str] = []
    if not cost["file_size_ok"]:
        failure_reasons.append("file_size_exceeds_limit")
    if not forbidden["passed"]:
        failure_reasons.append(f"forbidden_ops={forbidden['forbidden_ops_found']}")

    return {
        **report,
        **cost,
        "valid": not failure_reasons,
        "failure_reason": "; ".join(failure_reasons),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="task")
    parser.add_argument("--model-dir", default="outputs/onnx")
    parser.add_argument("--validated-dir", default="outputs/current_model_bank_verified_onnx")
    parser.add_argument("--report", default="outputs/reports/current_model_bank_report.csv")
    parser.add_argument("--zip", dest="zip_path", default="outputs/submission.zip")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--validation-mode",
        choices=["strict", "trusted"],
        default="strict",
        help="strict validates train output; trusted only packages a known online-clean model bank",
    )
    args = parser.parse_args()
    build_current_model_submission(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        validated_dir=args.validated_dir,
        report_path=args.report,
        zip_path=args.zip_path,
        allow_partial=args.allow_partial,
        timeout_seconds=args.timeout_seconds,
        validation_mode=args.validation_mode,
    )


if __name__ == "__main__":
    main()
