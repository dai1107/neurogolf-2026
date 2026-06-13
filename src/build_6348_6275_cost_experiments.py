"""Build local-cost 6275/6348 experiment packages.

The current best known-online package is a 6348.56 hybrid stack with
``base_submission/`` and ``overrides/`` lanes. The older 6275.09 reference is a
flat task bank. This module intentionally keeps the two experiments separate:

* a flat local-cost candidate selecting the cheapest structural model per task;
* one-task ablation zips that keep the 6348 stack as base and replace both
  lanes for one task with the 6275 model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .arc_io import load_all_tasks
from .cost_estimator import check_forbidden_ops, estimate_model_cost
from .inspect_submission import inspect_submission


SELECTION_FIELDS = [
    "task_id",
    "selected_source",
    "selected_model_path",
    "selected_estimated_cost",
    "selected_file_size_bytes",
    "ref6275_valid",
    "ref6275_estimated_cost",
    "ref6275_file_size_bytes",
    "ref6275_failure_reason",
    "ref6348_base_valid",
    "ref6348_base_estimated_cost",
    "ref6348_base_file_size_bytes",
    "ref6348_base_failure_reason",
    "ref6348_overrides_valid",
    "ref6348_overrides_estimated_cost",
    "ref6348_overrides_file_size_bytes",
    "ref6348_overrides_failure_reason",
    "best_6348_source",
    "best_6348_estimated_cost",
    "cost_delta_6275_minus_best_6348",
]

ABLATION_FIELDS = [
    "task_id",
    "candidate_zip_path",
    "upload_submission_path",
    "replacement_model_path",
    "replacement_estimated_cost",
    "best_6348_estimated_cost",
    "cost_delta_6275_minus_best_6348",
    "base_lane_replaced",
    "overrides_lane_replaced",
    "replacement_sha256",
    "base_lane_old_sha256",
    "overrides_lane_old_sha256",
    "candidate_valid",
    "failure_reason",
]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_model(model_path: Path) -> dict[str, Any]:
    if not model_path.is_file():
        return {"valid": False, "model_path": str(model_path), "failure_reason": "missing_model"}
    try:
        cost = estimate_model_cost(str(model_path))
        forbidden = check_forbidden_ops(str(model_path))
    except Exception as exc:
        return {
            "valid": False,
            "model_path": str(model_path),
            "failure_reason": f"model_validation_exception: {exc}",
        }
    failures: list[str] = []
    if not cost["file_size_ok"]:
        failures.append("file_size_exceeds_limit")
    if not forbidden["passed"]:
        failures.append(f"forbidden_ops={forbidden['forbidden_ops_found']}")
    return {
        **cost,
        "valid": not failures,
        "model_path": str(model_path),
        "failure_reason": "; ".join(failures),
    }


def _model_summary(report: dict[str, Any], field: str) -> Any:
    return report.get(field, "")


def _choose_model(
    ref6275: dict[str, Any],
    ref6348_base: dict[str, Any],
    ref6348_overrides: dict[str, Any],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for source, report, priority in (
        ("ref6348_overrides", ref6348_overrides, 0),
        ("ref6348_base", ref6348_base, 1),
        ("ref6275", ref6275, 2),
    ):
        if report.get("valid"):
            candidates.append({"source": source, "priority": priority, **report})
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            int(item["estimated_cost"]),
            int(item["file_size_bytes"]),
            int(item["priority"]),
        ),
    )


def _best_6348(
    ref6348_base: dict[str, Any],
    ref6348_overrides: dict[str, Any],
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    if ref6348_overrides.get("valid"):
        candidates.append({"source": "ref6348_overrides", "priority": 0, **ref6348_overrides})
    if ref6348_base.get("valid"):
        candidates.append({"source": "ref6348_base", "priority": 1, **ref6348_base})
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            int(item["estimated_cost"]),
            int(item["file_size_bytes"]),
            int(item["priority"]),
        ),
    )


def _write_selection_report(report_path: Path, rows: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_local_cost_candidate(
    data_dir: str,
    ref6275_dir: str,
    ref6348_stack_dir: str,
    output_zip: str,
    report_path: str,
) -> dict[str, Any]:
    """Build a flat 400-entry package by local structural cost only."""
    task_ids = sorted(load_all_tasks(data_dir))
    ref6275_root = Path(ref6275_dir)
    ref6348_root = Path(ref6348_stack_dir)
    output = Path(output_zip)
    report = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    selected: list[tuple[str, Path, str]] = []
    for task_id in task_ids:
        ref6275 = _validate_model(ref6275_root / f"{task_id}.onnx")
        ref6348_base = _validate_model(ref6348_root / "base_submission" / f"{task_id}.onnx")
        ref6348_overrides = _validate_model(ref6348_root / "overrides" / f"{task_id}.onnx")
        best = _choose_model(ref6275, ref6348_base, ref6348_overrides)
        best_6348 = _best_6348(ref6348_base, ref6348_overrides)

        selected_source = ""
        selected_path = ""
        selected_cost = ""
        selected_file_size = ""
        if best is not None:
            selected_source = best["source"]
            selected_path = best["model_path"]
            selected_cost = best["estimated_cost"]
            selected_file_size = best["file_size_bytes"]
            selected.append((task_id, Path(best["model_path"]), selected_source))

        ref6275_cost = _model_summary(ref6275, "estimated_cost")
        best_6348_cost = _model_summary(best_6348 or {}, "estimated_cost")
        cost_delta = ""
        if ref6275_cost != "" and best_6348_cost != "":
            cost_delta = int(ref6275_cost) - int(best_6348_cost)

        rows.append(
            {
                "task_id": task_id,
                "selected_source": selected_source,
                "selected_model_path": selected_path,
                "selected_estimated_cost": selected_cost,
                "selected_file_size_bytes": selected_file_size,
                "ref6275_valid": bool(ref6275.get("valid")),
                "ref6275_estimated_cost": ref6275_cost,
                "ref6275_file_size_bytes": _model_summary(ref6275, "file_size_bytes"),
                "ref6275_failure_reason": _model_summary(ref6275, "failure_reason"),
                "ref6348_base_valid": bool(ref6348_base.get("valid")),
                "ref6348_base_estimated_cost": _model_summary(ref6348_base, "estimated_cost"),
                "ref6348_base_file_size_bytes": _model_summary(ref6348_base, "file_size_bytes"),
                "ref6348_base_failure_reason": _model_summary(ref6348_base, "failure_reason"),
                "ref6348_overrides_valid": bool(ref6348_overrides.get("valid")),
                "ref6348_overrides_estimated_cost": _model_summary(ref6348_overrides, "estimated_cost"),
                "ref6348_overrides_file_size_bytes": _model_summary(ref6348_overrides, "file_size_bytes"),
                "ref6348_overrides_failure_reason": _model_summary(ref6348_overrides, "failure_reason"),
                "best_6348_source": _model_summary(best_6348 or {}, "source"),
                "best_6348_estimated_cost": best_6348_cost,
                "cost_delta_6275_minus_best_6348": cost_delta,
            }
        )

    if len(selected) != len(task_ids):
        _write_selection_report(report, rows)
        raise ValueError(f"selected {len(selected)} models for {len(task_ids)} tasks")

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for task_id, model_path, _source in selected:
            archive.write(model_path, arcname=f"{task_id}.onnx")

    inspection = inspect_submission(str(output), layout="flat")
    _write_selection_report(report, rows)

    source_counts: dict[str, int] = {}
    for _task_id, _model_path, source in selected:
        source_counts[source] = source_counts.get(source, 0) + 1

    summary = {
        "output_zip": str(output),
        "report_path": str(report),
        "layout": "flat",
        "selected_tasks": len(selected),
        "source_counts": dict(sorted(source_counts.items())),
        "inspection": inspection,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _load_selection_rows(selection_report: Path) -> list[dict[str, str]]:
    if not selection_report.is_file():
        raise FileNotFoundError(f"selection report does not exist: {selection_report}")
    with selection_report.open("r", newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row.get("task_id")]


def _selected_6275_task_ids(selection_report: Path) -> list[str]:
    rows = _load_selection_rows(selection_report)
    task_ids = [row["task_id"] for row in rows if row.get("selected_source") == "ref6275"]
    return sorted(task_ids)


def _validate_hybrid_base_zip(base_zip: Path) -> dict[str, bytes]:
    inspect_submission(str(base_zip), layout="hybrid_stack")
    with zipfile.ZipFile(base_zip, "r") as archive:
        return {name: archive.read(name) for name in sorted(archive.namelist()) if not name.endswith("/")}


def _write_one_task_hybrid_ablation(
    base_entries: dict[str, bytes],
    task_id: str,
    replacement_model: Path,
    output_zip: Path,
) -> tuple[str, str, str]:
    base_entry = f"base_submission/{task_id}.onnx"
    overrides_entry = f"overrides/{task_id}.onnx"
    if base_entry not in base_entries or overrides_entry not in base_entries:
        raise ValueError(f"base hybrid stack missing {task_id} entries")
    replacement_data = replacement_model.read_bytes()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in base_entries.items():
            if name in {base_entry, overrides_entry}:
                archive.writestr(name, replacement_data)
            else:
                archive.writestr(name, data)
    with zipfile.ZipFile(output_zip, "r") as archive:
        names = sorted(name for name in archive.namelist() if not name.endswith("/"))
    if names != sorted(base_entries):
        raise ValueError("zip entry set changed")
    return (
        _sha256_bytes(replacement_data),
        _sha256_bytes(base_entries[base_entry]),
        _sha256_bytes(base_entries[overrides_entry]),
    )


def build_one_task_ablations(
    base_6348_zip: str,
    ref6275_dir: str,
    selection_report: str,
    output_dir: str,
    report_path: str,
    task_ids: set[str] | None = None,
    upload_friendly_folders: bool = True,
    inspect_first: bool = True,
) -> dict[str, Any]:
    """Build 6348-base one-task ablations for selected 6275 replacements."""
    base_entries = _validate_hybrid_base_zip(Path(base_6348_zip))
    ref6275_root = Path(ref6275_dir)
    selection_rows = {row["task_id"]: row for row in _load_selection_rows(Path(selection_report))}
    selected_task_ids = sorted(task_ids) if task_ids is not None else _selected_6275_task_ids(Path(selection_report))
    output_root = Path(output_dir)
    report = Path(report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    first_valid_zip = ""
    for task_id in selected_task_ids:
        replacement = ref6275_root / f"{task_id}.onnx"
        output_zip = output_root / f"{task_id}_6275Over6348BothLanes.zip"
        upload_path = output_root / f"{task_id}_6275Over6348BothLanes" / "submission.zip"
        source_row = selection_rows.get(task_id, {})
        try:
            validation = _validate_model(replacement)
            if not validation["valid"]:
                raise ValueError(str(validation["failure_reason"]))
            replacement_sha, base_sha, overrides_sha = _write_one_task_hybrid_ablation(
                base_entries,
                task_id,
                replacement,
                output_zip,
            )
            upload_submission_path = ""
            if upload_friendly_folders:
                upload_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(output_zip, upload_path)
                upload_submission_path = str(upload_path)
            if not first_valid_zip:
                first_valid_zip = str(output_zip)
            rows.append(
                {
                    "task_id": task_id,
                    "candidate_zip_path": str(output_zip),
                    "upload_submission_path": upload_submission_path,
                    "replacement_model_path": str(replacement),
                    "replacement_estimated_cost": validation["estimated_cost"],
                    "best_6348_estimated_cost": source_row.get("best_6348_estimated_cost", ""),
                    "cost_delta_6275_minus_best_6348": source_row.get("cost_delta_6275_minus_best_6348", ""),
                    "base_lane_replaced": True,
                    "overrides_lane_replaced": True,
                    "replacement_sha256": replacement_sha,
                    "base_lane_old_sha256": base_sha,
                    "overrides_lane_old_sha256": overrides_sha,
                    "candidate_valid": True,
                    "failure_reason": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "task_id": task_id,
                    "candidate_zip_path": str(output_zip),
                    "upload_submission_path": "",
                    "replacement_model_path": str(replacement),
                    "replacement_estimated_cost": "",
                    "best_6348_estimated_cost": source_row.get("best_6348_estimated_cost", ""),
                    "cost_delta_6275_minus_best_6348": source_row.get("cost_delta_6275_minus_best_6348", ""),
                    "base_lane_replaced": False,
                    "overrides_lane_replaced": False,
                    "replacement_sha256": "",
                    "base_lane_old_sha256": "",
                    "overrides_lane_old_sha256": "",
                    "candidate_valid": False,
                    "failure_reason": str(exc),
                }
            )

    if inspect_first and first_valid_zip:
        inspect_submission(first_valid_zip, layout="hybrid_stack")

    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ABLATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "base_6348_zip": base_6348_zip,
        "ref6275_dir": str(ref6275_root),
        "output_dir": str(output_root),
        "report_path": str(report),
        "selected_task_count": len(selected_task_ids),
        "valid_zip_count": sum(1 for row in rows if row["candidate_valid"]),
        "failed_count": sum(1 for row in rows if not row["candidate_valid"]),
        "first_valid_zip_inspected": first_valid_zip if inspect_first else "",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_task_ids(raw: str) -> set[str] | None:
    task_ids = {item.strip() for item in raw.split(",") if item.strip()}
    return task_ids or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("local-cost", help="Build flat local-cost candidate")
    local.add_argument("--data-dir", default="task")
    local.add_argument("--ref6275-dir", default="outputs/reference_6275_flat")
    local.add_argument("--ref6348-stack-dir", default="outputs/reference_6348_56_stack")
    local.add_argument(
        "--output-zip",
        default="outputs/ablation_submissions/6348_6275_local_cost_20260612/submission.zip",
    )
    local.add_argument(
        "--report",
        default="outputs/reports/6348_6275_local_cost_selection_20260612.csv",
    )

    ablation = subparsers.add_parser("one-task", help="Build one-task 6275-over-6348 ablations")
    ablation.add_argument("--base-6348-zip", default="outputs/submissions/6348_56_hybrid_stack_submission.zip")
    ablation.add_argument("--ref6275-dir", default="outputs/reference_6275_flat")
    ablation.add_argument(
        "--selection-report",
        default="outputs/reports/6348_6275_local_cost_selection_20260612.csv",
    )
    ablation.add_argument(
        "--output-dir",
        default="outputs/ablation_submissions/6348_56_ref6275_one_task_20260612",
    )
    ablation.add_argument(
        "--report",
        default="outputs/reports/6348_56_ref6275_one_task_20260612.csv",
    )
    ablation.add_argument("--task-ids", default="")
    ablation.add_argument("--no-upload-friendly-folders", action="store_true")
    ablation.add_argument("--no-inspect-first", action="store_true")

    args = parser.parse_args()
    if args.command == "local-cost":
        build_local_cost_candidate(
            data_dir=args.data_dir,
            ref6275_dir=args.ref6275_dir,
            ref6348_stack_dir=args.ref6348_stack_dir,
            output_zip=args.output_zip,
            report_path=args.report,
        )
    elif args.command == "one-task":
        build_one_task_ablations(
            base_6348_zip=args.base_6348_zip,
            ref6275_dir=args.ref6275_dir,
            selection_report=args.selection_report,
            output_dir=args.output_dir,
            report_path=args.report,
            task_ids=_parse_task_ids(args.task_ids),
            upload_friendly_folders=not args.no_upload_friendly_folders,
            inspect_first=not args.no_inspect_first,
        )


if __name__ == "__main__":
    main()
