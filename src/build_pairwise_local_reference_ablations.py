"""Build one-task local-over-reference ablations from two model banks.

This is intended for online attribution after an aggregate hybrid regresses:
keep a known reference submission as the base, replace exactly one task with
the local model in each generated zip, and submit those zips one at a time.
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

from .cost_estimator import FILE_SIZE_LIMIT_BYTES, check_forbidden_ops, estimate_model_cost
from .inspect_submission import TASK_ONNX_RE, inspect_submission


FIELDS = [
    "task_id",
    "selected_by",
    "replacement_model_path",
    "candidate_zip_path",
    "upload_submission_path",
    "archive_estimated_cost",
    "current_estimated_cost",
    "cost_delta_archive_minus_current",
    "base_entry_size_bytes",
    "replacement_file_size_bytes",
    "base_sha256",
    "replacement_sha256",
    "candidate_valid",
    "failure_reason",
]


def _parse_task_ids(raw: str) -> set[str] | None:
    task_ids = {item.strip() for item in raw.split(",") if item.strip()}
    return task_ids or None


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_base_zip(base_zip: Path) -> dict[str, bytes]:
    inspect_submission(str(base_zip))
    entries: dict[str, bytes] = {}
    with zipfile.ZipFile(base_zip, "r") as archive:
        for name in sorted(archive.namelist()):
            if not TASK_ONNX_RE.match(name):
                raise ValueError(f"invalid base zip entry: {name}")
            entries[name] = archive.read(name)
    return entries


def _task_ids_from_report(report_path: Path, selection: str) -> tuple[list[str], dict[str, dict[str, str]]]:
    if not report_path.is_file():
        raise FileNotFoundError(f"selection report does not exist: {report_path}")

    rows_by_task: dict[str, dict[str, str]] = {}
    selected: list[str] = []
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            task_id = row.get("task_id", "").strip()
            if not task_id:
                continue
            rows_by_task[task_id] = row
            if selection == "selected-current" and row.get("selected_source") == "current":
                selected.append(task_id)
            elif selection == "current-cheaper" and _int_or_none(row.get("cost_delta_ref_minus_current")):
                if int(row["cost_delta_ref_minus_current"]) > 0:
                    selected.append(task_id)
            elif selection == "all":
                selected.append(task_id)
    return selected, rows_by_task


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _report_cost(row: dict[str, str], key: str) -> str:
    value = row.get(key, "")
    return value if value is not None else ""


def _cost_delta(row: dict[str, str]) -> str:
    if "cost_delta_ref_minus_current" in row:
        return _report_cost(row, "cost_delta_ref_minus_current")
    archive_cost = _int_or_none(row.get("archive_estimated_cost"))
    current_cost = _int_or_none(row.get("current_estimated_cost"))
    if archive_cost is None or current_cost is None:
        return ""
    return str(archive_cost - current_cost)


def _validate_replacement(model_path: Path) -> dict[str, Any]:
    if not model_path.is_file():
        return {"valid": False, "failure_reason": "missing_replacement_model"}
    try:
        cost = estimate_model_cost(str(model_path))
        forbidden = check_forbidden_ops(str(model_path))
    except Exception as exc:
        return {"valid": False, "failure_reason": f"replacement_validation_exception: {exc}"}

    failures: list[str] = []
    if not cost["file_size_ok"]:
        failures.append("file_size_exceeds_limit")
    if not forbidden["passed"]:
        failures.append(f"forbidden_ops={forbidden['forbidden_ops_found']}")
    return {**cost, "valid": not failures, "failure_reason": "; ".join(failures)}


def _write_replacement_zip(
    base_entries: dict[str, bytes],
    task_id: str,
    replacement_model: Path,
    output_zip: Path,
) -> None:
    entry_name = f"{task_id}.onnx"
    if entry_name not in base_entries:
        raise ValueError(f"base submission does not contain {entry_name}")

    replacement_size = replacement_model.stat().st_size
    if replacement_size > FILE_SIZE_LIMIT_BYTES:
        raise ValueError(f"replacement exceeds file size limit: {replacement_size}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in base_entries.items():
            if name == entry_name:
                archive.write(replacement_model, arcname=name)
            else:
                archive.writestr(name, data)

    with zipfile.ZipFile(output_zip, "r") as archive:
        names = archive.namelist()
        if len(names) != len(base_entries):
            raise ValueError(f"zip entry count changed: {len(names)} != {len(base_entries)}")
        if sorted(names) != sorted(base_entries):
            raise ValueError("zip entry set changed")


def build_pairwise_local_reference_ablations(
    base_zip: str,
    replacement_dir: str,
    output_dir: str,
    report_path: str,
    selection_report: str = "",
    selection: str = "selected-current",
    task_ids: set[str] | None = None,
    replacement_label: str = "LocalOverReference",
    upload_friendly_folders: bool = True,
    skip_identical: bool = True,
) -> dict[str, Any]:
    """Generate one local-over-reference replacement zip per selected task."""
    if selection not in {"selected-current", "current-cheaper", "all"}:
        raise ValueError(f"unsupported selection: {selection}")

    base_path = Path(base_zip)
    replacement_root = Path(replacement_dir)
    output_root = Path(output_dir)
    report = Path(report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    base_entries = _validate_base_zip(base_path)
    report_rows: dict[str, dict[str, str]] = {}
    if task_ids is None:
        if not selection_report:
            selected_task_ids = sorted(path.stem for path in replacement_root.glob("task*.onnx"))
        else:
            selected_task_ids, report_rows = _task_ids_from_report(Path(selection_report), selection)
    else:
        selected_task_ids = sorted(task_ids)
        if selection_report:
            _, report_rows = _task_ids_from_report(Path(selection_report), "all")

    rows: list[dict[str, Any]] = []
    for task_id in sorted(set(selected_task_ids)):
        entry_name = f"{task_id}.onnx"
        replacement_path = replacement_root / entry_name
        output_zip = output_root / f"{task_id}_{replacement_label}.zip"
        upload_path = output_root / f"{task_id}_{replacement_label}" / "submission.zip"
        row = report_rows.get(task_id, {})
        base_data = base_entries.get(entry_name, b"")
        base_sha = _sha256_bytes(base_data) if base_data else ""
        replacement_sha = ""
        try:
            replacement_data = replacement_path.read_bytes()
            replacement_sha = _sha256_bytes(replacement_data)
            if skip_identical and replacement_sha == base_sha:
                rows.append(
                    {
                        "task_id": task_id,
                        "selected_by": selection,
                        "replacement_model_path": str(replacement_path),
                        "candidate_zip_path": "",
                        "upload_submission_path": "",
                        "archive_estimated_cost": _report_cost(row, "archive_estimated_cost"),
                        "current_estimated_cost": _report_cost(row, "current_estimated_cost"),
                        "cost_delta_archive_minus_current": _cost_delta(row),
                        "base_entry_size_bytes": len(base_data),
                        "replacement_file_size_bytes": replacement_path.stat().st_size,
                        "base_sha256": base_sha,
                        "replacement_sha256": replacement_sha,
                        "candidate_valid": False,
                        "failure_reason": "skipped_identical_to_base",
                    }
                )
                continue

            validation = _validate_replacement(replacement_path)
            if not validation["valid"]:
                raise ValueError(str(validation["failure_reason"]))
            _write_replacement_zip(base_entries, task_id, replacement_path, output_zip)
            if upload_friendly_folders:
                upload_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(output_zip, upload_path)
                upload_submission_path = str(upload_path)
            else:
                upload_submission_path = ""

            rows.append(
                {
                    "task_id": task_id,
                    "selected_by": selection,
                    "replacement_model_path": str(replacement_path),
                    "candidate_zip_path": str(output_zip),
                    "upload_submission_path": upload_submission_path,
                    "archive_estimated_cost": _report_cost(row, "archive_estimated_cost"),
                    "current_estimated_cost": _report_cost(row, "current_estimated_cost"),
                    "cost_delta_archive_minus_current": _cost_delta(row),
                    "base_entry_size_bytes": len(base_data),
                    "replacement_file_size_bytes": replacement_path.stat().st_size,
                    "base_sha256": base_sha,
                    "replacement_sha256": replacement_sha,
                    "candidate_valid": True,
                    "failure_reason": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "task_id": task_id,
                    "selected_by": selection,
                    "replacement_model_path": str(replacement_path),
                    "candidate_zip_path": str(output_zip),
                    "upload_submission_path": "",
                    "archive_estimated_cost": _report_cost(row, "archive_estimated_cost"),
                    "current_estimated_cost": _report_cost(row, "current_estimated_cost"),
                    "cost_delta_archive_minus_current": _cost_delta(row),
                    "base_entry_size_bytes": len(base_data),
                    "replacement_file_size_bytes": replacement_path.stat().st_size if replacement_path.exists() else "",
                    "base_sha256": base_sha,
                    "replacement_sha256": replacement_sha,
                    "candidate_valid": False,
                    "failure_reason": str(exc),
                }
            )

    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "base_zip": str(base_path),
        "replacement_dir": str(replacement_root),
        "output_dir": str(output_root),
        "report_path": str(report),
        "selected_task_count": len(set(selected_task_ids)),
        "replacement_label": replacement_label,
        "valid_zip_count": sum(1 for row in rows if row["candidate_valid"]),
        "skipped_identical_count": sum(1 for row in rows if row["failure_reason"] == "skipped_identical_to_base"),
        "failed_count": sum(
            1
            for row in rows
            if not row["candidate_valid"] and row["failure_reason"] != "skipped_identical_to_base"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-zip", required=True)
    parser.add_argument("--replacement-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--selection-report", default="")
    parser.add_argument("--selection", choices=["selected-current", "current-cheaper", "all"], default="selected-current")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--replacement-label", default="LocalOverReference")
    parser.add_argument("--no-upload-friendly-folders", action="store_true")
    parser.add_argument("--include-identical", action="store_true")
    args = parser.parse_args()
    build_pairwise_local_reference_ablations(
        base_zip=args.base_zip,
        replacement_dir=args.replacement_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        selection_report=args.selection_report,
        selection=args.selection,
        task_ids=_parse_task_ids(args.task_ids),
        replacement_label=args.replacement_label,
        upload_friendly_folders=not args.no_upload_friendly_folders,
        skip_identical=not args.include_identical,
    )


if __name__ == "__main__":
    main()
