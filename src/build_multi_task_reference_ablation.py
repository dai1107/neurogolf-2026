"""Build one multi-task replacement submission from a reference zip."""

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
    "ordinal",
    "task_id",
    "replacement_model_path",
    "source_estimated_cost",
    "replacement_estimated_cost",
    "cost_delta",
    "base_entry_size_bytes",
    "replacement_file_size_bytes",
    "base_sha256",
    "replacement_sha256",
    "candidate_valid",
    "failure_reason",
]


def _parse_ordinals(raw: str) -> list[int]:
    ordinals = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not ordinals:
        raise ValueError("no ordinals provided")
    if any(value <= 0 for value in ordinals):
        raise ValueError(f"ordinals are 1-based positive integers: {ordinals}")
    if len(set(ordinals)) != len(ordinals):
        raise ValueError(f"duplicate ordinals: {ordinals}")
    return ordinals


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


def _load_report_rows(selection_report: Path) -> list[dict[str, str]]:
    if not selection_report.is_file():
        raise FileNotFoundError(f"selection report does not exist: {selection_report}")
    with selection_report.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("task_id", "").strip()]
    return sorted(rows, key=lambda row: row["task_id"])


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


def build_multi_task_reference_ablation(
    base_zip: str,
    replacement_dir: str,
    selection_report: str,
    ordinals: list[int],
    output_zip: str,
    upload_path: str,
    report_path: str,
) -> dict[str, Any]:
    """Replace all selected report-ordinal tasks in one reference submission."""
    base_entries = _validate_base_zip(Path(base_zip))
    replacement_root = Path(replacement_dir)
    report_rows = _load_report_rows(Path(selection_report))

    max_ordinal = max(ordinals)
    if max_ordinal > len(report_rows):
        raise ValueError(f"ordinal {max_ordinal} exceeds report row count {len(report_rows)}")

    selected_rows = [(ordinal, report_rows[ordinal - 1]) for ordinal in ordinals]
    output = Path(output_zip)
    upload = Path(upload_path)
    report = Path(report_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    upload.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    replacement_by_entry: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    total_cost_delta = 0

    for ordinal, source_row in selected_rows:
        task_id = source_row["task_id"].strip()
        entry_name = f"{task_id}.onnx"
        if entry_name not in base_entries:
            raise ValueError(f"base submission does not contain {entry_name}")

        model_path = replacement_root / entry_name
        validation = _validate_replacement(model_path)
        base_data = base_entries[entry_name]
        replacement_data = model_path.read_bytes() if model_path.is_file() else b""
        source_cost = estimate_model_cost_from_bytes(base_data)
        replacement_cost = int(validation.get("estimated_cost", 0)) if validation["valid"] else 0
        cost_delta = replacement_cost - source_cost if validation["valid"] else ""

        if not validation["valid"]:
            rows.append(
                {
                    "ordinal": ordinal,
                    "task_id": task_id,
                    "replacement_model_path": str(model_path),
                    "source_estimated_cost": source_cost,
                    "replacement_estimated_cost": replacement_cost or "",
                    "cost_delta": cost_delta,
                    "base_entry_size_bytes": len(base_data),
                    "replacement_file_size_bytes": model_path.stat().st_size if model_path.exists() else "",
                    "base_sha256": _sha256_bytes(base_data),
                    "replacement_sha256": _sha256_bytes(replacement_data) if replacement_data else "",
                    "candidate_valid": False,
                    "failure_reason": validation["failure_reason"],
                }
            )
            continue

        if model_path.stat().st_size > FILE_SIZE_LIMIT_BYTES:
            raise ValueError(f"{task_id} replacement exceeds file size limit")

        replacement_by_entry[entry_name] = model_path
        total_cost_delta += int(cost_delta)
        rows.append(
            {
                "ordinal": ordinal,
                "task_id": task_id,
                "replacement_model_path": str(model_path),
                "source_estimated_cost": source_cost,
                "replacement_estimated_cost": replacement_cost,
                "cost_delta": cost_delta,
                "base_entry_size_bytes": len(base_data),
                "replacement_file_size_bytes": model_path.stat().st_size,
                "base_sha256": _sha256_bytes(base_data),
                "replacement_sha256": _sha256_bytes(replacement_data),
                "candidate_valid": True,
                "failure_reason": "",
            }
        )

    invalid = [row for row in rows if not row["candidate_valid"]]
    if invalid:
        with report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        raise ValueError(f"invalid replacements: {[row['task_id'] for row in invalid]}")

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in base_entries.items():
            replacement = replacement_by_entry.get(name)
            if replacement is None:
                archive.writestr(name, data)
            else:
                archive.write(replacement, arcname=name)

    with zipfile.ZipFile(output, "r") as archive:
        names = archive.namelist()
        if sorted(names) != sorted(base_entries):
            raise ValueError("zip entry set changed")

    inspect_submission(str(output))
    shutil.copyfile(output, upload)
    inspect_submission(str(upload))

    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "base_zip": str(base_zip),
        "output_zip": str(output),
        "upload_submission_path": str(upload),
        "report_path": str(report),
        "selected_count": len(selected_rows),
        "selected_tasks": [row["task_id"] for row in rows],
        "total_cost_delta": total_cost_delta,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def estimate_model_cost_from_bytes(data: bytes) -> int:
    """Estimate one embedded model by writing a short-lived temp file."""
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
        path = Path(handle.name)
        handle.write(data)
    try:
        return int(estimate_model_cost(str(path))["estimated_cost"])
    finally:
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-zip", required=True)
    parser.add_argument("--replacement-dir", required=True)
    parser.add_argument("--selection-report", required=True)
    parser.add_argument("--ordinals", required=True)
    parser.add_argument("--output-zip", required=True)
    parser.add_argument("--upload-path", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    build_multi_task_reference_ablation(
        base_zip=args.base_zip,
        replacement_dir=args.replacement_dir,
        selection_report=args.selection_report,
        ordinals=_parse_ordinals(args.ordinals),
        output_zip=args.output_zip,
        upload_path=args.upload_path,
        report_path=args.report,
    )


if __name__ == "__main__":
    main()
