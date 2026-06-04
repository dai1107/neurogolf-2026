"""Build one-task replacement submission zips for online ablation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from .inspect_submission import TASK_ONNX_RE, inspect_submission


REPLACEMENT_RE = re.compile(r"^(task\d{3})_(.+)\.onnx$")

FIELDS = [
    "task_id",
    "candidate_model_path",
    "candidate_zip_path",
    "base_entry_replaced",
    "inspection_passed",
    "failure_reason",
]


def _candidate_models(candidate_dir: Path, task_ids: set[str] | None) -> list[Path]:
    paths = []
    for path in sorted(candidate_dir.glob("task*.onnx")):
        match = REPLACEMENT_RE.match(path.name)
        if match is None:
            continue
        task_id = match.group(1)
        if task_ids is not None and task_id not in task_ids:
            continue
        paths.append(path)
    return paths


def _task_id_from_candidate(path: Path) -> str:
    match = REPLACEMENT_RE.match(path.name)
    if match is None:
        raise ValueError(f"candidate name must be taskNNN_RuleName.onnx: {path.name}")
    return match.group(1)


def _write_single_replacement_zip(base_zip: Path, candidate_model: Path, output_zip: Path) -> None:
    task_id = _task_id_from_candidate(candidate_model)
    entry_name = f"{task_id}.onnx"
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(base_zip, "r") as base:
        names = sorted(base.namelist())
        if entry_name not in names:
            raise ValueError(f"base submission does not contain {entry_name}")
        for name in names:
            if not TASK_ONNX_RE.match(name):
                raise ValueError(f"invalid base submission entry: {name}")

        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                if name == entry_name:
                    archive.write(candidate_model, arcname=name)
                else:
                    archive.writestr(name, base.read(name))


def build_ablation_submissions(
    base_zip: str,
    candidate_dir: str,
    output_dir: str,
    report_path: str,
    task_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Create one submission zip per candidate model without changing the base zip."""
    base_path = Path(base_zip)
    candidate_root = Path(candidate_dir)
    output_root = Path(output_dir)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    inspect_submission(str(base_path))

    rows: list[dict[str, Any]] = []
    candidates = _candidate_models(candidate_root, task_ids)
    for candidate in candidates:
        task_id = _task_id_from_candidate(candidate)
        output_zip = output_root / f"{candidate.stem}.zip"
        try:
            _write_single_replacement_zip(base_path, candidate, output_zip)
            inspect_submission(str(output_zip))
            rows.append(
                {
                    "task_id": task_id,
                    "candidate_model_path": str(candidate),
                    "candidate_zip_path": str(output_zip),
                    "base_entry_replaced": f"{task_id}.onnx",
                    "inspection_passed": True,
                    "failure_reason": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "task_id": task_id,
                    "candidate_model_path": str(candidate),
                    "candidate_zip_path": str(output_zip),
                    "base_entry_replaced": f"{task_id}.onnx",
                    "inspection_passed": False,
                    "failure_reason": str(exc),
                }
            )

    with Path(report_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "base_zip": str(base_path),
        "candidate_dir": str(candidate_root),
        "output_dir": str(output_root),
        "report_path": report_path,
        "candidate_count": len(candidates),
        "valid_zip_count": sum(1 for row in rows if row["inspection_passed"]),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_task_ids(raw: str) -> set[str] | None:
    task_ids = {item.strip() for item in raw.split(",") if item.strip()}
    return task_ids or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-zip", default="outputs/submission.zip")
    parser.add_argument("--candidate-dir", default="outputs/candidates/replacements")
    parser.add_argument("--output-dir", default="outputs/ablation_submissions")
    parser.add_argument("--report", default="outputs/reports/ablation_submission_report.csv")
    parser.add_argument("--task-ids", default="")
    args = parser.parse_args()
    build_ablation_submissions(
        base_zip=args.base_zip,
        candidate_dir=args.candidate_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        task_ids=_parse_task_ids(args.task_ids),
    )


if __name__ == "__main__":
    main()
