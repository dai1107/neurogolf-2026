"""Select the best known-online submission zip and mirror its model stack."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inspect_submission import inspect_submission


FIELDS = [
    "label",
    "zip_path",
    "known_online_score",
    "layout",
    "num_models",
    "num_task_ids",
    "zip_size_bytes",
    "selected",
    "failure_reason",
]


@dataclass(frozen=True)
class SubmissionCandidate:
    label: str
    zip_path: str
    known_online_score: float


def _safe_remove_tree(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"refusing to remove path outside allowed root: {path}")
    if path.exists():
        shutil.rmtree(path)


def _safe_extract_zip(zip_path: Path, extract_dir: Path) -> None:
    temp_root = extract_dir.parent / f".{extract_dir.name}_extract_tmp"
    allowed_root = extract_dir.parent
    _safe_remove_tree(temp_root, allowed_root)
    temp_root.mkdir(parents=True, exist_ok=False)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            for name in archive.namelist():
                if name.endswith("/"):
                    continue
                destination = temp_root / name
                if temp_root.resolve() not in destination.resolve().parents:
                    raise ValueError(f"unsafe zip entry path: {name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(name))
        _safe_remove_tree(extract_dir, allowed_root)
        shutil.move(str(temp_root), str(extract_dir))
    finally:
        _safe_remove_tree(temp_root, allowed_root)


def _candidate_row(candidate: SubmissionCandidate) -> dict[str, Any]:
    path = Path(candidate.zip_path)
    try:
        inspection = inspect_submission(str(path))
        return {
            "label": candidate.label,
            "zip_path": str(path),
            "known_online_score": candidate.known_online_score,
            "layout": inspection.get("layout", "flat"),
            "num_models": inspection["num_models"],
            "num_task_ids": inspection.get("num_task_ids", inspection["num_models"]),
            "zip_size_bytes": path.stat().st_size,
            "selected": False,
            "failure_reason": "",
        }
    except Exception as exc:
        return {
            "label": candidate.label,
            "zip_path": str(path),
            "known_online_score": candidate.known_online_score,
            "layout": "",
            "num_models": "",
            "num_task_ids": "",
            "zip_size_bytes": path.stat().st_size if path.exists() else "",
            "selected": False,
            "failure_reason": str(exc),
        }


def select_best_submission(
    candidates: list[SubmissionCandidate],
    output_zip: str,
    report_path: str,
    extract_dir: str = "",
) -> dict[str, Any]:
    """Validate candidates, select highest known score, and copy the zip.

    This intentionally uses known online score for the primary decision. Local
    cost estimates are useful for diagnostics, but previous aggregate hybrids in
    this repo proved they are not reliable enough for promotion decisions.
    """
    if not candidates:
        raise ValueError("at least one candidate is required")

    rows = [_candidate_row(candidate) for candidate in candidates]
    valid_rows = [row for row in rows if not row["failure_reason"]]
    if not valid_rows:
        raise ValueError("no valid submission candidates")

    selected = max(
        valid_rows,
        key=lambda row: (
            float(row["known_online_score"]),
            -int(row["zip_size_bytes"]),
            str(row["label"]),
        ),
    )
    selected["selected"] = True

    output = Path(output_zip)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected["zip_path"], output)

    extracted_to = ""
    if extract_dir:
        extract_root = Path(extract_dir)
        extract_root.parent.mkdir(parents=True, exist_ok=True)
        _safe_extract_zip(output, extract_root)
        extracted_to = str(extract_root)

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "selected_label": selected["label"],
        "selected_score": selected["known_online_score"],
        "selected_layout": selected["layout"],
        "selected_num_models": selected["num_models"],
        "output_zip": str(output),
        "report_path": str(report),
        "extract_dir": extracted_to,
        "candidate_count": len(candidates),
        "valid_candidate_count": len(valid_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_candidate(raw: str) -> SubmissionCandidate:
    parts = raw.split("=", 2)
    if len(parts) != 3:
        raise ValueError("candidate must have form label=score=zip_path")
    label, score_text, zip_path = parts
    if not label:
        raise ValueError("candidate label is empty")
    return SubmissionCandidate(
        label=label,
        known_online_score=float(score_text),
        zip_path=zip_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate in label=known_online_score=zip_path form; repeatable.",
    )
    parser.add_argument("--output-zip", default="outputs/submission.zip")
    parser.add_argument("--report", default="outputs/reports/submission_take_best.csv")
    parser.add_argument("--extract-dir", default="")
    args = parser.parse_args()
    select_best_submission(
        candidates=[_parse_candidate(raw) for raw in args.candidate],
        output_zip=args.output_zip,
        report_path=args.report,
        extract_dir=args.extract_dir,
    )


if __name__ == "__main__":
    main()
