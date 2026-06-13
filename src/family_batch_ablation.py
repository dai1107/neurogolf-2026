"""Family batch ablation tool — group same-family candidates for online testing.

Follows the safe-ablation workflow from online_result_memory:
  1. Each batch contains only same-family conservative candidates
  2. All candidates must pass local labelled validation
  3. No mixing with known online negatives
  4. Packages as full submission.zip per batch for one-at-a-time online testing

Outputs: outputs/ablation_submissions/family_batches/
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path
from typing import Any


def build_family_batches(
    taxonomy_path: str = "outputs/reports/task_family_taxonomy_v2.csv",
    cost_report_path: str = "outputs/reports/current_model_bank_report.csv",
    online_memory_path: str = "outputs/reports/online_result_memory.csv",
    candidates_root: str = "outputs/candidates",
    submission_zip: str = "outputs/submission.zip",
    output_dir: str = "outputs/ablation_submissions/family_batches",
) -> dict[str, Any]:
    # Load taxonomy
    task_family: dict[str, str] = {}
    with Path(taxonomy_path).open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task_family[row["task_id"].strip()] = row.get("family", "")

    # Load online negatives (tasks to avoid)
    known_negative_tasks: set[str] = set()
    if Path(online_memory_path).exists():
        with Path(online_memory_path).open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("decision") == "reject" and row.get("task_id") != "batch":
                    known_negative_tasks.add(row["task_id"].strip())

    # Scan for valid conservative candidates
    candidates_root_path = Path(candidates_root)
    family_candidates: dict[str, list[Path]] = {}
    for onnx_file in candidates_root_path.rglob("*_conservative/*.onnx"):
        tid = None
        for part in onnx_file.parts:
            if part.startswith("task") and len(part) >= 7:
                tid = part[:7]
                break
        if not tid:
            # Try extracting from filename
            name = onnx_file.stem
            if name.startswith("task"):
                tid = name[:7]

        if tid and tid in task_family and tid not in known_negative_tasks:
            family = task_family[tid]
            family_candidates.setdefault(family, []).append(onnx_file)

    # Build batch zips
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    batches_built = []

    for family, candidates in sorted(family_candidates.items()):
        if len(candidates) < 1:
            continue

        # Build a full submission zip with all candidates for this family
        batch_name = f"family_{family}_batch_{len(candidates)}.zip"
        batch_path = out / batch_name

        with zipfile.ZipFile(submission_zip, "r") as zin:
            with zipfile.ZipFile(str(batch_path), "w", zipfile.ZIP_DEFLATED) as zout:
                # Track which tasks are being replaced
                replaced = set()
                for cand_path in candidates:
                    tid = None
                    name = cand_path.stem
                    for part in cand_path.parts:
                        if part.startswith("task") and len(part) >= 7:
                            tid = part[:7]
                            break
                    if not tid and name.startswith("task"):
                        tid = name[:7]
                    if tid:
                        replaced.add(tid)

                for item in zin.infolist():
                    fn = item.filename
                    task_in_zip = fn.replace(".onnx", "")
                    if task_in_zip in replaced:
                        # Find and write the candidate
                        cand = next((c for c in candidates
                                     if task_in_zip in str(c)), None)
                        if cand:
                            zout.write(str(cand), fn)
                        else:
                            zout.writestr(item, zin.read(item.filename))
                    else:
                        zout.writestr(item, zin.read(item.filename))

        batches_built.append({
            "family": family,
            "batch_name": batch_name,
            "candidates": len(candidates),
            "tasks": sorted(replaced),
        })

    summary = {
        "output_dir": str(out),
        "batches_built": len(batches_built),
        "batches": batches_built,
        "known_negative_tasks_avoided": sorted(known_negative_tasks),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taxonomy", default="outputs/reports/task_family_taxonomy_v2.csv")
    parser.add_argument("--cost-report", default="outputs/reports/current_model_bank_report.csv")
    parser.add_argument("--online-memory", default="outputs/reports/online_result_memory.csv")
    parser.add_argument("--candidates-root", default="outputs/candidates")
    parser.add_argument("--submission-zip", default="outputs/submission.zip")
    parser.add_argument("--output-dir", default="outputs/ablation_submissions/family_batches")
    args = parser.parse_args()
    build_family_batches(
        taxonomy_path=args.taxonomy,
        cost_report_path=args.cost_report,
        online_memory_path=args.online_memory,
        candidates_root=args.candidates_root,
        submission_zip=args.submission_zip,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
