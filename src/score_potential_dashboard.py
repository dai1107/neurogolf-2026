"""Build the theoretical score-gain ceiling dashboard for all tasks.

Reads current cost data and task metadata, computes how much each task's score
would improve if its cost were reduced to various thresholds. This identifies
which tasks are worth attacking in batches rather than focusing on top-cost tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from .arc_io import load_task


DEFAULT_REPORT = "outputs/reports/score_potential_dashboard.csv"
COST_THRESHOLDS = [10000, 3000, 1000, 300]
FIELDS = [
    "task_id",
    "current_cost",
    "current_score",
    "score_if_10000",
    "score_if_3000",
    "score_if_1000",
    "score_if_300",
    "potential_to_1000",
    "potential_to_300",
    "gain_category",
    "same_shape",
    "num_colors_in",
    "num_colors_out",
    "shape_relation",
    "current_model_type",
    "task_family",
    "online_history",
]


def compute_score(cost: float) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def load_cost_report(report_path: str) -> dict[str, dict[str, Any]]:
    """Read current_model_bank_report.csv, return {task_id: {cost, file_size, model_path}}."""
    data: dict[str, dict[str, Any]] = {}
    with Path(report_path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tid = row["task_id"].strip()
            if row.get("valid", "True").strip().lower() != "true":
                continue
            cost = int(row.get("estimated_cost") or 0)
            data[tid] = {
                "current_cost": cost,
                "file_size_bytes": int(row.get("file_size_bytes") or 0),
                "model_path": row.get("model_path", "").strip(),
            }
    return data


def load_task_metadata(task_dir: str) -> dict[str, dict[str, Any]]:
    """Extract basic metadata from task JSONs: shapes, colors, same_shape flag."""
    root = Path(task_dir)
    meta: dict[str, dict[str, Any]] = {}
    for path in sorted(root.glob("task*.json")):
        tid = path.stem
        try:
            task = load_task(str(path))
        except Exception:
            continue
        train = task.get("train", [])
        if not train:
            continue

        shapes_in = [(len(c["input"]), len(c["input"][0])) for c in train]
        shapes_out = [(len(c["output"]), len(c["output"][0])) for c in train]

        colors_in = set()
        colors_out = set()
        for c in train:
            for row in c["input"]:
                colors_in.update(row)
            for row in c["output"]:
                colors_out.update(row)

        same_shape = all(si == so for si, so in zip(shapes_in, shapes_out))

        if all(so[0]*so[1] < si[0]*si[1] for si, so in zip(shapes_in, shapes_out)):
            shape_relation = "crop"
        elif all(so[0]*so[1] > si[0]*si[1] for si, so in zip(shapes_in, shapes_out)):
            shape_relation = "expand"
        elif same_shape:
            shape_relation = "same_shape"
        else:
            shape_relation = "mixed"

        current_model_type = "unknown"
        if same_shape and colors_in == colors_out:
            current_model_type = "same_shape_mask_or_remap"
        elif shape_relation == "crop":
            current_model_type = "crop_or_extract"
        elif shape_relation == "expand":
            current_model_type = "scale_or_tile"

        meta[tid] = {
            "same_shape": same_shape,
            "num_colors_in": len(colors_in),
            "num_colors_out": len(colors_out),
            "shape_relation": shape_relation,
            "current_model_type": current_model_type,
        }
    return meta


def load_online_history(reports_dir: str) -> dict[str, list[str]]:
    """Scan ablation reports and online_ablation_results for task-level history."""
    history: dict[str, list[str]] = {}
    root = Path(reports_dir)
    for path in sorted(root.glob("*.csv")):
        name = path.stem
        for keyword in ["ablation", "online"]:
            if keyword not in name.lower():
                continue
            try:
                with path.open("r", newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            except Exception:
                continue
            for row in rows:
                tid = row.get("task_id", "").strip()
                if not tid:
                    continue
                decision = row.get("decision", "").strip()
                online_delta = row.get("online_delta", "").strip()
                candidate = row.get("candidate_name", "").strip() or row.get(
                    "candidate_model_path", ""
                ).strip() or row.get("candidate_zip_path", "").strip()
                online_score = row.get("online_score", "").strip()

                parts: list[str] = []
                if candidate:
                    parts.append(candidate.replace("outputs\\", "").replace(
                        "outputs/", ""
                    ))
                if online_delta:
                    parts.append(f"delta={online_delta}")
                if online_score:
                    parts.append(f"online_score={online_score}")
                if decision:
                    parts.append(decision)
                if parts:
                    entry = " | ".join(parts)
                    history.setdefault(tid, []).append(entry)
    return history


def compute_dashboard(
    cost_report_path: str = "outputs/reports/current_model_bank_report.csv",
    task_dir: str = "task",
    report_path: str = DEFAULT_REPORT,
    reports_dir: str = "outputs/reports",
) -> dict[str, Any]:
    cost_data = load_cost_report(cost_report_path)
    task_meta = load_task_metadata(task_dir)
    online_history = load_online_history(reports_dir)

    rows: list[dict[str, Any]] = []
    for tid in sorted(cost_data):
        cost_info = cost_data[tid]
        current_cost = cost_info["current_cost"]
        current_score = compute_score(current_cost)

        scores = {f"score_if_{t}": compute_score(t) for t in COST_THRESHOLDS}
        row = {
            "task_id": tid,
            "current_cost": current_cost,
            "current_score": round(current_score, 4),
            "score_if_10000": round(scores["score_if_10000"], 4),
            "score_if_3000": round(scores["score_if_3000"], 4),
            "score_if_1000": round(scores["score_if_1000"], 4),
            "score_if_300": round(scores["score_if_300"], 4),
            "potential_to_1000": round(scores["score_if_1000"] - current_score, 4),
            "potential_to_300": round(scores["score_if_300"] - current_score, 4),
        }

        potential = row["potential_to_1000"]
        if current_score >= 18.09:
            row["gain_category"] = "already_good"
        elif potential >= 0.5:
            row["gain_category"] = "high_potential"
        elif potential >= 0.2:
            row["gain_category"] = "medium_potential"
        elif potential >= 0.01:
            row["gain_category"] = "low_potential"
        else:
            row["gain_category"] = "minimal"

        meta = task_meta.get(tid, {})
        row["same_shape"] = meta.get("same_shape", "")
        row["num_colors_in"] = meta.get("num_colors_in", "")
        row["num_colors_out"] = meta.get("num_colors_out", "")
        row["shape_relation"] = meta.get("shape_relation", "")
        row["current_model_type"] = meta.get("current_model_type", "")
        row["task_family"] = ""
        row["online_history"] = "; ".join(online_history.get(tid, []))

        rows.append(row)

    rows.sort(key=lambda r: (-r["potential_to_1000"], r["task_id"]))

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_cat: dict[str, int] = {}
    total_potential = 0.0
    for r in rows:
        by_cat[r["gain_category"]] = by_cat.get(r["gain_category"], 0) + 1
        total_potential += r["potential_to_1000"]

    summary = {
        "report_path": str(report),
        "task_count": len(rows),
        "by_gain_category": by_cat,
        "total_potential_to_1000": round(total_potential, 2),
        "top_opportunities": [
            {k: r[k] for k in ("task_id", "current_cost", "current_score", "potential_to_1000", "gain_category")}
            for r in rows[:20]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cost-report",
        default="outputs/reports/current_model_bank_report.csv",
    )
    parser.add_argument("--task-dir", default="task")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--reports-dir", default="outputs/reports")
    args = parser.parse_args()
    compute_dashboard(
        cost_report_path=args.cost_report,
        task_dir=args.task_dir,
        report_path=args.report,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
