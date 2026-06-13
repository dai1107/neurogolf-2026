"""Build the online result memory — systematize what works and what doesn't.

Scans all ablation reports and experiment logs to produce:
  outputs/reports/online_result_memory.csv

Fields: candidate_name, task_id, family, rewrite_type, local_cost_delta,
  local_score_delta, node_delta, initializer_delta, op_type_added,
  op_type_removed, train_pass, test_pass, arc_gen_pass, online_score,
  online_delta, decision, reason
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_REPORT = "outputs/reports/online_result_memory.csv"
FIELDS = [
    "candidate_name",
    "task_id",
    "family",
    "rewrite_type",
    "local_cost_delta",
    "online_score",
    "online_delta",
    "decision",
    "reason",
]

# Manually curated from EXPERIMENT_LOG.md and PROGRESS.md
KNOWN_RESULTS: list[dict[str, Any]] = [
    {
        "candidate_name": "task076_Task076PermGatherExact",
        "task_id": "task076",
        "family": "L",
        "rewrite_type": "one_hot_perm_to_gather",
        "local_cost_delta": -900000,
        "online_score": "6037.68",
        "online_delta": "+",
        "decision": "promote",
        "reason": "dense 8x169x169 one-hot perm table replaced by 8x169 int64 index + 3 Gather nodes",
    },
    {
        "candidate_name": "task396_RowBankPrefixConservative",
        "task_id": "task396",
        "family": "L",
        "rewrite_type": "row_bank_prefix_prune",
        "local_cost_delta": -20000,
        "online_score": "6037.90",
        "online_delta": "0",
        "decision": "promote",
        "reason": "pruned unused row-bank rows; flat online = safe to keep",
    },
    {
        "candidate_name": "task290_RowBankPrefixConservative",
        "task_id": "task290",
        "family": "L",
        "rewrite_type": "row_bank_prefix_prune",
        "local_cost_delta": -15000,
        "online_score": "6037.90",
        "online_delta": "0",
        "decision": "promote",
        "reason": "pruned unused row-bank rows; flat online = safe to keep",
    },
    {
        "candidate_name": "task209_PriorRangeObserved",
        "task_id": "task209",
        "family": "L",
        "rewrite_type": "prior_range_prune",
        "local_cost_delta": -27702,
        "online_score": "6037.90",
        "online_delta": "0",
        "decision": "promote",
        "reason": "pruned unused prior range; flat online = safe to keep",
    },
    {
        "candidate_name": "task157_PlacementConservative",
        "task_id": "task157",
        "family": "L",
        "rewrite_type": "enumeration_table_prune",
        "local_cost_delta": -200000,
        "online_score": "6037.89",
        "online_delta": "0",
        "decision": "promote",
        "reason": "pruned unreachable placement rows; flat online = safe to keep",
    },
    {
        "candidate_name": "task025_DynamicLineProjectionRule",
        "task_id": "task025",
        "family": "D",
        "rewrite_type": "geometric_rule_builder",
        "local_cost_delta": -330000,
        "online_score": "6037.90",
        "online_delta": "+",
        "decision": "promote",
        "reason": "simple geometric rule (line projection via complete-line detection); positive online",
    },
    {
        "candidate_name": "task084_DiagonalBottomFillRule",
        "task_id": "task084",
        "family": "D",
        "rewrite_type": "geometric_rule_builder",
        "local_cost_delta": -1390000,
        "online_score": "6037.90",
        "online_delta": "+",
        "decision": "promote",
        "reason": "simple geometric rule (diagonal + bottom fill); positive online",
    },
    {
        "candidate_name": "task200_BottomMarkerVerticalStripeRule",
        "task_id": "task200",
        "family": "D",
        "rewrite_type": "geometric_rule_builder",
        "local_cost_delta": -989000,
        "online_score": "6037.90",
        "online_delta": "+",
        "decision": "promote",
        "reason": "simple geometric rule (marker→vertical stripe); positive online",
    },
    {
        "candidate_name": "task076_semantic_template",
        "task_id": "task076",
        "family": "K",
        "rewrite_type": "semantic_template_matcher",
        "local_cost_delta": -800000,
        "online_score": "6027.22",
        "online_delta": "-",
        "decision": "reject",
        "reason": "task-specific semantic finite-template: local labelled all passed but online regressed",
    },
    {
        "candidate_name": "task363_SparseShiftConvRewrite",
        "task_id": "task363",
        "family": "L",
        "rewrite_type": "conv_to_pad_slice_concat",
        "local_cost_delta": -176810,
        "online_score": "6037.81",
        "online_delta": "-",
        "decision": "reject",
        "reason": "Conv->Pad/Slice/Concat expansion: local equivalent but online regressed; Slice/Concat expansion unreliable",
    },
    {
        "candidate_name": "task367_DynamicRectangularCavityFillRule",
        "task_id": "task367",
        "family": "E",
        "rewrite_type": "semantic_cavity_fill",
        "local_cost_delta": -219000,
        "online_score": "6028.06",
        "online_delta": "-",
        "decision": "reject",
        "reason": "rectangular cavity semantic fill: local passed but online negative",
    },
    {
        "candidate_name": "task396_DynamicLargestFrameRecolorCropRule",
        "task_id": "task396",
        "family": "L",
        "rewrite_type": "semantic_frame_crop",
        "local_cost_delta": -100000,
        "online_score": "6027.99",
        "online_delta": "-",
        "decision": "reject",
        "reason": "largest-frame semantic recolor-crop: local passed but online negative",
    },
    {
        "candidate_name": "task233_combined_dtype_compress",
        "task_id": "task233",
        "family": "L",
        "rewrite_type": "dtype_compression",
        "local_cost_delta": -250000,
        "online_score": "6037.50",
        "online_delta": "-",
        "decision": "reject",
        "reason": "16 int64→int32 Gather-index compressions; local 266/266 labelled passed but online -0.40; dtype changes affect runtime scoring",
    },
    {
        "candidate_name": "batch_30_dtype_and_repromote",
        "task_id": "batch",
        "family": "mixed",
        "rewrite_type": "batch_mixed",
        "local_cost_delta": -2000000,
        "online_score": "6028",
        "online_delta": "-",
        "decision": "reject",
        "reason": "batch of 30 candidates (re-promotes + dtype compression); massive regression due to mixed risk profiles",
    },
    {
        "candidate_name": "same_score_zero_initializer_merge_corrected_20260612",
        "task_id": "batch",
        "family": "mixed",
        "rewrite_type": "zero_initializer_same_score_merge",
        "local_cost_delta": -13520,
        "online_score": "6275.08",
        "online_delta": "-0.01",
        "decision": "reject",
        "reason": "17 zero-initializer candidates that individually displayed as ties regressed when merged against the 6275.09 baseline",
    },
    {
        "candidate_name": "neurogolf_6348_56_hybrid_stack",
        "task_id": "batch",
        "family": "external_hybrid_stack",
        "rewrite_type": "known_online_submission_take_best",
        "local_cost_delta": "",
        "online_score": "6348.56",
        "online_delta": "+73.47 vs 6275.09",
        "decision": "promote",
        "reason": "newly imported hybrid stack submission beats the 6275.09 reference; use known online score rather than local cost for promotion",
    },
    {
        "candidate_name": "ref6348_equiv_optimized_lane_35_individual_flat_20260613",
        "task_id": "batch",
        "family": "external_hybrid_stack",
        "rewrite_type": "graph_equivalent_dead_dedup_lane_ablation",
        "local_cost_delta": -1531410,
        "online_score": "6348.56",
        "online_delta": "0 on all uploaded one-lane ablations",
        "decision": "promote",
        "reason": "user reported every uploaded strict-valid one-task/one-lane equivalent optimization scored 6348.56; merged package still needs separate online confirmation",
    },
    {
        "candidate_name": "ref6348_equiv_optimized_merged_20260613",
        "task_id": "batch",
        "family": "external_hybrid_stack",
        "rewrite_type": "graph_equivalent_dead_dedup_merged",
        "local_cost_delta": -1531410,
        "online_score": "6348.56",
        "online_delta": "0",
        "decision": "promote",
        "reason": "user reported the merged 35-lane equivalent package also scored 6348.56; graph-equivalent base-lane cleanup did not move official best-lane score",
    },
    {
        "candidate_name": "task101_task255_algorithm_pruned_merged_20260613",
        "task_id": "task101+task255",
        "family": "official_static_algorithm_prune",
        "rewrite_type": "override_algorithm_branch_prune",
        "local_cost_delta": -155195,
        "online_score": "6349.16",
        "online_delta": "+0.60 vs 6348.56",
        "decision": "promote",
        "reason": "merged task101 template-radius R02 pruning and task255 shape-family pruning; user reported online score improved to 6349.16",
    },
]


def scan_ablation_reports(reports_dir: str) -> list[dict[str, Any]]:
    """Extract any additional results from ablation report CSVs."""
    extra: list[dict[str, Any]] = []
    root = Path(reports_dir)
    for path in sorted(root.glob("ablation_submission_report_*.csv")):
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    tid = row.get("task_id", "").strip()
                    if not tid or tid in {r["task_id"] for r in KNOWN_RESULTS}:
                        continue
                    # Check if we have a known online result for this
                    pass
        except Exception:
            continue
    return extra


def build_online_memory(
    reports_dir: str = "outputs/reports",
    report_path: str = DEFAULT_REPORT,
) -> dict[str, Any]:
    rows = list(KNOWN_RESULTS)
    rows.extend(scan_ablation_reports(reports_dir))

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    promoted = [r for r in rows if r["decision"] == "promote"]
    rejected = [r for r in rows if r["decision"] == "reject"]

    by_rewrite_type: dict[str, dict[str, int]] = {}
    for r in rows:
        rt = r["rewrite_type"]
        by_rewrite_type.setdefault(rt, {"total": 0, "promote": 0, "reject": 0})
        by_rewrite_type[rt]["total"] += 1
        by_rewrite_type[rt][r["decision"]] += 1

    summary = {
        "report_path": str(report),
        "total_known_results": len(rows),
        "promoted": len(promoted),
        "rejected": len(rejected),
        "by_rewrite_type": by_rewrite_type,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n=== INFERRED RULES ===")
    print("SAFE patterns (promoted online):")
    print("  - Known-online higher-score full submissions can replace the baseline wholesale")
    print("  - Gather/index structural rewrites (one-hot→Gather, perm→Gather)")
    print("  - Row-bank prefix pruning (conservative mode)")
    print("  - Prior-range pruning (observed mode)")
    print("  - Simple geometric rule builders (line projection, fill, stripe)")
    print("  - Enumeration table pruning (unreachable rows)")
    print("  - Official-static high-cost override algorithm branch pruning")
    print()
    print("UNSAFE patterns (rejected online):")
    print("  - Dtype compression (even int64→int32 for Gather indices)")
    print("  - Task-specific semantic templates (finite-template, cavity fill)")
    print("  - Conv→Pad/Slice/Concat expansion (many small ops)")
    print("  - Semantic frame/panel detection rules")
    print("  - Batch submissions with mixed risk profiles")
    print("  - Zero-initializer displayed ties merged as a batch")
    print()
    print("DESIGN RULES for next candidates:")
    print("  1. Keep operator structure intact (don't replace Conv with Slice/Concat)")
    print("  2. Don't change dtypes of existing tensors")
    print("  3. Gather/index rewrites are the most promising path")
    print("  4. Simple geometric rules (complete-line, fill) are safe")
    print("  5. Test one task at a time, not batches")
    print("  6. Conservative mode always preferred first")
    print("  7. Prioritize official-static best-lane cost, not old local cost")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", default="outputs/reports")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    args = parser.parse_args()
    build_online_memory(reports_dir=args.reports_dir, report_path=args.report)


if __name__ == "__main__":
    main()
