"""Discover ONNX enumeration tables that may be safe to prune later.

This is a read-only scouting pass.  It looks for groups of initializers or
Constant-node tensors that share a large first dimension, which is the pattern
used by task233's combo table.  The output is a CSV report for manual follow-up;
it does not edit models or generate candidates.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from .build_ablation_submissions import build_ablation_submissions
from .evaluate_onnx_candidate import evaluate
from .row_bank_prefix_prune import generate_row_bank_prefix_candidates
from .task157_placement_prune import generate_task157_placement_candidates
from .task255_interval_prune import prune_task255_intervals
from .validate_labelled_splits import validate_labelled_splits


DEFAULT_TASK_IDS = (
    "task076,task157,task367,task363,task209,task396,task028,task255,"
    "task382,task107,task313,task290,task105,task027,task009,task058,"
    "task319"
)

FIELDS = [
    "task_id",
    "model_path",
    "row_count",
    "shared_value_count",
    "initializer_count",
    "constant_count",
    "total_elements",
    "total_bytes",
    "rank_set",
    "dtype_set",
    "max_tail_elements",
    "has_arange_vector",
    "has_arange_column",
    "has_zero_table",
    "has_small_integer_table",
    "example_values",
    "value_names",
    "priority_score",
]

CANDIDATE_FIELDS = [
    "task_id",
    "rule_name",
    "mode",
    "source_model",
    "candidate_model_path",
    "generated",
    "evaluate_valid",
    "labelled_passed",
    "package_eligible",
    "failure_reason",
    "estimated_cost",
    "file_size_bytes",
    "labelled_report_path",
]

DEFAULT_CANDIDATE_DIR = "outputs/candidates/enumeration_table_prune"
DEFAULT_CONSERVATIVE_DIR = "outputs/candidates/enumeration_table_prune_conservative"
DEFAULT_CANDIDATE_REPORT = "outputs/reports/enumeration_table_prune_candidates.csv"
DEFAULT_ABLATION_DIR = "outputs/ablation_submissions/enumeration_table_prune"
DEFAULT_ABLATION_REPORT = "outputs/reports/ablation_submission_report_enumeration_table_prune.csv"
DEFAULT_TASK157_SOURCE = (
    "outputs/candidates/online_safe_reverts/head_extract/outputs/onnx/task157.onnx"
)


def _task_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _shape_tail_size(shape: tuple[int, ...]) -> int:
    if len(shape) <= 1:
        return 1
    return int(np.prod(shape[1:], dtype=np.int64))


def _is_arange_vector(array: np.ndarray) -> bool:
    if array.ndim != 1 or array.size == 0:
        return False
    if not np.issubdtype(array.dtype, np.integer):
        return False
    return bool(np.array_equal(array, np.arange(array.shape[0], dtype=array.dtype)))


def _is_arange_column(array: np.ndarray) -> bool:
    if array.ndim != 2 or array.shape[1] != 1 or array.shape[0] == 0:
        return False
    if not np.issubdtype(array.dtype, np.integer):
        return False
    return bool(np.array_equal(array.reshape(-1), np.arange(array.shape[0], dtype=array.dtype)))


def _is_small_integer_table(array: np.ndarray) -> bool:
    if array.ndim < 2 or not np.issubdtype(array.dtype, np.integer):
        return False
    if array.size == 0:
        return False
    values = np.unique(array)
    if values.size > 16:
        return False
    return int(values.min()) >= 0 and int(values.max()) <= 15


def _dtype_name(array: np.ndarray) -> str:
    return str(array.dtype)


def _constant_arrays(model: onnx.ModelProto) -> list[tuple[str, np.ndarray]]:
    values: list[tuple[str, np.ndarray]] = []
    for index, node in enumerate(model.graph.node):
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            tensor = onnx.helper.get_attribute_value(attr)
            array = numpy_helper.to_array(tensor)
            name = node.output[0] if node.output else node.name or f"Constant#{index}"
            values.append((name, array))
    return values


def _model_values(model: onnx.ModelProto) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        values.append({"kind": "initializer", "name": initializer.name, "array": array})
    for name, array in _constant_arrays(model):
        values.append({"kind": "constant", "name": name, "array": array})
    return values


def _value_summary(name: str, kind: str, array: np.ndarray) -> str:
    shape = "x".join(str(dim) for dim in array.shape) if array.shape else "scalar"
    return f"{kind}:{name}:{shape}:{array.dtype}"


def discover_model_tables(
    task_id: str,
    model_path: Path,
    min_first_dim: int,
    min_shared_values: int,
) -> list[dict[str, Any]]:
    """Return shared-first-dimension table groups for one ONNX model."""
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for value in _model_values(model):
        array = value["array"]
        if array.ndim == 0 or array.shape[0] < min_first_dim:
            continue
        groups[int(array.shape[0])].append(value)

    rows: list[dict[str, Any]] = []
    for row_count, values in sorted(groups.items()):
        if len(values) < min_shared_values:
            continue
        arrays = [value["array"] for value in values]
        total_elements = int(sum(array.size for array in arrays))
        total_bytes = int(sum(array.nbytes for array in arrays))
        names = [
            _value_summary(value["name"], value["kind"], value["array"])
            for value in values
        ]
        example_names = names[:12]
        priority_score = total_elements * max(1, len(values) - 1)
        rows.append(
            {
                "task_id": task_id,
                "model_path": str(model_path),
                "row_count": row_count,
                "shared_value_count": len(values),
                "initializer_count": sum(1 for value in values if value["kind"] == "initializer"),
                "constant_count": sum(1 for value in values if value["kind"] == "constant"),
                "total_elements": total_elements,
                "total_bytes": total_bytes,
                "rank_set": "|".join(str(rank) for rank in sorted({array.ndim for array in arrays})),
                "dtype_set": "|".join(sorted({_dtype_name(array) for array in arrays})),
                "max_tail_elements": max(_shape_tail_size(tuple(array.shape)) for array in arrays),
                "has_arange_vector": any(_is_arange_vector(array) for array in arrays),
                "has_arange_column": any(_is_arange_column(array) for array in arrays),
                "has_zero_table": any(array.ndim >= 2 and np.count_nonzero(array) == 0 for array in arrays),
                "has_small_integer_table": any(_is_small_integer_table(array) for array in arrays),
                "example_values": "; ".join(example_names),
                "value_names": "; ".join(names),
                "priority_score": priority_score,
            }
        )
    return rows


def discover_enumeration_tables(
    model_dir: str = "outputs/onnx",
    task_ids: list[str] | None = None,
    report_path: str = "outputs/reports/enumeration_table_prune_discovery.csv",
    min_first_dim: int = 32,
    min_shared_values: int = 2,
) -> dict[str, Any]:
    """Scan task models for shared-row enumeration-table candidates."""
    task_ids = task_ids or _task_ids(DEFAULT_TASK_IDS)
    root = Path(model_dir)
    rows: list[dict[str, Any]] = []
    missing_tasks: list[str] = []
    for task_id in task_ids:
        model_path = root / f"{task_id}.onnx"
        if not model_path.is_file():
            missing_tasks.append(task_id)
            continue
        rows.extend(
            discover_model_tables(
                task_id=task_id,
                model_path=model_path,
                min_first_dim=min_first_dim,
                min_shared_values=min_shared_values,
            )
        )

    rows.sort(
        key=lambda row: (
            -int(row["priority_score"]),
            -int(row["total_elements"]),
            row["task_id"],
            int(row["row_count"]),
        )
    )

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_dir": str(root),
        "report_path": str(report),
        "task_count": len(task_ids),
        "missing_tasks": missing_tasks,
        "candidate_group_count": len(rows),
        "top_groups": rows[:5],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _task_path(task_id: str) -> str:
    return str(Path("task") / f"{task_id}.json")


def _mode_from_candidate_name(path: Path) -> str:
    stem = path.stem
    if "Conservative" in stem:
        return "conservative"
    if "Medium" in stem:
        return "medium"
    if "Observed" in stem:
        return "observed"
    return "unknown"


def _rule_name_from_candidate(path: Path) -> str:
    parts = path.stem.split("_", 1)
    return parts[1] if len(parts) == 2 else path.stem


def _candidate_task_id(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def _write_candidate_report(rows: list[dict[str, Any]], report_path: str) -> None:
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _validate_generated_candidate(
    candidate: Path,
    task_id: str,
    report_root: Path,
    source_model: str,
) -> dict[str, Any]:
    task_path = _task_path(task_id)
    labelled_report = report_root / f"{candidate.stem}_labelled_validation.csv"
    evaluation = evaluate(str(candidate), task_path)
    labelled_passed = False
    labelled_failure = ""
    try:
        labelled = validate_labelled_splits(
            str(candidate),
            task_path,
            str(labelled_report),
        )
        labelled_passed = bool(labelled["passed"])
        if not labelled_passed:
            labelled_failure = "labelled_splits_failed"
    except Exception as exc:
        labelled_failure = f"labelled_validation_exception: {exc}"

    failure_reasons = []
    if not evaluation.get("valid", False):
        failure_reasons.append(str(evaluation.get("failure_reason", "evaluation_failed")))
    if labelled_failure:
        failure_reasons.append(labelled_failure)

    mode = _mode_from_candidate_name(candidate)
    return {
        "task_id": task_id,
        "rule_name": _rule_name_from_candidate(candidate),
        "mode": mode,
        "source_model": source_model,
        "candidate_model_path": str(candidate),
        "generated": True,
        "evaluate_valid": bool(evaluation.get("valid", False)),
        "labelled_passed": labelled_passed,
        "package_eligible": (
            mode == "conservative"
            and bool(evaluation.get("valid", False))
            and labelled_passed
        ),
        "failure_reason": "; ".join(reason for reason in failure_reasons if reason),
        "estimated_cost": evaluation.get("estimated_cost", ""),
        "file_size_bytes": evaluation.get("file_size_bytes", ""),
        "labelled_report_path": str(labelled_report),
    }


def _build_task157_candidates(
    candidate_root: Path,
    report_root: Path,
    source_model: str,
) -> list[dict[str, Any]]:
    source_path = Path(source_model)
    if not source_path.is_file():
        source_path = Path("outputs/onnx/task157.onnx")
    if not source_path.is_file():
        raise FileNotFoundError("task157 source model does not exist")
    summary = generate_task157_placement_candidates(
        source_model=str(source_path),
        task_path=_task_path("task157"),
        candidate_dir=str(candidate_root),
        observed_report=str(report_root / "task157_selected_placement_observed.csv"),
        observed_summary=str(report_root / "task157_placement_prune_summary.json"),
        report_dir=str(report_root),
    )
    return [
        _validate_generated_candidate(
            Path(candidate["output_model"]),
            "task157",
            report_root,
            str(source_path),
        )
        for candidate in summary["candidates"]
    ]


def _build_task255_candidates(
    candidate_root: Path,
    report_root: Path,
) -> list[dict[str, Any]]:
    source_model = "outputs/onnx/task255.onnx"
    observed_report = "outputs/reports/task255_selected_interval_observed.csv"
    if not Path(source_model).is_file():
        raise FileNotFoundError(f"task255 source model does not exist: {source_model}")
    if not Path(observed_report).is_file():
        raise FileNotFoundError(
            f"task255 observed interval report does not exist: {observed_report}"
        )
    mode_names = {
        "conservative": "IntervalPruneConservative",
        "medium": "IntervalPruneMedium",
        "observed": "IntervalPruneObserved",
    }
    rows: list[dict[str, Any]] = []
    for mode, name in mode_names.items():
        output_model = candidate_root / f"task255_{name}.onnx"
        prune_task255_intervals(
            source_model=source_model,
            output_model=str(output_model),
            report_path=str(report_root / f"task255_interval_{mode}.csv"),
            observed_report=observed_report,
            mode=mode,
        )
        rows.append(
            _validate_generated_candidate(
                output_model,
                "task255",
                report_root,
                source_model,
            )
        )
    return rows


def _build_row_bank_prefix_candidates(
    task_id: str,
    candidate_root: Path,
    report_root: Path,
) -> list[dict[str, Any]]:
    source_model = str(Path("outputs/onnx") / f"{task_id}.onnx")
    if not Path(source_model).is_file():
        raise FileNotFoundError(f"{task_id} source model does not exist: {source_model}")
    summary = generate_row_bank_prefix_candidates(
        task_id=task_id,
        source_model=source_model,
        task_path=_task_path(task_id),
        candidate_dir=str(candidate_root),
        report_dir=str(report_root),
    )
    return [
        _validate_generated_candidate(
            Path(candidate["output_model"]),
            task_id,
            report_root,
            source_model,
        )
        for candidate in summary["candidates"]
    ]


def _copy_package_eligible_conservative(
    rows: list[dict[str, Any]],
    conservative_dir: str,
) -> list[str]:
    output_dir = Path(conservative_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for row in rows:
        if not row["package_eligible"]:
            continue
        source = Path(str(row["candidate_model_path"]))
        target = output_dir / source.name
        shutil.copyfile(source, target)
        copied.append(str(target))
    return copied


def generate_supported_prune_candidates(
    task_ids: list[str],
    candidate_dir: str = DEFAULT_CANDIDATE_DIR,
    candidate_report: str = DEFAULT_CANDIDATE_REPORT,
    conservative_dir: str = DEFAULT_CONSERVATIVE_DIR,
    task157_source: str = DEFAULT_TASK157_SOURCE,
    build_conservative_zips: bool = True,
    base_zip: str = "outputs/submission.zip",
    ablation_dir: str = DEFAULT_ABLATION_DIR,
    ablation_report: str = DEFAULT_ABLATION_REPORT,
) -> dict[str, Any]:
    """Build and validate supported row-prune candidates from observed rows."""
    candidate_root = Path(candidate_dir)
    report_root = Path(candidate_report).parent
    candidate_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    builders = {
        "task157": lambda: _build_task157_candidates(
            candidate_root,
            report_root,
            task157_source,
        ),
        "task255": lambda: _build_task255_candidates(candidate_root, report_root),
        "task290": lambda: _build_row_bank_prefix_candidates(
            "task290",
            candidate_root,
            report_root,
        ),
        "task396": lambda: _build_row_bank_prefix_candidates(
            "task396",
            candidate_root,
            report_root,
        ),
    }
    for task_id in task_ids:
        if task_id not in builders:
            continue
        try:
            rows.extend(builders[task_id]())
        except Exception as exc:
            rows.append(
                {
                    "task_id": task_id,
                    "rule_name": "supported_prune_builder",
                    "mode": "",
                    "source_model": "",
                    "candidate_model_path": "",
                    "generated": False,
                    "evaluate_valid": False,
                    "labelled_passed": False,
                    "package_eligible": False,
                    "failure_reason": str(exc),
                    "estimated_cost": "",
                    "file_size_bytes": "",
                    "labelled_report_path": "",
                }
            )

    _write_candidate_report(rows, candidate_report)
    copied = _copy_package_eligible_conservative(rows, conservative_dir)

    ablation_summary: dict[str, Any] | None = None
    if build_conservative_zips and copied:
        ablation_summary = build_ablation_submissions(
            base_zip=base_zip,
            candidate_dir=conservative_dir,
            output_dir=ablation_dir,
            report_path=ablation_report,
            task_ids={_candidate_task_id(Path(path)) for path in copied},
            upload_friendly_folders=True,
        )

    summary = {
        "candidate_dir": candidate_dir,
        "candidate_report": candidate_report,
        "candidate_count": sum(1 for row in rows if row["generated"]),
        "valid_candidate_count": sum(
            1
            for row in rows
            if row["evaluate_valid"] and row["labelled_passed"]
        ),
        "package_eligible_count": sum(1 for row in rows if row["package_eligible"]),
        "conservative_dir": conservative_dir,
        "copied_conservative_candidates": copied,
        "ablation_summary": ablation_summary,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def run_enumeration_table_prune_round(
    model_dir: str = "outputs/onnx",
    task_ids: list[str] | None = None,
    discovery_report: str = "outputs/reports/enumeration_table_prune_discovery.csv",
    min_first_dim: int = 32,
    min_shared_values: int = 2,
    generate_candidates: bool = False,
    candidate_dir: str = DEFAULT_CANDIDATE_DIR,
    candidate_report: str = DEFAULT_CANDIDATE_REPORT,
    conservative_dir: str = DEFAULT_CONSERVATIVE_DIR,
    task157_source: str = DEFAULT_TASK157_SOURCE,
    build_conservative_zips: bool = True,
    base_zip: str = "outputs/submission.zip",
    ablation_dir: str = DEFAULT_ABLATION_DIR,
    ablation_report: str = DEFAULT_ABLATION_REPORT,
) -> dict[str, Any]:
    """Run discovery and optionally generate validated supported candidates."""
    task_ids = task_ids or _task_ids(DEFAULT_TASK_IDS)
    discovery = discover_enumeration_tables(
        model_dir=model_dir,
        task_ids=task_ids,
        report_path=discovery_report,
        min_first_dim=min_first_dim,
        min_shared_values=min_shared_values,
    )
    candidates = None
    if generate_candidates:
        candidates = generate_supported_prune_candidates(
            task_ids=task_ids,
            candidate_dir=candidate_dir,
            candidate_report=candidate_report,
            conservative_dir=conservative_dir,
            task157_source=task157_source,
            build_conservative_zips=build_conservative_zips,
            base_zip=base_zip,
            ablation_dir=ablation_dir,
            ablation_report=ablation_report,
        )
    summary = {
        "discovery": discovery,
        "candidates": candidates,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/onnx")
    parser.add_argument("--task-ids", default=DEFAULT_TASK_IDS)
    parser.add_argument("--report", default="outputs/reports/enumeration_table_prune_discovery.csv")
    parser.add_argument("--min-first-dim", type=int, default=32)
    parser.add_argument("--min-shared-values", type=int, default=2)
    parser.add_argument("--generate-candidates", action="store_true")
    parser.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--candidate-report", default=DEFAULT_CANDIDATE_REPORT)
    parser.add_argument("--conservative-dir", default=DEFAULT_CONSERVATIVE_DIR)
    parser.add_argument("--task157-source", default=DEFAULT_TASK157_SOURCE)
    parser.add_argument("--base-zip", default="outputs/submission.zip")
    parser.add_argument("--ablation-dir", default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--ablation-report", default=DEFAULT_ABLATION_REPORT)
    parser.add_argument("--skip-conservative-zips", action="store_true")
    args = parser.parse_args()
    run_enumeration_table_prune_round(
        model_dir=args.model_dir,
        task_ids=_task_ids(args.task_ids),
        discovery_report=args.report,
        min_first_dim=args.min_first_dim,
        min_shared_values=args.min_shared_values,
        generate_candidates=args.generate_candidates,
        candidate_dir=args.candidate_dir,
        candidate_report=args.candidate_report,
        conservative_dir=args.conservative_dir,
        task157_source=args.task157_source,
        build_conservative_zips=not args.skip_conservative_zips,
        base_zip=args.base_zip,
        ablation_dir=args.ablation_dir,
        ablation_report=args.ablation_report,
    )


if __name__ == "__main__":
    main()
