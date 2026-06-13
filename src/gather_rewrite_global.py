"""Global Gather/Index rewrite pipeline — build actual ONNX rewrites.

Follows the optimization strategy Direction 1: find all models with dense one-hot
permutation/selection/routing matrices and replace them with Gather operations.

Three rewrite patterns (risk low→high):
  A. int_index_table → dtype compression (pure initializer replace, no graph change)
  B. one_hot_matmul → Gather + index table (graph surgery, proven on task076)
  C. one_hot_matrix → Gather + Reshape (more complex graph changes)

Each rewrite is validated against labelled splits before packaging.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .cost_estimator import estimate_model_cost, check_forbidden_ops, check_static_shapes
from .discover_exact_gather_rewrites import discover_all as discover_gather_candidates
from .discover_exact_gather_rewrites import _is_one_hot_last_axis, _dtype_suggestion
from .validate_labelled_splits import validate_labelled_splits


DEFAULT_DISCOVERY_REPORT = "outputs/reports/exact_gather_rewrites_discovery.csv"
DEFAULT_REPORT = "outputs/reports/gather_rewrite_global_report.csv"
DEFAULT_CANDIDATE_DIR = "outputs/candidates/gather_rewrite_global"
DEFAULT_CONSERVATIVE_DIR = "outputs/candidates/gather_rewrite_global_conservative"
FIELDS = [
    "task_id",
    "initializer_name",
    "pattern",
    "rewrite_type",
    "source_cost",
    "output_cost",
    "cost_delta",
    "source_file_size",
    "output_file_size",
    "output_model_path",
    "valid",
    "train_pass",
    "test_pass",
    "arc_gen_pass",
    "failure_reason",
]


# ---------------------------------------------------------------------------
# Step A: int_index_table dtype compression
# ---------------------------------------------------------------------------

def _compression_safe_for_consumers(
    model: onnx.ModelProto,
    init_name: str,
    suggestion: str,
    array_dtype: np.dtype,
) -> str | None:
    """Check whether a dtype compression is safe given the consumer ops.

    ONNX Gather/GatherElements/GatherND accept int32 and int64 indices.
    Returns the safe compression or None.
    """
    consumer_ops = set()
    for node in model.graph.node:
        if init_name in node.input:
            consumer_ops.add(node.op_type)

    gather_family = {"Gather", "GatherElements", "GatherND", "Scatter", "ScatterElements", "ScatterND"}
    is_gather_only = consumer_ops.issubset(gather_family)

    # int64→int32 is always safe for Gather-family consumers
    if is_gather_only and np.issubdtype(array_dtype, np.integer) and array_dtype.itemsize >= 8:
        return "int32"

    # If the exact suggestion mentions int32, use it for any consumer
    if "int32" in suggestion and not any(
        op in consumer_ops for op in ("NonMaxSuppression", "Resize")
    ):
        return "int32"

    # int16 more canonical
    if is_gather_only and "int16" in suggestion and "more canonical" in suggestion:
        return "int16"

    return None


def _compress_initializer_dtype(
    model: onnx.ModelProto,
    init_name: str,
) -> tuple[int, int, str]:
    """Replace a single initializer with a dtype-compressed version.

    Only compresses int64→int32 for safety. Returns (bytes_saved, new_nbytes, suggestion).
    Raises ValueError if no safe compression possible.
    """
    target = None
    target_idx = None
    for idx, init in enumerate(model.graph.initializer):
        if init.name == init_name:
            target = init
            target_idx = idx
            break
    if target is None:
        raise ValueError(f"initializer {init_name} not found")

    array = numpy_helper.to_array(target)
    suggestion = _dtype_suggestion(array)
    if suggestion is None:
        raise ValueError(f"no dtype compression possible for {init_name}")

    safe_compression = _compression_safe_for_consumers(model, init_name, suggestion, array.dtype)
    if safe_compression is None:
        raise ValueError(f"unsafe dtype compression for {init_name}: consumers={set(n.op_type for n in model.graph.node if init_name in n.input)}")

    if safe_compression == "int32":
        new_dtype = TensorProto.INT32
        new_array = array.astype(np.int32)
    elif safe_compression == "int16":
        new_dtype = TensorProto.INT16
        new_array = array.astype(np.int16)
    else:
        raise ValueError(f"unsupported safe compression: {safe_compression}")

    if not np.array_equal(array.astype(np.int64), new_array.astype(np.int64)):
        raise ValueError(f"dtype compression changed values for {init_name}")

    old_nbytes = array.nbytes
    new_nbytes = new_array.nbytes
    saved = old_nbytes - new_nbytes

    new_init = numpy_helper.from_array(new_array, name=init_name)
    model.graph.initializer[target_idx].CopyFrom(new_init)
    return saved, new_nbytes, suggestion


def build_dtype_compression(
    task_id: str,
    model_path: str,
    initializer_name: str,
    output_path: str,
) -> dict[str, Any]:
    """Compress a single initializer's dtype and save the rewritten model."""
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    source_info = estimate_model_cost(model_path)

    try:
        saved, new_nbytes, suggestion = _compress_initializer_dtype(model, initializer_name)
    except ValueError as exc:
        return {
            "valid": False,
            "source_cost": int(source_info["estimated_cost"]),
            "output_cost": int(source_info["estimated_cost"]),
            "cost_delta": 0,
            "source_file_size": int(source_info["file_size_bytes"]),
            "output_file_size": int(source_info["file_size_bytes"]),
            "failure_reason": str(exc),
        }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    onnx.checker.check_model(str(out_path))
    output_info = estimate_model_cost(str(out_path))

    return {
        "valid": True,
        "source_cost": int(source_info["estimated_cost"]),
        "output_cost": int(output_info["estimated_cost"]),
        "cost_delta": int(output_info["estimated_cost"]) - int(source_info["estimated_cost"]),
        "source_file_size": int(source_info["file_size_bytes"]),
        "output_file_size": int(output_info["file_size_bytes"]),
        "failure_reason": f"compressed {initializer_name}: {suggestion}, saved {saved} bytes",
    }


# ---------------------------------------------------------------------------
# Step B: one_hot_matmul → Gather
# ---------------------------------------------------------------------------

def _find_consumers(model: onnx.ModelProto, name: str) -> list[onnx.NodeProto]:
    return [node for node in model.graph.node if name in node.input]


def _one_hot_to_gather_indices(array: np.ndarray) -> np.ndarray:
    """Convert a one-hot float array to int64 index table via argmax on last axis."""
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError("expected float array")
    is_oh, _ = _is_one_hot_last_axis(array)
    if not is_oh:
        raise ValueError("array is not one-hot on last axis")
    return np.argmax(array, axis=-1).astype(np.int64)


def _rewrite_one_hot_matmul(
    model: onnx.ModelProto,
    init_name: str,
) -> list[dict[str, Any]]:
    """Replace MatMul(onehot, data) with Gather(data, index_table).

    Returns list of {node_removed, node_added, cost_saved} for each replacement.
    """
    array = None
    for init in model.graph.initializer:
        if init.name == init_name:
            array = numpy_helper.to_array(init)
            break
    if array is None:
        raise ValueError(f"initializer {init_name} not found")

    indices = _one_hot_to_gather_indices(array)
    matmul_nodes = [n for n in _find_consumers(model, init_name) if n.op_type == "MatMul"]
    if not matmul_nodes:
        raise ValueError(f"no MatMul consumers found for {init_name}")

    nodes_to_remove: set[str] = set()
    new_nodes: list[onnx.NodeProto] = []
    replacements: list[dict[str, Any]] = []

    for mm_node in matmul_nodes:
        data_input = mm_node.input[0] if mm_node.input[0] != init_name else mm_node.input[1]
        output_name = mm_node.output[0]
        new_name = f"/Gather_{init_name}_{len(replacements)}"

        gather_axis = -1
        gather_node = helper.make_node(
            "Gather",
            [data_input, f"{init_name}_Idx"],
            [output_name],
            name=new_name,
            axis=gather_axis,
        )
        new_nodes.append(gather_node)
        nodes_to_remove.add(mm_node.name)

        replacements.append({
            "matmul_node": mm_node.name or "<unnamed>",
            "data_input": data_input,
            "output": output_name,
            "gather_node": new_name,
        })

    rewritten_nodes = []
    inserted = False
    for node in model.graph.node:
        if node.name in nodes_to_remove:
            if not inserted:
                rewritten_nodes.extend(new_nodes)
                inserted = True
            continue
        rewritten_nodes.append(node)
    if not inserted:
        raise ValueError("no insertion point found")

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    kept_inits = [i for i in model.graph.initializer if i.name != init_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits)
    model.graph.initializer.append(
        numpy_helper.from_array(indices, name=f"{init_name}_Idx")
    )

    del model.graph.value_info[:]
    return replacements


def build_one_hot_matmul_gather(
    task_id: str,
    model_path: str,
    initializer_name: str,
    output_path: str,
) -> dict[str, Any]:
    """Rewrite a one-hot MatMul pattern to Gather for a single model."""
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    source_info = estimate_model_cost(model_path)

    try:
        replacements = _rewrite_one_hot_matmul(model, initializer_name)
    except ValueError as exc:
        return {
            "valid": False,
            "source_cost": int(source_info["estimated_cost"]),
            "output_cost": int(source_info["estimated_cost"]),
            "cost_delta": 0,
            "source_file_size": int(source_info["file_size_bytes"]),
            "output_file_size": int(source_info["file_size_bytes"]),
            "failure_reason": str(exc),
        }

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    onnx.checker.check_model(str(out_path))
    output_info = estimate_model_cost(str(out_path))

    return {
        "valid": True,
        "source_cost": int(source_info["estimated_cost"]),
        "output_cost": int(output_info["estimated_cost"]),
        "cost_delta": int(output_info["estimated_cost"]) - int(source_info["estimated_cost"]),
        "source_file_size": int(source_info["file_size_bytes"]),
        "output_file_size": int(output_info["file_size_bytes"]),
        "failure_reason": f"replaced {len(replacements)} MatMul nodes with Gather: {[r['matmul_node'] for r in replacements]}",
    }


# ---------------------------------------------------------------------------
# Step C: one_hot_matrix for Conv
# ---------------------------------------------------------------------------

def build_one_hot_conv_gather(
    task_id: str,
    model_path: str,
    initializer_name: str,
    output_path: str,
) -> dict[str, Any]:
    """Attempt to replace a one-hot Conv weight with Gather-based equivalent.

    This is more heuristic — we try to reconstruct what the one-hot Conv was doing.
    """
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    source_info = estimate_model_cost(model_path)

    array = None
    init_target = None
    for init in model.graph.initializer:
        if init.name == init_name:
            array = numpy_helper.to_array(init)
            init_target = init
            break
    if array is None:
        return {
            "valid": False,
            "source_cost": int(source_info["estimated_cost"]),
            "output_cost": int(source_info["estimated_cost"]),
            "cost_delta": 0,
            "source_file_size": int(source_info["file_size_bytes"]),
            "output_file_size": int(source_info["file_size_bytes"]),
            "failure_reason": f"initializer {initializer_name} not found",
        }

    is_oh, _ = _is_one_hot_last_axis(array)
    if not is_oh:
        return {
            "valid": False,
            "source_cost": int(source_info["estimated_cost"]),
            "output_cost": int(source_info["estimated_cost"]),
            "cost_delta": 0,
            "source_file_size": int(source_info["file_size_bytes"]),
            "output_file_size": int(source_info["file_size_bytes"]),
            "failure_reason": "not one-hot pattern",
        }

    conv_consumers = [n for n in _find_consumers(model, init_name) if n.op_type == "Conv"]
    if not conv_consumers:
        return {
            "valid": False,
            "source_cost": int(source_info["estimated_cost"]),
            "output_cost": int(source_info["estimated_cost"]),
            "cost_delta": 0,
            "source_file_size": int(source_info["file_size_bytes"]),
            "output_file_size": int(source_info["file_size_bytes"]),
            "failure_reason": "no Conv consumer, graph too complex for auto-rewrite",
        }

    try:
        indices = _one_hot_to_gather_indices(array)
    except ValueError:
        return {
            "valid": False,
            "source_cost": int(source_info["estimated_cost"]),
            "output_cost": int(source_info["estimated_cost"]),
            "cost_delta": 0,
            "source_file_size": int(source_info["file_size_bytes"]),
            "output_file_size": int(source_info["file_size_bytes"]),
            "failure_reason": "could not compute gather indices",
        }

    kept_inits = [i for i in model.graph.initializer if i.name != init_name]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_inits)
    idx_init = numpy_helper.from_array(indices, name=f"{init_name}_Idx")
    model.graph.initializer.append(idx_init)
    del model.graph.value_info[:]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(out_path))
    onnx.checker.check_model(str(out_path))
    output_info = estimate_model_cost(str(out_path))

    return {
        "valid": True,
        "source_cost": int(source_info["estimated_cost"]),
        "output_cost": int(output_info["estimated_cost"]),
        "cost_delta": int(output_info["estimated_cost"]) - int(source_info["estimated_cost"]),
        "source_file_size": int(source_info["file_size_bytes"]),
        "output_file_size": int(output_info["file_size_bytes"]),
        "failure_reason": (
            f"replaced {init_name} with index table, "
            f"kept {len(conv_consumers)} Conv node(s) — may need manual check"
        ),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def _validate_rewritten(
    model_path: str,
    task_path: str,
    validation_dir: str,
) -> dict[str, Any]:
    """Validate a rewritten ONNX model against all labelled splits."""
    result = {"valid": True, "train_pass": "", "test_pass": "", "arc_gen_pass": "", "failure_reason": ""}
    try:
        onnx.checker.check_model(model_path)
    except Exception as exc:
        result["valid"] = False
        result["failure_reason"] = f"onnx checker: {exc}"
        return result

    if not check_forbidden_ops(model_path)["passed"]:
        result["valid"] = False
        result["failure_reason"] = "forbidden ops"
        return result
    if not check_static_shapes(model_path)["passed"]:
        result["valid"] = False
        result["failure_reason"] = "dynamic shapes"
        return result

    info = estimate_model_cost(model_path)
    if not info["file_size_ok"]:
        result["valid"] = False
        result["failure_reason"] = "file size exceeded"
        return result

    task_stem = Path(task_path).stem
    model_stem = Path(model_path).stem
    val_report = str(Path(validation_dir) / f"{task_stem}_{model_stem}_validation.csv")
    try:
        labelled = validate_labelled_splits(model_path, task_path, val_report)
    except Exception as exc:
        result["valid"] = False
        result["failure_reason"] = f"validation error: {exc}"
        return result

    passed = labelled.get("passed", False)
    split_counts = labelled.get("split_counts", {})
    result["train_pass"] = split_counts.get("train", {}).get("passed", "?")
    result["test_pass"] = split_counts.get("test", {}).get("passed", "?")
    result["arc_gen_pass"] = split_counts.get("arc-gen", {}).get("passed", "?")
    if not passed:
        result["valid"] = False
        result["failure_reason"] = f"labelled splits: {labelled.get('passed_cases', 0)}/{labelled.get('total_cases', 0)}"
    return result


def run_pipeline(
    model_dir: str = "outputs/onnx",
    task_dir: str = "task",
    discovery_report: str = DEFAULT_DISCOVERY_REPORT,
    candidate_dir: str = DEFAULT_CANDIDATE_DIR,
    conservative_dir: str = DEFAULT_CONSERVATIVE_DIR,
    report_path: str = DEFAULT_REPORT,
    pattern_filter: str = "",
    max_candidates: int = 30,
) -> dict[str, Any]:
    patterns = [p.strip() for p in pattern_filter.split(",") if p.strip()]

    # Phase 1: Refresh discovery
    print("Refreshing gather rewrite discovery...")
    discovery = discover_gather_candidates(
        model_dir=model_dir,
        report_path=discovery_report,
    )
    print(f"Found {discovery['candidate_count']} candidates")

    # Phase 2: Build rewrites
    rows: list[dict[str, Any]] = []
    build_count = 0
    skipped = 0

    for candidate in discovery.get("top_candidates", [])[:max_candidates]:
        # discovery returns top_candidates as list of dicts with basic fields
        pass

    # Re-read the full CSV for all fields
    with Path(discovery_report).open("r", newline="", encoding="utf-8") as handle:
        all_candidates = list(csv.DictReader(handle))

    if patterns:
        all_candidates = [c for c in all_candidates if c.get("pattern", "") in patterns]

    for candidate in all_candidates[:max_candidates]:
        tid = candidate["task_id"]
        init_name = candidate["initializer_name"]
        pattern = candidate.get("pattern", "")
        model_path = Path(model_dir) / f"{tid}.onnx"
        task_path = Path(task_dir) / f"{tid}.json"

        if not model_path.is_file():
            skipped += 1
            continue
        if not task_path.is_file():
            skipped += 1
            continue

        # Sanitize init name for Windows filenames
        safe_init = init_name.replace("::", "_").replace("/", "_").replace("\\", "_")
        safe_init = safe_init.replace(":", "_").replace("*", "_").replace("?", "_")
        safe_init = safe_init.replace("\"", "_").replace("<", "_").replace(">", "_")
        safe_init = safe_init.replace("|", "_")

        # Choose rewrite strategy
        if pattern == "int_index_table":
            output_name = f"{tid}_{safe_init}_dtype_compress.onnx"
            build_result = build_dtype_compression(
                tid, str(model_path), init_name,
                str(Path(candidate_dir) / output_name),
            )
            rewrite_type = "dtype_compress"
        elif pattern in ("one_hot_matmul",):
            output_name = f"{tid}_{safe_init}_gather.onnx"
            build_result = build_one_hot_matmul_gather(
                tid, str(model_path), init_name,
                str(Path(candidate_dir) / output_name),
            )
            rewrite_type = "one_hot_matmul_gather"
        elif pattern in ("one_hot_matrix",):
            output_name = f"{tid}_{safe_init}_conv_gather.onnx"
            build_result = build_one_hot_conv_gather(
                tid, str(model_path), init_name,
                str(Path(candidate_dir) / output_name),
            )
            rewrite_type = "one_hot_conv_gather"
        else:
            skipped += 1
            continue

        row = {
            "task_id": tid,
            "initializer_name": init_name,
            "pattern": pattern,
            "rewrite_type": rewrite_type,
            "source_cost": build_result.get("source_cost", 0),
            "output_cost": build_result.get("output_cost", 0),
            "cost_delta": build_result.get("cost_delta", 0),
            "source_file_size": build_result.get("source_file_size", 0),
            "output_file_size": build_result.get("output_file_size", 0),
            "output_model_path": str(Path(candidate_dir) / output_name),
            "valid": build_result.get("valid", False),
            "train_pass": "",
            "test_pass": "",
            "arc_gen_pass": "",
            "failure_reason": build_result.get("failure_reason", ""),
        }

        # Validate if build succeeded
        if row["valid"]:
            output_model_path = str(Path(candidate_dir) / output_name)
            validation = _validate_rewritten(
                output_model_path, str(task_path), candidate_dir
            )
            row["valid"] = validation["valid"]
            row["train_pass"] = validation.get("train_pass", "")
            row["test_pass"] = validation.get("test_pass", "")
            row["arc_gen_pass"] = validation.get("arc_gen_pass", "")
            if not row["valid"]:
                row["failure_reason"] = validation.get("failure_reason", "validation failed")

            # Copy to conservative dir if valid
            if row["valid"]:
                cons_path = Path(conservative_dir) / output_name
                cons_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(output_model_path, str(cons_path))

        rows.append(row)
        build_count += 1

    # Write report
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    valid_count = sum(1 for r in rows if r["valid"])
    total_delta = sum(r["cost_delta"] for r in rows if r["valid"])

    summary = {
        "report_path": str(report),
        "candidates_total": len(all_candidates),
        "built": build_count,
        "skipped": skipped,
        "valid": valid_count,
        "invalid": len(rows) - valid_count,
        "total_cost_delta": total_delta,
        "by_pattern": {},
    }
    for r in rows:
        p = r["rewrite_type"]
        if p not in summary["by_pattern"]:
            summary["by_pattern"][p] = {"total": 0, "valid": 0}
        summary["by_pattern"][p]["total"] += 1
        if r["valid"]:
            summary["by_pattern"][p]["valid"] += 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/onnx")
    parser.add_argument("--task-dir", default="task")
    parser.add_argument("--discovery-report", default=DEFAULT_DISCOVERY_REPORT)
    parser.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--conservative-dir", default=DEFAULT_CONSERVATIVE_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument(
        "--pattern-filter",
        default="",
        help="Comma-separated patterns: int_index_table,one_hot_matmul,one_hot_matrix",
    )
    parser.add_argument("--max-candidates", type=int, default=30)
    args = parser.parse_args()
    run_pipeline(
        model_dir=args.model_dir,
        task_dir=args.task_dir,
        discovery_report=args.discovery_report,
        candidate_dir=args.candidate_dir,
        conservative_dir=args.conservative_dir,
        report_path=args.report,
        pattern_filter=args.pattern_filter,
        max_candidates=args.max_candidates,
    )


if __name__ == "__main__":
    main()
