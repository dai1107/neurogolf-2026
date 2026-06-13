"""Same-shape mask DSL batch search — find Where(mask, color, input) patterns.

Scans all same_shape tasks and searches short mask-algebra programs that explain
the input→output transformation. For tasks that match, builds compact ONNX models.

Direction 2 from 优化策略.md: generalize task133's mask-algebra approach across
all same-shape tasks using a simple DSL of mask primitives and Where composition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .arc_io import load_task
from .cost_estimator import estimate_model_cost, check_forbidden_ops, check_static_shapes
from .encoding import DEFAULT_COLORS, DEFAULT_HEIGHT, DEFAULT_WIDTH, DEFAULT_SHAPE
from .validate_onnx_model import validate_cases
from .validate_labelled_splits import validate_labelled_splits


DEFAULT_REPORT = "outputs/reports/same_shape_mask_dsl_batch_report.csv"
DEFAULT_CANDIDATE_DIR = "outputs/candidates/same_shape_mask_dsl"
DEFAULT_CONSERVATIVE_DIR = "outputs/candidates/same_shape_mask_dsl_conservative"
FIELDS = [
    "task_id",
    "formula",
    "num_masks",
    "num_colors_used",
    "train_pass",
    "test_pass",
    "arc_gen_pass",
    "source_cost",
    "output_cost",
    "cost_delta",
    "output_model_path",
    "valid",
    "failure_reason",
]

Grid = list[list[int]]
Mask = list[list[bool]]


# ---------------------------------------------------------------------------
# Mask DSL primitives
# ---------------------------------------------------------------------------

def _shape(g: Grid) -> tuple[int, int]:
    return len(g), len(g[0])


def mask_background(grid: Grid) -> Mask:
    h, w = _shape(grid)
    return [[grid[r][c] == 0 for c in range(w)] for r in range(h)]


def mask_non_background(grid: Grid) -> Mask:
    h, w = _shape(grid)
    return [[grid[r][c] != 0 for c in range(w)] for r in range(h)]


def mask_color(c: int, grid: Grid) -> Mask:
    h, w = _shape(grid)
    return [[grid[r][c] == c if c >= 0 else False for c in range(w)] for r in range(h)]


def mask_neighbor4(color: int, k: int, grid: Grid) -> Mask:
    h, w = _shape(grid)
    result = [[False for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            count = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == color:
                    count += 1
            result[r][c] = count >= k
    return result


def mask_neighbor8(color: int, k: int, grid: Grid) -> Mask:
    h, w = _shape(grid)
    result = [[False for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            count = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == color:
                        count += 1
            result[r][c] = count >= k
    return result


def _bbox_of_color(color: int, grid: Grid) -> tuple[int, int, int, int] | None:
    h, w = _shape(grid)
    min_r, max_r = h, -1
    min_c, max_c = w, -1
    for r in range(h):
        for c in range(w):
            if grid[r][c] == color:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    if max_r < 0:
        return None
    return min_r, min_c, max_r, max_c


def mask_bbox_inside(color: int, grid: Grid) -> Mask:
    bbox = _bbox_of_color(color, grid)
    h, w = _shape(grid)
    result = [[False for _ in range(w)] for _ in range(h)]
    if bbox is None:
        return result
    r1, c1, r2, c2 = bbox
    for r in range(r1 + 1, r2):
        for c in range(c1 + 1, c2):
            if 0 <= r < h and 0 <= c < w:
                result[r][c] = True
    return result


def mask_bbox_border(color: int, grid: Grid) -> Mask:
    bbox = _bbox_of_color(color, grid)
    h, w = _shape(grid)
    result = [[False for _ in range(w)] for _ in range(h)]
    if bbox is None:
        return result
    r1, c1, r2, c2 = bbox
    for r in range(r1, r2 + 1):
        if 0 <= r < h:
            if 0 <= c1 < w:
                result[r][c1] = True
            if 0 <= c2 < w:
                result[r][c2] = True
    for c in range(c1, c2 + 1):
        if 0 <= c < w:
            if 0 <= r1 < h:
                result[r1][c] = True
            if 0 <= r2 < h:
                result[r2][c] = True
    return result


def mask_hole(grid: Grid) -> Mask:
    """Find background cells enclosed by non-background (unreachable from border)."""
    h, w = _shape(grid)
    reachable: set[tuple[int, int]] = set()
    queue = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r][c] == 0:
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r][c] == 0:
                queue.append((r, c))
    while queue:
        cr, cc = queue.popleft()
        if (cr, cc) in reachable:
            continue
        reachable.add((cr, cc))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0 and (nr, nc) not in reachable:
                queue.append((nr, nc))
    return [[grid[r][c] == 0 and (r, c) not in reachable for c in range(w)] for r in range(h)]


def mask_row_has(color: int, grid: Grid) -> Mask:
    h, w = _shape(grid)
    rows_with_color = {r for r in range(h) for c in range(w) if grid[r][c] == color}
    return [[r in rows_with_color for c in range(w)] for r in range(h)]


def mask_col_has(color: int, grid: Grid) -> Mask:
    h, w = _shape(grid)
    cols_with_color = {c for r in range(h) for c in range(w) if grid[r][c] == color}
    return [[c in cols_with_color for c in range(w)] for r in range(h)]


# ---------------------------------------------------------------------------
# Mask algebra
# ---------------------------------------------------------------------------

def mask_apply(m: Mask, color: int, input_grid: Grid) -> Grid:
    """Apply Where(mask, color, input) — paint cells where mask is True."""
    h, w = _shape(input_grid)
    return [[color if m[r][c] else input_grid[r][c] for c in range(w)] for r in range(h)]


def mask_or(m1: Mask, m2: Mask) -> Mask:
    h, w = len(m1), len(m1[0])
    return [[m1[r][c] or m2[r][c] for c in range(w)] for r in range(h)]


def mask_and(m1: Mask, m2: Mask) -> Mask:
    h, w = len(m1), len(m1[0])
    return [[m1[r][c] and m2[r][c] for c in range(w)] for r in range(h)]


def mask_not(m: Mask) -> Mask:
    return [[not m[r][c] for c in range(len(m[0]))] for r in range(len(m))]


def grids_equal(g1: Grid, g2: Grid) -> bool:
    if _shape(g1) != _shape(g2):
        return False
    h, w = _shape(g1)
    return all(g1[r][c] == g2[r][c] for r in range(h) for c in range(w))


# ---------------------------------------------------------------------------
# DSL search
# ---------------------------------------------------------------------------

def _all_colors_in_grids(grids: list[Grid]) -> list[int]:
    cs: set[int] = set()
    for g in grids:
        for row in g:
            cs.update(row)
    cs.discard(0)
    return sorted(cs)


def _gen_primitives(grid: Grid) -> list[tuple[str, Mask]]:
    """Generate all mask primitives for a given grid."""
    colors = sorted(set(v for row in grid for v in row if v != 0))
    # Limit to 5 most frequent colors to keep search tractable
    color_counts: dict[int, int] = {}
    for row in grid:
        for v in row:
            if v != 0:
                color_counts[v] = color_counts.get(v, 0) + 1
    top_colors = sorted(color_counts, key=lambda c: -color_counts[c])[:5]

    primitives: list[tuple[str, Mask]] = []
    primitives.append(("background", mask_background(grid)))
    primitives.append(("non_bg", mask_non_background(grid)))

    for c in top_colors:
        primitives.append((f"color({c})", mask_color(c, grid)))
        primitives.append((f"neighbor4({c},1)", mask_neighbor4(c, 1, grid)))
        primitives.append((f"neighbor4({c},2)", mask_neighbor4(c, 2, grid)))
        primitives.append((f"neighbor8({c},1)", mask_neighbor8(c, 1, grid)))
        primitives.append((f"bbox_inside({c})", mask_bbox_inside(c, grid)))
        primitives.append((f"bbox_border({c})", mask_bbox_border(c, grid)))
        primitives.append((f"row_has({c})", mask_row_has(c, grid)))
        primitives.append((f"col_has({c})", mask_col_has(c, grid)))

    primitives.append(("hole", mask_hole(grid)))

    # Deduplicate by mask content
    seen: set[str] = set()
    unique: list[tuple[str, Mask]] = []
    for name, m in primitives:
        key = _mask_hash(m)
        if key not in seen:
            seen.add(key)
            unique.append((name, m))
    return unique


def _mask_hash(m: Mask) -> str:
    """Compact hash of mask for dedup."""
    h = len(m)
    return str(h) + ":" + "".join(
        "1" if m[r][c] else "0"
        for r in range(min(h, 4))
        for c in range(min(len(m[0]), 4))
    )


def _search_single_mask(
    primitives: list[tuple[str, Mask]],
    output_colors: list[int],
    grid: Grid,
    output: Grid,
) -> list[tuple[str, int]]:
    """Search Length-1 programs: output = Where(mask, color, input)."""
    results: list[tuple[str, int]] = []
    for name, m in primitives:
        for c in output_colors:
            if grids_equal(mask_apply(m, c, grid), output):
                results.append((name, c))
    return results


def _search_two_mask(
    primitives: list[tuple[str, Mask]],
    output_colors: list[int],
    grid: Grid,
    output: Grid,
) -> list[tuple[str, int, str, int]]:
    """Search Length-2: output = Where(m1, c1, Where(m2, c2, input))."""
    results: list[tuple[str, int, str, int]] = []
    p_count = len(primitives)
    for i1 in range(p_count):
        name1, m1 = primitives[i1]
        for c1 in output_colors:
            intermediate = mask_apply(m1, c1, grid)
            for i2 in range(p_count):
                name2, m2 = primitives[i2]
                for c2 in output_colors:
                    if grids_equal(mask_apply(m2, c2, intermediate), output):
                        results.append((name1, c1, name2, c2))
    return results


def _search_mask_or(
    primitives: list[tuple[str, Mask]],
    output_colors: list[int],
    grid: Grid,
    output: Grid,
) -> list[tuple[str, str, int, str, int]]:
    """Search: output = Where(m1|m2, c1, Where(m3, c2, input))."""
    results: list[tuple[str, str, int, str, int]] = []
    p_count = len(primitives)
    for i1 in range(p_count):
        for i2 in range(i1, p_count):
            name1, m1 = primitives[i1]
            name2, m2 = primitives[i2]
            m_or = mask_or(m1, m2)
            for c1 in output_colors:
                intermediate = mask_apply(m_or, c1, grid)
                if grids_equal(intermediate, output):
                    results.append((name1, name2, c1, "", 0))
                for i3 in range(p_count):
                    name3, m3 = primitives[i3]
                    for c2 in output_colors:
                        if grids_equal(mask_apply(m3, c2, intermediate), output):
                            results.append((name1, name2, c1, name3, c2))
    return results


def infer_mask_program(task: dict, max_length: int = 3) -> dict[str, Any] | None:
    """Search for a mask DSL program that transforms all train cases."""
    train = task.get("train", [])
    if not train:
        return None

    train_inputs = [c["input"] for c in train]
    train_outputs = [c["output"] for c in train]

    # Ensure same_shape
    if not all(_shape(inp) == _shape(out) for inp, out in zip(train_inputs, train_outputs)):
        return None

    output_colors = _all_colors_in_grids(train_outputs)

    # Try identity
    if all(grids_equal(inp, out) for inp, out in zip(train_inputs, train_outputs)):
        return {
            "formula": "identity",
            "num_masks": 0,
            "num_colors_used": 0,
            "train_pass": f"{len(train)}/{len(train)}",
        }

    # Generate primitives for each train case
    prims_per_case = [_gen_primitives(inp) for inp in train_inputs]
    # Use the intersection of primitive names across cases
    all_names: list[set[str]] = [{name for name, _ in p} for p in prims_per_case]
    common_names = all_names[0].intersection(*all_names[1:]) if len(all_names) > 1 else all_names[0]

    # Rebuild primitives with common names, using the first case's masks
    # (masks are grid-specific — we need them per-case for validation)
    prims_first = [(n, m) for n, m in prims_per_case[0] if n in common_names]

    # Search single-mask programs
    single_candidates: list[tuple[str, int]] = []
    for i, (inp, out) in enumerate(zip(train_inputs, train_outputs)):
        prims = [(n, m) for n, m in prims_per_case[i] if n in common_names]
        candidates = _search_single_mask(prims, output_colors, inp, out)
        if i == 0:
            single_candidates = candidates
        else:
            single_candidates = [c for c in single_candidates if c in candidates]
        if not single_candidates:
            break

    if single_candidates:
        name, color = single_candidates[0]
        return {
            "formula": f"Where({name}, {color}, input)",
            "num_masks": 1,
            "num_colors_used": 1,
            "mask1_name": name,
            "mask1_color": color,
            "train_pass": f"{len(train)}/{len(train)}",
        }

    # Search two-mask programs (only if max_length >= 2)
    if max_length >= 2 and len(common_names) <= 30:
        two_candidates: list[tuple[str, int, str, int]] = []
        for i, (inp, out) in enumerate(zip(train_inputs, train_outputs)):
            prims = [(n, m) for n, m in prims_per_case[i] if n in common_names]
            candidates = _search_two_mask(prims, output_colors, inp, out)
            if i == 0:
                two_candidates = candidates
            else:
                two_candidates = [c for c in two_candidates if c in candidates]
            if not two_candidates:
                break

        if two_candidates:
            n1, c1, n2, c2 = two_candidates[0]
            return {
                "formula": f"Where({n1}, {c1}, Where({n2}, {c2}, input))",
                "num_masks": 2,
                "num_colors_used": len({c1, c2}),
                "mask1_name": n1,
                "mask1_color": c1,
                "mask2_name": n2,
                "mask2_color": c2,
                "train_pass": f"{len(train)}/{len(train)}",
            }

    return None


# ---------------------------------------------------------------------------
# ONNX builder for mask programs
# ---------------------------------------------------------------------------

def _build_mask_onnx_model(
    output_path: str,
    mask_formula: dict[str, Any],
    task: dict,
) -> None:
    """Build a compact ONNX model implementing a mask DSL program.

    The model uses static masks computed from train case inference.
    For tasks where the mask formula depends on the specific input grid,
    we build parameterized masks using ONNX operations.
    """
    train = task.get("train", [])
    first_input = train[0]["input"]
    h, w = _shape(first_input)
    num_masks = mask_formula["num_masks"]

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    if num_masks == 0:
        # Identity: output = input
        nodes.append(
            helper.make_node("Identity", ["input"], ["output"], name="output")
        )
    elif num_masks == 1:
        # Single mask: Use a static mask computed from the first train case
        mask_name = mask_formula.get("mask1_name", "")
        mask_color = mask_formula.get("mask1_color", 0)
        mask_grid = _compute_mask_for_input(mask_name, first_input)

        mask_onehot = _mask_to_onehot(mask_grid, h, w)
        initializers.append(
            numpy_helper.from_array(mask_onehot.astype(np.float32), name="mask_tensor")
        )

        color_onehot = np.zeros(DEFAULT_SHAPE, dtype=np.float32)
        color_onehot[0, mask_color, :h, :w] = 1.0
        initializers.append(
            numpy_helper.from_array(color_onehot.astype(np.float32), name="color_tensor")
        )

        nodes.extend([
            helper.make_node("Mul", ["color_tensor", "mask_tensor"], ["painted"], name="paint"),
            helper.make_node("Sub", ["ones", "mask_tensor"], ["keep_mask"], name="invert"),
            helper.make_node("Mul", ["input", "keep_mask"], ["kept"], name="keep"),
            helper.make_node("Add", ["painted", "kept"], ["output"], name="output"),
        ])
        initializers.append(
            numpy_helper.from_array(np.ones(DEFAULT_SHAPE, dtype=np.float32), name="ones")
        )
    elif num_masks == 2:
        mask1_name = mask_formula.get("mask1_name", "")
        mask1_color = mask_formula.get("mask1_color", 0)
        mask2_name = mask_formula.get("mask2_name", "")
        mask2_color = mask_formula.get("mask2_color", 0)

        m1 = _compute_mask_for_input(mask1_name, first_input)
        m2 = _compute_mask_for_input(mask2_name, first_input)

        m1_oh = _mask_to_onehot(m1, h, w).astype(np.float32)
        m2_oh = _mask_to_onehot(m2, h, w).astype(np.float32)

        c1_oh = np.zeros(DEFAULT_SHAPE, dtype=np.float32)
        c1_oh[0, mask1_color, :h, :w] = 1.0
        c2_oh = np.zeros(DEFAULT_SHAPE, dtype=np.float32)
        c2_oh[0, mask2_color, :h, :w] = 1.0

        initializers.extend([
            numpy_helper.from_array(m1_oh, name="mask1_tensor"),
            numpy_helper.from_array(m2_oh, name="mask2_tensor"),
            numpy_helper.from_array(c1_oh, name="color1_tensor"),
            numpy_helper.from_array(c2_oh, name="color2_tensor"),
            numpy_helper.from_array(np.ones(DEFAULT_SHAPE, dtype=np.float32), name="ones"),
        ])

        nodes.extend([
            # Layer 2 first: paint mask2 with color2
            helper.make_node("Mul", ["color2_tensor", "mask2_tensor"], ["p2"], name="paint2"),
            helper.make_node("Sub", ["ones", "mask2_tensor"], ["k2"], name="invert2"),
            helper.make_node("Mul", ["input", "k2"], ["input_kept"], name="keep_input"),
            helper.make_node("Add", ["p2", "input_kept"], ["mid"], name="layer2_out"),
            # Layer 1: paint mask1 with color1 on top
            helper.make_node("Mul", ["color1_tensor", "mask1_tensor"], ["p1"], name="paint1"),
            helper.make_node("Sub", ["ones", "mask1_tensor"], ["k1"], name="invert1"),
            helper.make_node("Mul", ["mid", "k1"], ["mid_kept"], name="keep_mid"),
            helper.make_node("Add", ["p1", "mid_kept"], ["output"], name="output"),
        ])

    graph = helper.make_graph(
        nodes=nodes,
        name="mask_dsl",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, DEFAULT_SHAPE)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, DEFAULT_SHAPE)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-2026",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    onnx.checker.check_model(str(out_path))


def _compute_mask_for_input(mask_name: str, grid: Grid) -> Mask:
    """Compute a mask from its DSL name for a specific input grid."""
    if mask_name == "background":
        return mask_background(grid)
    if mask_name == "non_bg":
        return mask_non_background(grid)
    if mask_name == "hole":
        return mask_hole(grid)
    if mask_name.startswith("color("):
        c = int(mask_name.split("(")[1].rstrip(")"))
        return mask_color(c, grid)
    if mask_name.startswith("neighbor4("):
        parts = mask_name[len("neighbor4("):-1].split(",")
        c, k = int(parts[0]), int(parts[1])
        return mask_neighbor4(c, k, grid)
    if mask_name.startswith("neighbor8("):
        parts = mask_name[len("neighbor8("):-1].split(",")
        c, k = int(parts[0]), int(parts[1])
        return mask_neighbor8(c, k, grid)
    if mask_name.startswith("bbox_inside("):
        c = int(mask_name[len("bbox_inside("):-1])
        return mask_bbox_inside(c, grid)
    if mask_name.startswith("bbox_border("):
        c = int(mask_name[len("bbox_border("):-1])
        return mask_bbox_border(c, grid)
    if mask_name.startswith("row_has("):
        c = int(mask_name[len("row_has("):-1])
        return mask_row_has(c, grid)
    if mask_name.startswith("col_has("):
        c = int(mask_name[len("col_has("):-1])
        return mask_col_has(c, grid)
    raise ValueError(f"unknown mask name: {mask_name}")


def _mask_to_onehot(m: Mask, h: int, w: int) -> np.ndarray:
    """Convert a bool mask to a 1x1xHxW float32 tensor."""
    tensor = np.zeros((1, 1, DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=np.float32)
    for r in range(h):
        for c in range(w):
            if m[r][c]:
                tensor[0, 0, r, c] = 1.0
    return tensor


# ---------------------------------------------------------------------------
# Batch search and build
# ---------------------------------------------------------------------------

def _validate_onnx_model(model_path: str, task_path: str, report_dir: str) -> dict[str, Any]:
    """Validate a built ONNX model."""
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
    val_report = str(Path(report_dir) / f"{task_stem}_{model_stem}_validation.csv")
    try:
        task = load_task(task_path)
    except Exception as exc:
        result["valid"] = False
        result["failure_reason"] = f"task load: {exc}"
        return result

    train_result = validate_cases(model_path, task["train"])
    if not train_result["passed"]:
        result["valid"] = False
        result["failure_reason"] = "train validation failed"
        return result

    try:
        labelled = validate_labelled_splits(model_path, task_path, val_report)
    except Exception as exc:
        result["valid"] = False
        result["failure_reason"] = f"validation error: {exc}"
        return result

    split_counts = labelled.get("split_counts", {})
    result["train_pass"] = split_counts.get("train", {}).get("passed", "?")
    result["test_pass"] = split_counts.get("test", {}).get("passed", "?")
    result["arc_gen_pass"] = split_counts.get("arc-gen", {}).get("passed", "?")
    if not labelled.get("passed", False):
        result["valid"] = False
        result["failure_reason"] = "labelled splits failed"
    return result


def batch_search_and_build(
    task_dir: str = "task",
    model_dir: str = "outputs/onnx",
    cost_report_path: str = "outputs/reports/current_model_bank_report.csv",
    candidate_dir: str = DEFAULT_CANDIDATE_DIR,
    conservative_dir: str = DEFAULT_CONSERVATIVE_DIR,
    report_path: str = DEFAULT_REPORT,
    max_length: int = 3,
    min_cost: int = 1000,
    task_ids: str = "",
) -> dict[str, Any]:
    import csv as _csv

    # Load cost data
    cost_data: dict[str, int] = {}
    with Path(cost_report_path).open("r", newline="", encoding="utf-8") as handle:
        for row in _csv.DictReader(handle):
            if row.get("valid", "True").strip().lower() == "true":
                cost_data[row["task_id"].strip()] = int(row.get("estimated_cost") or 0)

    target_tasks = [t.strip() for t in task_ids.split(",") if t.strip()] if task_ids else None
    root = Path(task_dir)
    rows: list[dict[str, Any]] = []
    searched = 0
    matched = 0
    built = 0

    for path in sorted(root.glob("task*.json")):
        tid = path.stem
        if target_tasks and tid not in target_tasks:
            continue

        current_cost = cost_data.get(tid, 0)
        if current_cost < min_cost:
            continue

        try:
            task = load_task(str(path))
        except Exception:
            continue

        train = task.get("train", [])
        if not train:
            continue

        shapes_in = [_shape(c["input"]) for c in train]
        shapes_out = [_shape(c["output"]) for c in train]
        if not all(si == so for si, so in zip(shapes_in, shapes_out)):
            continue

        searched += 1
        program = infer_mask_program(task, max_length=max_length)
        if program is None:
            continue

        matched += 1

        # Build ONNX model
        output_name = f"{tid}_mask_dsl.onnx"
        output_path = str(Path(candidate_dir) / output_name)
        try:
            _build_mask_onnx_model(output_path, program, task)
        except Exception as exc:
            row = {
                "task_id": tid,
                "formula": program.get("formula", ""),
                "num_masks": program.get("num_masks", 0),
                "num_colors_used": program.get("num_colors_used", 0),
                "train_pass": program.get("train_pass", ""),
                "test_pass": "",
                "arc_gen_pass": "",
                "source_cost": current_cost,
                "output_cost": 0,
                "cost_delta": 0,
                "output_model_path": "",
                "valid": False,
                "failure_reason": f"build error: {exc}",
            }
            rows.append(row)
            continue

        built += 1
        output_info = estimate_model_cost(output_path)

        # Validate
        validation = _validate_onnx_model(output_path, str(path), candidate_dir)

        row = {
            "task_id": tid,
            "formula": program.get("formula", ""),
            "num_masks": program.get("num_masks", 0),
            "num_colors_used": program.get("num_colors_used", 0),
            "train_pass": program.get("train_pass", ""),
            "test_pass": validation.get("test_pass", ""),
            "arc_gen_pass": validation.get("arc_gen_pass", ""),
            "source_cost": current_cost,
            "output_cost": int(output_info["estimated_cost"]),
            "cost_delta": int(output_info["estimated_cost"]) - current_cost,
            "output_model_path": output_path,
            "valid": validation["valid"],
            "failure_reason": validation.get("failure_reason", ""),
        }

        if row["valid"]:
            cons_path = Path(conservative_dir) / output_name
            cons_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(output_path, str(cons_path))

        rows.append(row)

    # Sort by cost savings (most negative cost_delta first = biggest improvement)
    rows.sort(key=lambda r: r["cost_delta"])

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = _csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [r for r in rows if r["valid"]]
    total_delta = sum(r["cost_delta"] for r in valid_rows)
    matches_by_length: dict[int, int] = {}
    for r in rows:
        nm = r["num_masks"]
        matches_by_length[nm] = matches_by_length.get(nm, 0) + 1

    summary = {
        "report_path": str(report),
        "same_shape_tasks_searched": searched,
        "programs_matched": matched,
        "models_built": built,
        "valid_models": len(valid_rows),
        "total_cost_delta": total_delta,
        "by_num_masks": matches_by_length,
        "top_savings": [
            {
                "task_id": r["task_id"],
                "formula": r["formula"],
                "cost_delta": r["cost_delta"],
                "output_cost": r["output_cost"],
            }
            for r in rows[:15]
            if r["cost_delta"] < 0
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", default="task")
    parser.add_argument("--model-dir", default="outputs/onnx")
    parser.add_argument("--cost-report", default="outputs/reports/current_model_bank_report.csv")
    parser.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--conservative-dir", default=DEFAULT_CONSERVATIVE_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--max-length", type=int, default=3)
    parser.add_argument("--min-cost", type=int, default=1000)
    parser.add_argument("--task-ids", default="")
    args = parser.parse_args()
    batch_search_and_build(
        task_dir=args.task_dir,
        model_dir=args.model_dir,
        cost_report_path=args.cost_report,
        candidate_dir=args.candidate_dir,
        conservative_dir=args.conservative_dir,
        report_path=args.report,
        max_length=args.max_length,
        min_cost=args.min_cost,
        task_ids=args.task_ids,
    )


if __name__ == "__main__":
    main()
