"""Rebuild the 400-task taxonomy with structural, color, and semantic features.

Classifies every task into one of 12 families (A-L) and extracts detailed
features from the ARC task JSON data without touching ONNX models.

outputs/reports/task_family_taxonomy_v2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

from .arc_io import load_task


DEFAULT_REPORT = "outputs/reports/task_family_taxonomy_v2.csv"
FIELDS = [
    "task_id",
    "input_shape_relation",
    "output_shape_relation",
    "same_shape",
    "crop",
    "expand",
    "panel",
    "scale",
    "num_colors_in",
    "num_colors_out",
    "color_role_stability",
    "has_frame",
    "has_panel_separator",
    "has_objects",
    "has_lines",
    "has_holes",
    "has_symmetry",
    "has_periodicity",
    "has_translation",
    "has_recolor",
    "has_counting",
    "has_template_copy",
    "current_cost",
    "potential_to_1000",
    "current_model_source",
    "known_online_status",
    "family",
    "family_reason",
]

Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def _shape(g: Grid) -> tuple[int, int]:
    return len(g), len(g[0])


def _shapes_in_out(task: dict) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    train = task.get("train", [])
    s_in = [_shape(c["input"]) for c in train]
    s_out = [_shape(c["output"]) for c in train]
    return s_in, s_out


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _colors(grids: list[Grid]) -> set[int]:
    c: set[int] = set()
    for g in grids:
        for row in g:
            c.update(row)
    return c


def _color_role_stability(task: dict) -> float:
    """Fraction of colors whose input→output mapping is consistent across train cases."""
    train = task.get("train", [])
    transitions: dict[int, set[int]] = {}
    for case in train:
        inp = case["input"]
        out = case["output"]
        if _shape(inp) != _shape(out):
            return 0.0
        h, w = _shape(inp)
        for r in range(h):
            for c in range(w):
                ic = inp[r][c]
                oc = out[r][c]
                transitions.setdefault(ic, set()).add(oc)
    stable = sum(1 for v in transitions.values() if len(v) == 1)
    return round(stable / max(1, len(transitions)), 4)


# ---------------------------------------------------------------------------
# Structure detection
# ---------------------------------------------------------------------------

def _has_frame(grid: Grid) -> bool:
    """Check if the grid border is a single uniform non-zero color."""
    h, w = _shape(grid)
    if h < 3 or w < 3:
        return False
    border_color = grid[0][0]
    if border_color == 0:
        return False
    for c in range(w):
        if grid[0][c] != border_color or grid[h - 1][c] != border_color:
            return False
    for r in range(1, h - 1):
        if grid[r][0] != border_color or grid[r][w - 1] != border_color:
            return False
    interior_colors = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            interior_colors.add(grid[r][c])
    return len(interior_colors - {border_color, 0}) > 0


def _has_panel_separator(grid: Grid) -> bool:
    """Detect full-row or full-column dividers of a uniform non-zero color."""
    h, w = _shape(grid)
    for r in range(1, h - 1):
        row_vals = set(grid[r])
        if len(row_vals) == 1 and 0 not in row_vals:
            return True
    for c in range(1, w - 1):
        col_vals = {grid[r][c] for r in range(h)}
        if len(col_vals) == 1 and 0 not in col_vals:
            return True
    return False


def _has_objects(grid: Grid) -> bool:
    """Check for multiple distinct non-background connected components."""
    h, w = _shape(grid)
    seen: set[tuple[int, int]] = set()
    components = 0
    for r in range(h):
        for c in range(w):
            if (r, c) in seen or grid[r][c] == 0:
                continue
            components += 1
            if components >= 2:
                return True
            queue = deque([(r, c)])
            seen.add((r, c))
            while queue:
                cr, cc = queue.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and grid[nr][nc] != 0:
                        seen.add((nr, nc))
                        queue.append((nr, nc))
    return components >= 2


def _has_lines(grid: Grid) -> bool:
    """Detect complete rows or columns of a single non-zero color."""
    h, w = _shape(grid)
    for r in range(h):
        if len(set(grid[r])) == 1 and grid[r][0] != 0:
            return True
    for c in range(w):
        col = {grid[r][c] for r in range(h)}
        if len(col) == 1 and 0 not in col:
            return True
    return False


def _has_holes(grid: Grid) -> bool:
    """Flood-fill from border; any unreachable background cell is a hole."""
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
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and (r, c) not in reachable:
                return True
    return False


# ---------------------------------------------------------------------------
# Geometry detection
# ---------------------------------------------------------------------------

def _has_symmetry(grid: Grid) -> str:
    """Check for horizontal, vertical, or rotational symmetry."""
    h, w = _shape(grid)
    h_sym = all(grid[r][c] == grid[r][w - 1 - c] for r in range(h) for c in range(w // 2))
    v_sym = all(grid[r][c] == grid[h - 1 - r][c] for r in range(h // 2) for c in range(w))
    if h_sym and v_sym:
        return "both"
    if h_sym:
        return "horizontal"
    if v_sym:
        return "vertical"
    if h >= 2 and w >= 2:
        rot = all(grid[r][c] == grid[h - 1 - r][w - 1 - c] for r in range(h) for c in range(w))
        if rot:
            return "rotational"
    return "none"


def _has_periodicity(grid: Grid) -> bool:
    """Check if the grid can be decomposed into a repeating tile."""
    h, w = _shape(grid)
    for tile_h in range(1, h // 2 + 1):
        if h % tile_h != 0:
            continue
        for tile_w in range(1, w // 2 + 1):
            if w % tile_w != 0:
                continue
            tile = [row[:tile_w] for row in grid[:tile_h]]
            ok = True
            for tr in range(0, h, tile_h):
                for tc in range(0, w, tile_w):
                    for r in range(tile_h):
                        for c in range(tile_w):
                            if grid[tr + r][tc + c] != tile[r][c]:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                return True
    return False


def _has_translation(task: dict) -> bool:
    """Check if output is a translation of input across train cases."""
    train = task.get("train", [])
    for case in train:
        inp = case["input"]
        out = case["output"]
        if _shape(inp) != _shape(out):
            return False
    for case in train:
        inp = case["input"]
        out = case["output"]
        h, w = _shape(inp)
        non_bg_in = [(r, c, inp[r][c]) for r in range(h) for c in range(w) if inp[r][c] != 0]
        non_bg_out = {(r, c): out[r][c] for r in range(h) for c in range(w) if out[r][c] != 0}
        if not non_bg_in or len(non_bg_in) != len(non_bg_out):
            continue
        ri, ci, _ = non_bg_in[0]
        found = False
        for (ro, co), val in non_bg_out.items():
            dr, dc = ro - ri, co - ci
            match = True
            for r, c, v in non_bg_in:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in non_bg_out or non_bg_out[(nr, nc)] != v:
                    match = False
                    break
            if match:
                found = True
                break
        if not found:
            return False
    return len(train) > 0


# ---------------------------------------------------------------------------
# Semantic detection
# ---------------------------------------------------------------------------

def _has_recolor(task: dict) -> bool:
    """At least one output color not present in any input."""
    train = task.get("train", [])
    inputs = [c["input"] for c in train]
    outputs = [c["output"] for c in train]
    return bool(_colors(outputs) - _colors(inputs))


def _has_counting(task: dict) -> bool:
    """Output is 1xN or Nx1 where N equals object/color count in input."""
    train = task.get("train", [])
    for case in train:
        out = case["output"]
        inp = case["input"]
        oh, ow = _shape(out)
        num_objs = sum(1 for r in range(len(inp)) for c in range(len(inp[0])) if inp[r][c] != 0)
        num_colors = len(set(v for row in inp for v in row if v != 0))
        if oh == 1 and ow > 0:
            if ow in (num_objs, num_colors):
                return True
        if ow == 1 and oh > 0:
            if oh in (num_objs, num_colors):
                return True
    return False


def _has_template_copy(task: dict) -> bool:
    """A pattern from input appears at multiple locations in output."""
    train = task.get("train", [])
    for case in train:
        inp = case["input"]
        out = case["output"]
        ih, iw = _shape(inp)
        oh, ow = _shape(out)
        for th in range(2, min(ih, oh // 2) + 1):
            for tw in range(2, min(iw, ow // 2) + 1):
                for ir in range(ih - th + 1):
                    for ic in range(iw - tw + 1):
                        template = [inp[ir + r][ic + c] for r in range(th) for c in range(tw)]
                        if all(v == 0 for v in template):
                            continue
                        matches = 0
                        for or_ in range(oh - th + 1):
                            for oc in range(ow - tw + 1):
                                if all(
                                    out[or_ + r][oc + c] == template[r * tw + c]
                                    for r in range(th)
                                    for c in range(tw)
                                ):
                                    matches += 1
                        if matches >= 2:
                            return True
    return False


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _shape_relation_label(s_in, s_out) -> str:
    if all(si == so for si, so in zip(s_in, s_out)):
        return "same_shape"
    if all(so[0] * so[1] < si[0] * si[1] for si, so in zip(s_in, s_out)):
        return "crop"
    if all(so[0] * so[1] > si[0] * si[1] for si, so in zip(s_in, s_out)):
        return "expand"
    return "mixed"


def _is_scale(s_in, s_out) -> bool:
    """Check if output dimensions are integer multiples of input."""
    return all(
        so[0] % si[0] == 0 and so[1] % si[1] == 0 and (so[0] // si[0]) == (so[1] // si[1])
        for si, so in zip(s_in, s_out)
        if so[0] >= si[0] and so[1] >= si[1]
    )


def classify_family(row: dict[str, Any]) -> tuple[str, str]:
    """Assign a task to one of families A-L."""
    same = row.get("same_shape") is True
    crop = row.get("crop") is True
    expand = row.get("expand") is True
    panel = row.get("panel") is True
    scale = row.get("scale") is True
    has_frame = row.get("has_frame") is True
    has_sep = row.get("has_panel_separator") is True
    has_objects = row.get("has_objects") is True
    has_lines = row.get("has_lines") is True
    has_holes = row.get("has_holes") is True
    has_sym = row.get("has_symmetry") not in ("none", "", None)
    has_period = row.get("has_periodicity") is True
    has_trans = row.get("has_translation") is True
    has_recolor = row.get("has_recolor") is True
    has_counting = row.get("has_counting") is True
    has_tmpl = row.get("has_template_copy") is True
    stability = float(row.get("color_role_stability") or 0)
    cost = int(row.get("current_cost") or 0)

    # L: High-cost tasks that are likely table-based representation rewrites
    if cost > 100000:
        return "L", "high_cost_representation_rewrite_candidate"

    # A: same-shape local mask / recolor
    if same and has_recolor and stability > 0.5:
        return "A", "same_shape_local_mask_or_recolor"

    # B: bbox crop / object extraction
    if crop and (has_frame or has_objects):
        return "B", "bbox_crop_or_object_extraction"

    # C: panel selection / panel algebra
    if has_sep or panel:
        return "C", "panel_selection_or_algebra"

    # D: line completion / projection / fill
    if has_lines and same:
        return "D", "line_completion_or_projection"

    # E: hole fill / enclosed region fill
    if has_holes and same:
        return "E", "hole_fill_or_enclosed_region"

    # F: symmetry / mirror / rotation completion
    if has_sym and same:
        return "F", "symmetry_or_mirror_completion"

    # G: periodic extension / tiling
    if has_period:
        return "G", "periodic_extension_or_tiling"

    # H: object copy / marker-controlled paste
    if has_objects and has_tmpl:
        return "H", "object_copy_or_marker_paste"

    # I: color role remapping
    if has_recolor and not same:
        return "I", "color_role_remapping"

    # J: scale / repeat / kron
    if scale:
        return "J", "scale_repeat_or_kron"

    # K: template matching / finite pattern completion
    if has_tmpl:
        return "K", "template_matching_or_finite_pattern"

    # Fallback based on shape relation
    if has_trans and same:
        return "A", "same_shape_translation_based"
    if same:
        if has_recolor:
            return "A", "same_shape_with_recolor"
        return "A", "same_shape_generic"
    if crop:
        return "B", "crop_generic"
    if expand:
        return "J", "expand_generic"
    return "L", "unclassified_representation_candidate"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_score(cost: float) -> float:
    return max(1.0, 25.0 - math.log(max(1, cost)))


def load_cost_data(report_path: str) -> dict[str, dict[str, Any]]:
    data: dict[str, dict[str, Any]] = {}
    with Path(report_path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            tid = row["task_id"].strip()
            if row.get("valid", "True").strip().lower() != "true":
                continue
            data[tid] = {
                "current_cost": int(row.get("estimated_cost") or 0),
                "model_source": row.get("model_path", ""),
            }
    return data


def load_online_status(reports_dir: str) -> set[str]:
    """Collect task_ids that have been tried online (any ablation)."""
    tried: set[str] = set()
    root = Path(reports_dir)
    for path in sorted(root.glob("*.csv")):
        if "ablation" not in path.stem.lower():
            continue
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    tid = row.get("task_id", "").strip()
                    if tid:
                        tried.add(tid)
        except Exception:
            continue
    return tried


def build_taxonomy(
    task_dir: str = "task",
    cost_report_path: str = "outputs/reports/current_model_bank_report.csv",
    report_path: str = DEFAULT_REPORT,
    reports_dir: str = "outputs/reports",
) -> dict[str, Any]:
    cost_data = load_cost_data(cost_report_path)
    online_tasks = load_online_status(reports_dir)
    root = Path(task_dir)

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in sorted(root.glob("task*.json")):
        tid = path.stem
        try:
            task = load_task(str(path))
        except Exception:
            missing.append(tid)
            continue

        s_in, s_out = _shapes_in_out(task)
        relation = _shape_relation_label(s_in, s_out)
        train = task.get("train", [])

        # Feature extraction per train case, aggregate with "any" logic
        inputs = [c["input"] for c in train]
        outputs = [c["output"] for c in train]
        colors_in = _colors(inputs)
        colors_out = _colors(outputs)

        any_frame = any(_has_frame(g) for g in inputs)
        any_sep = any(_has_panel_separator(g) for g in inputs)
        any_objects = any(_has_objects(g) for g in inputs)
        any_lines = any(_has_lines(g) for g in outputs)
        any_holes = any(_has_holes(g) for g in inputs) or any(_has_holes(g) for g in outputs)
        syms = [_has_symmetry(g) for g in inputs]
        best_sym = "none"
        for s in ["both", "horizontal", "vertical", "rotational"]:
            if s in syms:
                best_sym = s
                break
        any_period = any(_has_periodicity(g) for g in inputs)
        has_trans = _has_translation(task)
        has_recolor = bool(colors_out - colors_in)
        has_count = _has_counting(task)
        has_tmpl = _has_template_copy(task)
        stability = _color_role_stability(task)

        cost_info = cost_data.get(tid, {})
        current_cost = cost_info.get("current_cost", 0)
        current_score = compute_score(current_cost)
        score_if_1000 = compute_score(1000)
        potential = max(0, round(score_if_1000 - current_score, 4))

        row = {
            "task_id": tid,
            "input_shape_relation": ";".join(f"{h}x{w}" for h, w in s_in),
            "output_shape_relation": ";".join(f"{h}x{w}" for h, w in s_out),
            "same_shape": relation == "same_shape",
            "crop": relation == "crop",
            "expand": relation == "expand",
            "panel": any_sep,
            "scale": relation == "expand" and _is_scale(s_in, s_out),
            "num_colors_in": len(colors_in),
            "num_colors_out": len(colors_out),
            "color_role_stability": stability,
            "has_frame": any_frame,
            "has_panel_separator": any_sep,
            "has_objects": any_objects,
            "has_lines": any_lines,
            "has_holes": any_holes,
            "has_symmetry": best_sym,
            "has_periodicity": any_period,
            "has_translation": has_trans,
            "has_recolor": has_recolor,
            "has_counting": has_count,
            "has_template_copy": has_tmpl,
            "current_cost": current_cost,
            "potential_to_1000": potential,
            "current_model_source": cost_info.get("model_source", ""),
            "known_online_status": "tried" if tid in online_tasks else "",
        }

        family, reason = classify_family(row)
        row["family"] = family
        row["family_reason"] = reason
        rows.append(row)

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_family: dict[str, int] = {}
    total_cost_by_family: dict[str, int] = {}
    for r in rows:
        f = r["family"]
        by_family[f] = by_family.get(f, 0) + 1
        total_cost_by_family[f] = total_cost_by_family.get(f, 0) + int(r["current_cost"])

    summary = {
        "report_path": str(report),
        "task_count": len(rows),
        "missing_tasks": missing,
        "by_family": {k: by_family[k] for k in sorted(by_family)},
        "total_cost_by_family": {
            k: f"{total_cost_by_family[k]:,}" for k in sorted(total_cost_by_family)
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", default="task")
    parser.add_argument("--cost-report", default="outputs/reports/current_model_bank_report.csv")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--reports-dir", default="outputs/reports")
    args = parser.parse_args()
    build_taxonomy(
        task_dir=args.task_dir,
        cost_report_path=args.cost_report,
        report_path=args.report,
        reports_dir=args.reports_dir,
    )


if __name__ == "__main__":
    main()
