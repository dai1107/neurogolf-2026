"""Object/Crop Selector v2 — DSL-based object selection and crop optimization.

Direction 4 of the optimization strategy. Two-layer approach:
  Python DSL search: connected-component analysis for pattern discovery
  ONNX compile: finite bbox/color/mask selectors, no generic CC in graph

Selectors:
  non_background, color(c), largest_object, touching_border,
  not_touching_border, inside_frame(fc)

Actions:
  crop_to_bbox, identity, color_map, horizontal_mirror, vertical_mirror

Output: outputs/reports/object_crop_selector_v2_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .arc_io import load_task
from .cost_estimator import estimate_model_cost, check_forbidden_ops, check_static_shapes
from .encoding import DEFAULT_SHAPE
from .validate_labelled_splits import validate_labelled_splits


DEFAULT_REPORT = "outputs/reports/object_crop_selector_v2_report.csv"
DEFAULT_CANDIDATE_DIR = "outputs/candidates/object_crop_selector_v2"
FIELDS = [
    "task_id", "program", "selector_type", "action_type",
    "train_pass", "test_pass", "arc_gen_pass",
    "source_cost", "output_cost", "cost_delta",
    "valid", "failure_reason", "family",
]

Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Connected-component helpers (Python probe only, not ONNX)
# ---------------------------------------------------------------------------

def _shape(g: Grid) -> tuple[int, int]:
    return len(g), len(g[0])


def _find_components(grid: Grid, color: int) -> list[set[tuple[int, int]]]:
    """BFS connected components of a given color. 4-connected."""
    h, w = _shape(grid)
    visited = [[False] * w for _ in range(h)]
    components: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == color and not visited[r][c]:
                comp: set[tuple[int, int]] = set()
                q = deque([(r, c)])
                visited[r][c] = True
                while q:
                    cr, cc = q.popleft()
                    comp.add((cr, cc))
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and grid[nr][nc] == color:
                            visited[nr][nc] = True
                            q.append((nr, nc))
                if comp:
                    components.append(comp)
    return components


def _bbox_of(pixels: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    """(r1, c1, r2, c2) inclusive-exclusive."""
    if not pixels:
        return (0, 0, 0, 0)
    rs = [p[0] for p in pixels]
    cs = [p[1] for p in pixels]
    return (min(rs), min(cs), max(rs) + 1, max(cs) + 1)


def _touches_border(pixels: set[tuple[int, int]], h: int, w: int) -> bool:
    for r, c in pixels:
        if r == 0 or r == h - 1 or c == 0 or c == w - 1:
            return True
    return False


# ---------------------------------------------------------------------------
# Selectors (Python probe)
# ---------------------------------------------------------------------------

def select_non_background(grid: Grid) -> set[tuple[int, int]]:
    """All non-zero pixels."""
    h, w = _shape(grid)
    return {(r, c) for r in range(h) for c in range(w) if grid[r][c] != 0}


def select_color(grid: Grid, color: int) -> set[tuple[int, int]]:
    h, w = _shape(grid)
    return {(r, c) for r in range(h) for c in range(w) if grid[r][c] == color}


def select_largest_object(grid: Grid, color: int | None = None) -> set[tuple[int, int]] | None:
    """Largest connected component. If color is None, try all non-zero colors."""
    h, w = _shape(grid)
    if color is not None:
        comps = _find_components(grid, color)
        return max(comps, key=len) if comps else None
    best = None
    for c in range(1, 10):
        comps = _find_components(grid, c)
        for comp in comps:
            if best is None or len(comp) > len(best):
                best = comp
    return best


def select_smallest_object(grid: Grid, color: int | None = None) -> set[tuple[int, int]] | None:
    if color is not None:
        comps = _find_components(grid, color)
        return min(comps, key=len) if comps else None
    best = None
    for c in range(1, 10):
        comps = _find_components(grid, c)
        for comp in comps:
            if best is None or len(comp) < len(best):
                best = comp
    return best


def select_touching_border(grid: Grid) -> set[tuple[int, int]]:
    """All non-zero pixels touching the grid border."""
    h, w = _shape(grid)
    result: set[tuple[int, int]] = set()
    for c in range(1, 10):
        for comp in _find_components(grid, c):
            if _touches_border(comp, h, w):
                result |= comp
    return result


def select_not_touching_border(grid: Grid) -> set[tuple[int, int]]:
    """All non-zero pixels NOT touching the grid border."""
    h, w = _shape(grid)
    result: set[tuple[int, int]] = set()
    for c in range(1, 10):
        for comp in _find_components(grid, c):
            if not _touches_border(comp, h, w):
                result |= comp
    return result


def select_inside_frame(grid: Grid, frame_color: int) -> set[tuple[int, int]] | None:
    """Select all pixels inside a rectangular frame of frame_color."""
    h, w = _shape(grid)
    frame_pixels = {(r, c) for r in range(h) for c in range(w) if grid[r][c] == frame_color}
    if not frame_pixels:
        return None
    r1, c1, r2, c2 = _bbox_of(frame_pixels)
    # Verify frame is a hollow rectangle (only on border of its bbox)
    interior: set[tuple[int, int]] = set()
    for r in range(r1, r2):
        for c in range(c1, c2):
            if grid[r][c] != frame_color:
                interior.add((r, c))
    # Only return interior if frame forms actual border
    frame_border = {(r, c) for r, c in frame_pixels if r == r1 or r == r2 - 1 or c == c1 or c == c2 - 1}
    if len(frame_border) >= 0.8 * len(frame_pixels):
        return interior
    return None


# ---------------------------------------------------------------------------
# DSL Program search
# ---------------------------------------------------------------------------

def _grid_from_pixels(pixels: set[tuple[int, int]], ref_grid: Grid) -> Grid:
    """Create a grid with only the selected pixels, others 0."""
    h, w = _shape(ref_grid)
    result = [[0] * w for _ in range(h)]
    for r, c in pixels:
        result[r][c] = ref_grid[r][c]
    return result


def _crop_grid(grid: Grid, bbox: tuple[int, int, int, int]) -> Grid:
    r1, c1, r2, c2 = bbox
    return [row[c1:c2] for row in grid[r1:r2]]


def _identity_action(cropped: Grid) -> Grid:
    return cropped


def _horizontal_mirror_action(cropped: Grid) -> Grid:
    return [row[::-1] for row in cropped]


def _vertical_mirror_action(cropped: Grid) -> Grid:
    return cropped[::-1]


def _color_map_action(cropped: Grid, mapping: dict[int, int]) -> Grid:
    return [[mapping.get(v, v) for v in row] for row in cropped]


def _infer_color_map(input_grid: Grid, output_grid: Grid) -> dict[int, int] | None:
    """Infer a color mapping from input to output. Returns None if shapes differ."""
    if _shape(input_grid) != _shape(output_grid):
        return None
    mapping: dict[int, int] = {}
    for r in range(len(input_grid)):
        for c in range(len(input_grid[0])):
            iv = input_grid[r][c]
            ov = output_grid[r][c]
            if iv != ov:
                mapping[iv] = ov
    return mapping


def _get_pixels_for_selector(grid: Grid, selector_type: str) -> set[tuple[int, int]]:
    """Apply a selector to a grid and return the selected pixels."""
    if selector_type == "non_background":
        return select_non_background(grid)
    elif selector_type == "not_touching_border":
        return select_not_touching_border(grid)
    elif selector_type.startswith("color_"):
        color = int(selector_type.split("_")[1])
        return select_color(grid, color)
    elif selector_type == "largest_object":
        obj = select_largest_object(grid)
        return obj if obj else set()
    elif selector_type.startswith("inside_frame_"):
        fc = int(selector_type.split("_")[-1])
        interior = select_inside_frame(grid, fc)
        return interior if interior else set()
    return set()


def _apply_action(cropped: Grid, action_type: str) -> Grid:
    """Apply a post-crop action."""
    if action_type == "identity":
        return _identity_action(cropped)
    elif action_type == "horizontal_mirror":
        return _horizontal_mirror_action(cropped)
    elif action_type == "vertical_mirror":
        return _vertical_mirror_action(cropped)
    elif action_type.startswith("color_map_"):
        mapping_str = action_type[len("color_map_"):]
        mapping = json.loads(mapping_str)
        return _color_map_action(cropped, {int(k): int(v) for k, v in mapping.items()})
    return cropped


def search_program(task: dict) -> dict[str, Any] | None:
    """Search for a crop/object-selector program covering all train cases."""
    train = task.get("train", [])
    if len(train) < 2:
        return None

    inputs = [c["input"] for c in train]
    outputs = [c["output"] for c in train]

    # Strategy 1: crop non-background → identity
    # Output must equal cropped non-background region
    all_match = True
    for inp, out in zip(inputs, outputs):
        pixels = select_non_background(inp)
        if not pixels:
            all_match = False
            break
        bbox = _bbox_of(pixels)
        cropped = _crop_grid(inp, bbox)
        if cropped != out:
            all_match = False
            break
    if all_match:
        return {
            "program": "crop_non_background",
            "selector_type": "non_background",
            "action_type": "identity",
            "train_pass": f"{len(train)}/{len(train)}",
        }

    # Strategy 2: crop color(c) → identity
    for color in range(1, 10):
        all_match = True
        for inp, out in zip(inputs, outputs):
            pixels = select_color(inp, color)
            if not pixels:
                all_match = False
                break
            bbox = _bbox_of(pixels)
            cropped = _crop_grid(inp, bbox)
            if cropped != out:
                all_match = False
                break
        if all_match:
            return {
                "program": f"crop_color({color})",
                "selector_type": f"color_{color}",
                "action_type": "identity",
                "train_pass": f"{len(train)}/{len(train)}",
            }

    # Strategy 3: largest object → crop → identity
    all_match = True
    for inp, out in zip(inputs, outputs):
        obj = select_largest_object(inp)
        if obj is None:
            all_match = False
            break
        bbox = _bbox_of(obj)
        cropped = _crop_grid(inp, bbox)
        if cropped != out:
            all_match = False
            break
    if all_match:
        return {
            "program": "largest_object_crop",
            "selector_type": "largest_object",
            "action_type": "identity",
            "train_pass": f"{len(train)}/{len(train)}",
        }

    # Strategy 4: crop non-bg → color_map
    all_match = True
    mapping = None
    for inp, out in zip(inputs, outputs):
        pixels = select_non_background(inp)
        if not pixels:
            all_match = False
            break
        bbox = _bbox_of(pixels)
        cropped = _crop_grid(inp, bbox)
        cm = _infer_color_map(cropped, out)
        if cm is None:
            all_match = False
            break
        if mapping is None:
            mapping = cm
        elif cm != mapping:
            all_match = False
            break
        if _color_map_action(cropped, mapping) != out:
            all_match = False
            break
    if all_match and mapping is not None:
        return {
            "program": f"crop_non_bg_colormap({mapping})",
            "selector_type": "non_background",
            "action_type": f"color_map_{json.dumps(mapping)}",
            "train_pass": f"{len(train)}/{len(train)}",
        }

    # Strategy 5: crop not-touching-border → identity
    all_match = True
    for inp, out in zip(inputs, outputs):
        pixels = select_not_touching_border(inp)
        if not pixels:
            all_match = False
            break
        bbox = _bbox_of(pixels)
        cropped = _crop_grid(inp, bbox)
        if cropped != out:
            all_match = False
            break
    if all_match:
        return {
            "program": "crop_not_touching_border",
            "selector_type": "not_touching_border",
            "action_type": "identity",
            "train_pass": f"{len(train)}/{len(train)}",
        }

    # Strategy 6: crop inside_frame(c) → identity
    for fc in range(1, 10):
        all_match = True
        for inp, out in zip(inputs, outputs):
            interior = select_inside_frame(inp, fc)
            if interior is None or not interior:
                all_match = False
                break
            bbox = _bbox_of(interior)
            cropped = _crop_grid(inp, bbox)
            if cropped != out:
                all_match = False
                break
        if all_match:
            return {
                "program": f"crop_inside_frame({fc})",
                "selector_type": f"inside_frame_{fc}",
                "action_type": "identity",
                "train_pass": f"{len(train)}/{len(train)}",
            }

    # Strategy 7: crop non-bg → horizontal_mirror
    all_match = True
    for inp, out in zip(inputs, outputs):
        pixels = select_non_background(inp)
        if not pixels:
            all_match = False
            break
        bbox = _bbox_of(pixels)
        cropped = _crop_grid(inp, bbox)
        mirrored = _horizontal_mirror_action(cropped)
        if mirrored != out:
            all_match = False
            break
    if all_match:
        return {
            "program": "crop_non_bg_h_mirror",
            "selector_type": "non_background",
            "action_type": "horizontal_mirror",
            "train_pass": f"{len(train)}/{len(train)}",
        }

    # Strategy 8: crop non-bg → vertical_mirror
    all_match = True
    for inp, out in zip(inputs, outputs):
        pixels = select_non_background(inp)
        if not pixels:
            all_match = False
            break
        bbox = _bbox_of(pixels)
        cropped = _crop_grid(inp, bbox)
        mirrored = _vertical_mirror_action(cropped)
        if mirrored != out:
            all_match = False
            break
    if all_match:
        return {
            "program": "crop_non_bg_v_mirror",
            "selector_type": "non_background",
            "action_type": "vertical_mirror",
            "train_pass": f"{len(train)}/{len(train)}",
        }

    return None


# ---------------------------------------------------------------------------
# ONNX Builders
# ---------------------------------------------------------------------------

def _build_color_mask_conv(color: int, name: str, input_name: str = "input") -> tuple[list[onnx.NodeProto], list[onnx.TensorProto]]:
    """Build Conv+Greater mask detecting a specific color in one-hot 10-channel input.

    Uses: Conv with weight 1.0 at channel C → pixel value at that channel (0 or 1 for one-hot).
    Then Greater(0.5) → Cast(float32) → mask is 1.0 exactly where channel C is 1.0.
    """
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    weight = np.zeros((1, 10, 1, 1), dtype=np.float32)
    weight[0, color, 0, 0] = 1.0
    bias = np.zeros(1, dtype=np.float32)
    inits.append(numpy_helper.from_array(weight, name=f"{name}_w"))
    inits.append(numpy_helper.from_array(bias, name=f"{name}_b"))

    nodes.append(helper.make_node(
        "Conv", [input_name, f"{name}_w", f"{name}_b"], [f"{name}_conv"],
        name=name, kernel_shape=[1, 1]
    ))
    half = np.array([0.5], dtype=np.float32)
    inits.append(numpy_helper.from_array(half, name=f"{name}_half"))
    nodes.append(helper.make_node(
        "Greater", [f"{name}_conv", f"{name}_half"], [f"{name}_gt"], name=f"{name}_gt"
    ))
    nodes.append(helper.make_node(
        "Cast", [f"{name}_gt"], [f"{name}_mask"], name=f"{name}_cast", to=TensorProto.FLOAT
    ))
    return nodes, inits


def _build_crop_model(
    output_path: str,
    program: dict,
    task: dict,
) -> None:
    """Build a static crop ONNX model using Slice + Pad.

    Uses the bbox from the first train case as a fixed crop region.
    Only called when bbox is constant across all train cases.
    """
    train = task.get("train", [])
    first_inp = train[0]["input"]

    # Determine bbox from first case based on selector
    selector_type = program.get("selector_type", "")
    pixels = _get_pixels_for_selector(first_inp, selector_type)

    if not pixels:
        raise ValueError(f"no pixels found for selector {selector_type}")

    r1, c1, r2, c2 = _bbox_of(pixels)
    crop_h, crop_w = r2 - r1, c2 - c1

    action_type = program.get("action_type", "identity")

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = []

    # Slice input to bbox
    starts = np.array([0, 0, r1, c1], dtype=np.int64)
    ends = np.array([1, 10, r2, c2], dtype=np.int64)
    axes = np.array([0, 1, 2, 3], dtype=np.int64)
    steps = np.array([1, 1, 1, 1], dtype=np.int64)
    initializers.extend([
        numpy_helper.from_array(starts, name="crop_starts"),
        numpy_helper.from_array(ends, name="crop_ends"),
        numpy_helper.from_array(axes, name="crop_axes"),
        numpy_helper.from_array(steps, name="crop_steps"),
    ])
    nodes.append(helper.make_node(
        "Slice", ["input", "crop_starts", "crop_ends", "crop_axes", "crop_steps"],
        ["cropped"], name="crop_slice"
    ))

    current = "cropped"

    # Apply transform
    if action_type == "horizontal_mirror":
        h_idx = np.arange(crop_w - 1, -1, -1, dtype=np.int64)
        h_init = numpy_helper.from_array(h_idx, name="h_mirror_idx")
        h_axis = np.array([3], dtype=np.int64)
        initializers.extend([h_init, numpy_helper.from_array(h_axis, name="h_mirror_axis")])
        nodes.append(helper.make_node(
            "Gather", ["cropped", "h_mirror_idx", "h_mirror_axis"],
            ["mirrored"], name="h_mirror", axis=3
        ))
        current = "mirrored"
    elif action_type == "vertical_mirror":
        v_idx = np.arange(crop_h - 1, -1, -1, dtype=np.int64)
        v_init = numpy_helper.from_array(v_idx, name="v_mirror_idx")
        v_axis = np.array([2], dtype=np.int64)
        initializers.extend([v_init, numpy_helper.from_array(v_axis, name="v_mirror_axis")])
        nodes.append(helper.make_node(
            "Gather", ["cropped", "v_mirror_idx", "v_mirror_axis"],
            ["mirrored"], name="v_mirror", axis=2
        ))
        current = "mirrored"
    elif action_type.startswith("color_map_"):
        mapping_str = action_type[len("color_map_"):]
        mapping = json.loads(mapping_str)
        for src_c, dst_c in mapping.items():
            src_c = int(src_c)
            dst_c = int(dst_c)
            mask_name = f"cm_mask_{src_c}"
            m_nodes, m_inits = _build_color_mask_conv(src_c, mask_name, input_name=current)
            initializers.extend(m_inits)
            nodes.extend(m_nodes)
            fill_val = np.full((1, 1, crop_h, crop_w), float(dst_c), dtype=np.float32)
            fill_init = numpy_helper.from_array(fill_val, name=f"cm_fill_{src_c}")
            initializers.append(fill_init)
            after_map = f"cm_out_{src_c}"
            nodes.append(helper.make_node(
                "Where", [f"{mask_name}_mask", f"cm_fill_{src_c}", current],
                [after_map], name=f"cm_where_{src_c}"
            ))
            current = after_map

    # Always pad back to 30x30 (ONNX output must be DEFAULT_SHAPE)
    target_h, target_w = 30, 30
    pad_top = r1
    pad_left = c1
    pad_bottom = target_h - (r2 - r1) - pad_top
    pad_right = target_w - (c2 - c1) - pad_left

    # Clamp negative pads to 0 (crop won't exceed 30x30 for these tasks)
    pad_top = max(0, pad_top)
    pad_left = max(0, pad_left)
    pad_bottom = max(0, pad_bottom)
    pad_right = max(0, pad_right)

    pads_arr = np.array([0, 0, pad_top, pad_left, 0, 0, pad_bottom, pad_right], dtype=np.int64)
    initializers.append(numpy_helper.from_array(pads_arr, name="pads"))
    zero_const = np.zeros(1, dtype=np.float32)
    initializers.append(numpy_helper.from_array(zero_const, name="pad_zero"))
    nodes.append(helper.make_node(
        "Pad", [current, "pads", "pad_zero"],
        ["output"], name="pad_back", mode="constant"
    ))

    graph = helper.make_graph(
        nodes=nodes, name="object_crop_v2",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, DEFAULT_SHAPE)],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, DEFAULT_SHAPE)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph, producer_name="neurogolf-2026",
        ir_version=10, opset_imports=[helper.make_opsetid("", 11)]
    )
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    onnx.checker.check_model(str(out_path))


# ---------------------------------------------------------------------------
# Batch search
# ---------------------------------------------------------------------------

def batch_search(
    task_dir: str = "task",
    taxonomy_path: str = "outputs/reports/task_family_taxonomy_v2.csv",
    cost_report_path: str = "outputs/reports/current_model_bank_report.csv",
    candidate_dir: str = DEFAULT_CANDIDATE_DIR,
    report_path: str = DEFAULT_REPORT,
    min_cost: int = 1000,
    target_families: str = "",
) -> dict[str, Any]:
    cost_data: dict[str, int] = {}
    with Path(cost_report_path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("valid", "True").strip().lower() == "true":
                cost_data[row["task_id"].strip()] = int(row.get("estimated_cost") or 0)

    # Build taxonomy lookup
    taxonomy: dict[str, str] = {}
    if Path(taxonomy_path).exists():
        with Path(taxonomy_path).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                taxonomy[row["task_id"].strip()] = row.get("family", "")

    family_filter = {f.strip() for f in target_families.split(",") if f.strip()}

    rows = []
    searched = 0
    matched = 0
    valid_count = 0

    for tid in sorted(cost_data.keys()):
        current_cost = cost_data.get(tid, 0)
        if current_cost < min_cost:
            continue
        task_family = taxonomy.get(tid, "")
        if family_filter and task_family not in family_filter:
            continue
        task_path = Path(task_dir) / f"{tid}.json"
        if not task_path.exists():
            continue
        try:
            task = load_task(str(task_path))
        except Exception:
            continue
        searched += 1

        program = search_program(task)
        if program is None:
            continue
        matched += 1

        selector_type = program.get("selector_type", "")
        action_type = program.get("action_type", "")

        row = {
            "task_id": tid,
            "program": program.get("program", ""),
            "selector_type": selector_type,
            "action_type": action_type,
            "train_pass": program.get("train_pass", ""),
            "test_pass": "",
            "arc_gen_pass": "",
            "source_cost": current_cost,
            "output_cost": 0,
            "cost_delta": 0,
            "valid": False,
            "failure_reason": "",
            "family": task_family,
        }

        # Check if bbox is constant across all train cases
        train_cases = task.get("train", [])
        bboxes = []
        for case in train_cases:
            pixels = _get_pixels_for_selector(case["input"], selector_type)
            if pixels:
                bboxes.append(_bbox_of(pixels))
        unique_bboxes = set(bboxes)

        if len(unique_bboxes) > 1:
            row["failure_reason"] = f"dynamic_bbox: {len(unique_bboxes)} unique bboxes across train cases"
            rows.append(row)
            continue

        # Build ONNX candidate (static bbox)
        output_name = f"{tid}_object_crop_v2.onnx"
        output_path = str(Path(candidate_dir) / output_name)
        try:
            _build_crop_model(output_path, program, task)
            output_info = estimate_model_cost(output_path)
            row["output_cost"] = int(output_info["estimated_cost"])
            row["cost_delta"] = int(output_info["estimated_cost"]) - current_cost

            try:
                onnx.checker.check_model(output_path)
                if not check_forbidden_ops(output_path)["passed"]:
                    row["failure_reason"] = "forbidden ops"
                elif not check_static_shapes(output_path)["passed"]:
                    row["failure_reason"] = "dynamic shapes"
                elif not output_info["file_size_ok"]:
                    row["failure_reason"] = "file size exceeded"
                else:
                    val_result = validate_labelled_splits(
                        output_path, str(task_path),
                        str(Path(candidate_dir) / f"{tid}_validation.csv")
                    )
                    row["valid"] = val_result.get("passed", False)
                    sc = val_result.get("split_counts", {})
                    row["test_pass"] = str(sc.get("test", {}).get("passed", "?"))
                    row["arc_gen_pass"] = str(sc.get("arc-gen", {}).get("passed", "?"))
                    if not row["valid"]:
                        row["failure_reason"] = (
                            f"labelled: {val_result.get('passed_cases', 0)}/"
                            f"{val_result.get('total_cases', 0)}"
                        )
            except Exception as exc:
                row["failure_reason"] = str(exc)[:200]
        except Exception as exc:
            row["failure_reason"] = f"build: {exc}"[:200]

        if row["valid"]:
            valid_count += 1
        rows.append(row)

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "report_path": str(report),
        "searched": searched,
        "programs_matched": matched,
        "valid_candidates": valid_count,
        "total_cost_delta": sum(r["cost_delta"] for r in rows if r["valid"]),
        "by_selector": {},
        "by_family": {},
    }
    for r in rows:
        s = r["selector_type"]
        if s not in summary["by_selector"]:
            summary["by_selector"][s] = {"total": 0, "valid": 0}
        summary["by_selector"][s]["total"] += 1
        if r["valid"]:
            summary["by_selector"][s]["valid"] += 1
        f = r["family"]
        if f not in summary["by_family"]:
            summary["by_family"][f] = {"total": 0, "valid": 0}
        summary["by_family"][f]["total"] += 1
        if r["valid"]:
            summary["by_family"][f]["valid"] += 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", default="task")
    parser.add_argument("--taxonomy", default="outputs/reports/task_family_taxonomy_v2.csv")
    parser.add_argument("--cost-report", default="outputs/reports/current_model_bank_report.csv")
    parser.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--min-cost", type=int, default=1000)
    parser.add_argument(
        "--target-families",
        default="",
        help="Comma-separated family letters: B,H,K (empty = all)",
    )
    args = parser.parse_args()
    batch_search(
        task_dir=args.task_dir, taxonomy_path=args.taxonomy,
        cost_report_path=args.cost_report,
        candidate_dir=args.candidate_dir, report_path=args.report,
        min_cost=args.min_cost, target_families=args.target_families,
    )


if __name__ == "__main__":
    main()
