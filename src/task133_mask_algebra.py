"""Probe and build the task133 same-shape mask-algebra candidate."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .encoding import DEFAULT_COLORS, DEFAULT_HEIGHT, DEFAULT_WIDTH


Grid = list[list[int]]

TASK_ID = "task133"
TEMPLATE_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (-1, 3),
    (0, -2),
    (0, -1),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, -1),
    (1, 0),
    (1, 1),
    (1, 2),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
]
SEED_OFFSETS = [(-1, 0), (0, -1), (0, 1), (1, 0)]
MAX_TARGET_BLOCK_SIZE = 4

REPORT_FIELDS = [
    "task_id",
    "formula",
    "train_pass",
    "test_pass",
    "arc_gen_pass",
    "num_conditions",
    "num_colors_used",
    "risk_score",
    "builder_candidate_path",
    "failure_reason",
]


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0])


def _components_nonzero(grid: Grid) -> list[list[tuple[int, int, int]]]:
    height, width = _shape(grid)
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int, int]]] = []
    for row in range(height):
        for col in range(width):
            if (row, col) in seen or grid[row][col] == 0:
                continue
            queue = deque([(row, col)])
            seen.add((row, col))
            component: list[tuple[int, int, int]] = []
            while queue:
                current_row, current_col = queue.popleft()
                component.append((current_row, current_col, grid[current_row][current_col]))
                for row_delta, col_delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    next_row = current_row + row_delta
                    next_col = current_col + col_delta
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and (next_row, next_col) not in seen
                        and grid[next_row][next_col] != 0
                    ):
                        seen.add((next_row, next_col))
                        queue.append((next_row, next_col))
            components.append(component)
    return components


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [row for row, _ in cells]
    cols = [col for _, col in cells]
    return min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _solid_rect(grid: Grid, cells: list[tuple[int, int]], color: int) -> bool:
    top, left, height, width = _bbox(cells)
    return len(cells) == height * width and all(
        grid[row][col] == color
        for row in range(top, top + height)
        for col in range(left, left + width)
    )


def _template_offsets(
    grid: Grid,
    component: list[tuple[int, int, int]],
    marker_color: int,
    pattern_color: int,
) -> tuple[tuple[int, int], ...] | None:
    marker_cells = [(row, col) for row, col, color in component if color == marker_color]
    pattern_cells = [(row, col) for row, col, color in component if color == pattern_color]
    if not marker_cells or not pattern_cells or len(marker_cells) >= len(pattern_cells):
        return None
    marker_top, marker_left, marker_height, marker_width = _bbox(marker_cells)
    if (marker_height, marker_width) != (1, 1):
        return None
    if not _solid_rect(grid, marker_cells, marker_color):
        return None
    offsets = tuple(sorted((row - marker_top, col - marker_left) for row, col in pattern_cells))
    if not offsets or (0, 0) in offsets:
        return None
    return offsets


def _infer_template(grid: Grid) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    for component in _components_nonzero(grid):
        colors = sorted({color for _, _, color in component})
        if len(colors) != 2:
            continue
        for marker_color in colors:
            pattern_color = colors[1] if colors[0] == marker_color else colors[0]
            offsets = _template_offsets(grid, component, marker_color, pattern_color)
            if offsets is None:
                continue
            candidate = {
                "marker_color": marker_color,
                "pattern_color": pattern_color,
                "offsets": offsets,
            }
            if best is None or len(offsets) > len(best["offsets"]):
                best = candidate
    return best


def _color_block_offsets(
    cells: list[tuple[int, int]],
    block_height: int,
    block_width: int,
    marker_top: int,
    marker_left: int,
) -> set[tuple[int, int]] | None:
    remaining = set(cells)
    offsets: set[tuple[int, int]] = set()
    while remaining:
        start = next(iter(remaining))
        queue = deque([start])
        remaining.remove(start)
        component: list[tuple[int, int]] = []
        while queue:
            current_row, current_col = queue.popleft()
            component.append((current_row, current_col))
            for row_delta, col_delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                neighbor = (current_row + row_delta, current_col + col_delta)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        top, left, height, width = _bbox(component)
        if height != block_height or width != block_width or len(component) != block_height * block_width:
            return None
        if (top - marker_top) % block_height or (left - marker_left) % block_width:
            return None
        offsets.add(((top - marker_top) // block_height, (left - marker_left) // block_width))
    return offsets


def task133_mask_algebra_transform(grid: Grid) -> Grid:
    """Apply the task133 template-offset expansion rule in Python."""
    template = _infer_template(grid)
    if template is None:
        return [row[:] for row in grid]

    marker_color = int(template["marker_color"])
    pattern_color = int(template["pattern_color"])
    template_offsets = set(template["offsets"])
    height, width = _shape(grid)
    output = [row[:] for row in grid]

    for component in _components_nonzero(grid):
        colors = sorted({color for _, _, color in component})
        if len(colors) != 2 or marker_color not in colors:
            continue
        fill_color = colors[1] if colors[0] == marker_color else colors[0]
        if fill_color == pattern_color:
            continue
        marker_cells = [(row, col) for row, col, color in component if color == marker_color]
        fill_cells = [(row, col) for row, col, color in component if color == fill_color]
        if not marker_cells or not fill_cells:
            continue
        marker_top, marker_left, block_height, block_width = _bbox(marker_cells)
        if block_height != block_width or block_height > MAX_TARGET_BLOCK_SIZE:
            continue
        if not _solid_rect(grid, marker_cells, marker_color):
            continue
        seed_offsets = _color_block_offsets(fill_cells, block_height, block_width, marker_top, marker_left)
        if seed_offsets is None or not (seed_offsets & template_offsets):
            continue
        for offset_row, offset_col in template_offsets:
            top = marker_top + offset_row * block_height
            left = marker_left + offset_col * block_width
            if top < 0 or left < 0 or top + block_height > height or left + block_width > width:
                continue
            for row in range(top, top + block_height):
                for col in range(left, left + block_width):
                    if output[row][col] in {0, fill_color}:
                        output[row][col] = fill_color
    return output


def _score_cases(cases: list[dict[str, Any]]) -> tuple[int, int, str]:
    passed = 0
    first_failure = ""
    total = 0
    for index, case in enumerate(cases):
        if "output" not in case:
            continue
        total += 1
        prediction = task133_mask_algebra_transform(case["input"])
        if prediction == case["output"]:
            passed += 1
        elif not first_failure:
            first_failure = f"case {index} mismatch"
    return passed, total, first_failure


def probe_task(task: dict[str, Any], candidate_path: str = "") -> dict[str, Any]:
    """Return a report row for the task133 same-shape mask-algebra probe."""
    train_passed, train_total, train_failure = _score_cases(task.get("train", []))
    test_passed, test_total, test_failure = _score_cases(task.get("test", []))
    arc_passed, arc_total, arc_failure = _score_cases(task.get("arc-gen", []))
    colors: set[int] = set()
    for split in ("train", "test", "arc-gen"):
        for case in task.get(split, []):
            for row in case.get("input", []):
                colors.update(row)
    failures = [item for item in (train_failure, test_failure, arc_failure) if item]
    return {
        "task_id": TASK_ID,
        "formula": (
            "Infer the unique two-color template with an isolated marker cell; "
            "copy each template offset to same-marker square target blocks using the target seed color."
        ),
        "train_pass": f"{train_passed}/{train_total}",
        "test_pass": f"{test_passed}/{test_total}",
        "arc_gen_pass": f"{arc_passed}/{arc_total}",
        "num_conditions": 5,
        "num_colors_used": len(colors - {0}),
        "risk_score": "medium",
        "builder_candidate_path": candidate_path,
        "failure_reason": "; ".join(failures),
    }


def _value_info(name: str) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(
        name,
        TensorProto.FLOAT,
        [1, DEFAULT_COLORS, DEFAULT_HEIGHT, DEFAULT_WIDTH],
    )


def _scalar(value: float, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(value, dtype=np.float32), name=name)


def _one() -> np.ndarray:
    return np.ones((1, 1, 1, 1), dtype=np.float32)


def _clip01(nodes: list[onnx.NodeProto], input_name: str, output_name: str) -> None:
    nodes.append(helper.make_node("Clip", [input_name], [output_name], name=output_name, min=0.0, max=1.0))


def _cast_float(input_name: str, output_name: str) -> onnx.NodeProto:
    return helper.make_node("Cast", [input_name], [output_name], name=output_name, to=TensorProto.FLOAT)


def _float_equal_bool(
    nodes: list[onnx.NodeProto],
    input_name: str,
    target_name: str,
    output_name: str,
) -> None:
    diff_name = f"{output_name}_diff"
    abs_name = f"{output_name}_abs"
    nodes.extend(
        [
            helper.make_node("Sub", [input_name, target_name], [diff_name], name=diff_name),
            helper.make_node("Abs", [diff_name], [abs_name], name=abs_name),
            helper.make_node("Less", [abs_name, "Half"], [output_name], name=output_name),
        ]
    )


def _add_many(nodes: list[onnx.NodeProto], names: list[str], output_name: str) -> str:
    if not names:
        raise ValueError("names must not be empty")
    if len(names) == 1:
        nodes.append(helper.make_node("Identity", [names[0]], [output_name], name=output_name))
        return output_name
    current = names[0]
    for index, name in enumerate(names[1:], start=1):
        next_name = output_name if index == len(names) - 1 else f"{output_name}_{index}"
        nodes.append(helper.make_node("Add", [current, name], [next_name], name=next_name))
        current = next_name
    return current


def _conv_weight_for_color(color: int, kernel_height: int, kernel_width: int, name: str) -> onnx.TensorProto:
    weights = np.zeros((1, DEFAULT_COLORS, kernel_height, kernel_width), dtype=np.float32)
    weights[0, color, :, :] = 1.0
    return numpy_helper.from_array(weights, name=name)


def _template_offset_weight(marker_color: int, offset: tuple[int, int], name: str) -> onnx.TensorProto:
    weights = np.zeros((1, DEFAULT_COLORS, 4, 6), dtype=np.float32)
    row, col = offset
    kernel_row = row + 1
    kernel_col = col + 2
    for color in range(1, DEFAULT_COLORS):
        if color != marker_color:
            weights[0, color, kernel_row, kernel_col] = 1.0
    return numpy_helper.from_array(weights, name=name)


def _template_count_weight(marker_color: int, name: str) -> onnx.TensorProto:
    weights = np.zeros((1, DEFAULT_COLORS, 4, 6), dtype=np.float32)
    for row, col in TEMPLATE_OFFSETS:
        for color in range(1, DEFAULT_COLORS):
            if color != marker_color:
                weights[0, color, row + 1, col + 2] = 1.0
    return numpy_helper.from_array(weights, name=name)


def _color_block_weight(size: int, name: str) -> onnx.TensorProto:
    weights = np.zeros((DEFAULT_COLORS, DEFAULT_COLORS, size, size), dtype=np.float32)
    for color in range(DEFAULT_COLORS):
        weights[color, color, :, :] = 1.0
    return numpy_helper.from_array(weights, name=name)


def _allowed_fill_colors(marker_color: int, name: str) -> onnx.TensorProto:
    values = np.ones((1, DEFAULT_COLORS, 1, 1), dtype=np.float32)
    values[:, 0, :, :] = 0.0
    values[:, marker_color, :, :] = 0.0
    return numpy_helper.from_array(values, name=name)


def _group_shift_weight(
    relative_rows: list[int],
    relative_cols: list[int],
    name: str,
) -> tuple[onnx.TensorProto, list[int]]:
    min_row = min(relative_rows)
    max_row = max(relative_rows)
    min_col = min(relative_cols)
    max_col = max(relative_cols)
    pad_top = max(-min_row, 0)
    pad_bottom = max(max_row, 0)
    pad_left = max(-min_col, 0)
    pad_right = max(max_col, 0)
    kernel_height = pad_top + pad_bottom + 1
    kernel_width = pad_left + pad_right + 1
    weights = np.zeros((DEFAULT_COLORS, 1, kernel_height, kernel_width), dtype=np.float32)
    for rel_row, rel_col in zip(relative_rows, relative_cols):
        weights[:, 0, pad_top + rel_row, pad_left + rel_col] = 1.0
    return numpy_helper.from_array(weights, name=name), [pad_top, pad_left, pad_bottom, pad_right]


def _group_shift_conv(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    input_name: str,
    output_name: str,
    relative_positions: list[tuple[int, int]],
    weight_name: str,
) -> None:
    rows = [row for row, _ in relative_positions]
    cols = [col for _, col in relative_positions]
    weight, pads = _group_shift_weight(rows, cols, weight_name)
    initializers.append(weight)
    nodes.append(
        helper.make_node(
            "Conv",
            [input_name, weight_name],
            [output_name],
            name=output_name,
            group=DEFAULT_COLORS,
            kernel_shape=list(weight.dims[2:]),
            pads=pads,
        )
    )


def build_task133_mask_algebra_model(output_path: str) -> None:
    """Build a static ONNX graph for the task133 template-offset expansion DSL."""
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = [
        _scalar(1.0, "One"),
        _scalar(1.5, "OnePointFive"),
        _scalar(0.5, "Half"),
        numpy_helper.from_array(_one(), name="OneTensor"),
    ]

    color0_weights = np.zeros((1, DEFAULT_COLORS, 1, 1), dtype=np.float32)
    color0_weights[0, 0, 0, 0] = 1.0
    initializers.append(numpy_helper.from_array(color0_weights, name="Color0W"))
    nodes.append(helper.make_node("Conv", ["input", "Color0W"], ["color0_mask"], name="color0_mask", kernel_shape=[1, 1]))

    template_flags: dict[tuple[int, tuple[int, int]], str] = {}
    for marker_color in range(1, DEFAULT_COLORS):
        prefix = f"m{marker_color}"
        selector = np.zeros((1, DEFAULT_COLORS, 1, 1), dtype=np.float32)
        selector[0, marker_color, 0, 0] = 1.0
        initializers.append(numpy_helper.from_array(selector, name=f"{prefix}_SelectW"))
        initializers.append(_conv_weight_for_color(marker_color, 3, 3, f"{prefix}_NeighborW"))
        initializers.append(_template_count_weight(marker_color, f"{prefix}_TemplateCountW"))
        nodes.extend(
            [
                helper.make_node("Conv", ["input", f"{prefix}_SelectW"], [f"{prefix}_mask"], name=f"{prefix}_mask", kernel_shape=[1, 1]),
                helper.make_node(
                    "Conv",
                    ["input", f"{prefix}_NeighborW"],
                    [f"{prefix}_neighbor_count"],
                    name=f"{prefix}_neighbor_count",
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                ),
                helper.make_node(
                    "Conv",
                    ["input", f"{prefix}_TemplateCountW"],
                    [f"{prefix}_template_nonmarker_count"],
                    name=f"{prefix}_template_nonmarker_count",
                    kernel_shape=[4, 6],
                    pads=[1, 2, 2, 3],
                ),
                helper.make_node(
                    "Greater",
                    [f"{prefix}_template_nonmarker_count", "OnePointFive"],
                    [f"{prefix}_template_count_ok_bool"],
                    name=f"{prefix}_template_count_ok_bool",
                ),
                _cast_float(f"{prefix}_template_count_ok_bool", f"{prefix}_template_count_ok"),
            ]
        )
        _float_equal_bool(nodes, f"{prefix}_neighbor_count", "One", f"{prefix}_isolated_bool")
        nodes.append(_cast_float(f"{prefix}_isolated_bool", f"{prefix}_isolated"))
        nodes.extend(
            [
                helper.make_node("Mul", [f"{prefix}_mask", f"{prefix}_isolated"], [f"{prefix}_candidate_a"], name=f"{prefix}_candidate_a"),
                helper.make_node(
                    "Mul",
                    [f"{prefix}_candidate_a", f"{prefix}_template_count_ok"],
                    [f"{prefix}_template_candidate"],
                    name=f"{prefix}_template_candidate",
                ),
            ]
        )
        for offset_index, offset in enumerate(TEMPLATE_OFFSETS):
            offset_name = f"{prefix}_o{offset_index}"
            initializers.append(_template_offset_weight(marker_color, offset, f"{offset_name}_OffsetW"))
            nodes.extend(
                [
                    helper.make_node(
                        "Conv",
                        ["input", f"{offset_name}_OffsetW"],
                        [f"{offset_name}_offset_nonmarker"],
                        name=f"{offset_name}_offset_nonmarker",
                        kernel_shape=[4, 6],
                        pads=[1, 2, 2, 3],
                    ),
                    helper.make_node(
                        "Mul",
                        [f"{prefix}_template_candidate", f"{offset_name}_offset_nonmarker"],
                        [f"{offset_name}_hits"],
                        name=f"{offset_name}_hits",
                    ),
                    helper.make_node(
                        "ReduceSum",
                        [f"{offset_name}_hits"],
                        [f"{offset_name}_hit_sum"],
                        name=f"{offset_name}_hit_sum",
                        axes=[2, 3],
                        keepdims=1,
                    ),
                ]
            )
            _clip01(nodes, f"{offset_name}_hit_sum", f"{offset_name}_flag")
            template_flags[(marker_color, offset)] = f"{offset_name}_flag"

    fill_full_by_size: dict[int, str] = {}
    for size in range(1, MAX_TARGET_BLOCK_SIZE + 1):
        prefix = f"s{size}"
        initializers.append(_color_block_weight(size, f"{prefix}_ColorBlockW"))
        initializers.append(_scalar(float(size * size), f"FullCount_{size}"))
        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    ["input", f"{prefix}_ColorBlockW"],
                    [f"{prefix}_color_block_count"],
                    name=f"{prefix}_color_block_count",
                    kernel_shape=[size, size],
                    pads=[0, 0, size - 1, size - 1],
                ),
            ]
        )
        _float_equal_bool(nodes, f"{prefix}_color_block_count", f"FullCount_{size}", f"{prefix}_color_block_full_bool")
        nodes.append(_cast_float(f"{prefix}_color_block_full_bool", f"{prefix}_color_block_full"))
        fill_full_by_size[size] = f"{prefix}_color_block_full"

    generated_terms: list[str] = []
    for marker_color in range(1, DEFAULT_COLORS):
        marker_prefix = f"m{marker_color}"
        initializers.append(_allowed_fill_colors(marker_color, f"{marker_prefix}_AllowedFillColors"))
        for size in range(1, MAX_TARGET_BLOCK_SIZE + 1):
            prefix = f"{marker_prefix}_s{size}"
            initializers.append(_conv_weight_for_color(marker_color, size, size, f"{prefix}_FullW"))
            initializers.append(_conv_weight_for_color(marker_color, size + 2, size + 2, f"{prefix}_ExpandedW"))
            nodes.extend(
                [
                    helper.make_node(
                        "Conv",
                        ["input", f"{prefix}_FullW"],
                        [f"{prefix}_full_count"],
                        name=f"{prefix}_full_count",
                        kernel_shape=[size, size],
                        pads=[0, 0, size - 1, size - 1],
                    ),
                    helper.make_node(
                        "Conv",
                        ["input", f"{prefix}_ExpandedW"],
                        [f"{prefix}_expanded_count"],
                        name=f"{prefix}_expanded_count",
                        kernel_shape=[size + 2, size + 2],
                        pads=[1, 1, size, size],
                    ),
                ]
            )
            _float_equal_bool(nodes, f"{prefix}_full_count", f"FullCount_{size}", f"{prefix}_full_bool")
            _float_equal_bool(nodes, f"{prefix}_expanded_count", f"FullCount_{size}", f"{prefix}_exact_bool")
            nodes.extend(
                [
                    helper.make_node("And", [f"{prefix}_full_bool", f"{prefix}_exact_bool"], [f"{prefix}_marker_bool"], name=f"{prefix}_marker_bool"),
                    _cast_float(f"{prefix}_marker_bool", f"{prefix}_marker"),
                ]
            )

            seed_terms: list[str] = []
            for seed_index, seed_offset in enumerate(SEED_OFFSETS):
                seed_prefix = f"{prefix}_seed{seed_index}"
                rel_row = seed_offset[0] * size
                rel_col = seed_offset[1] * size
                _group_shift_conv(
                    nodes,
                    initializers,
                    fill_full_by_size[size],
                    f"{seed_prefix}_fill_at_marker",
                    [(rel_row, rel_col)],
                    f"{seed_prefix}_ShiftW",
                )
                flag_name = template_flags[(marker_color, seed_offset)]
                nodes.extend(
                    [
                        helper.make_node(
                            "Mul",
                            [f"{seed_prefix}_fill_at_marker", f"{marker_prefix}_AllowedFillColors"],
                            [f"{seed_prefix}_allowed_fill"],
                            name=f"{seed_prefix}_allowed_fill",
                        ),
                        helper.make_node(
                            "Mul",
                            [f"{seed_prefix}_allowed_fill", f"{prefix}_marker"],
                            [f"{seed_prefix}_on_marker"],
                            name=f"{seed_prefix}_on_marker",
                        ),
                        helper.make_node(
                            "Mul",
                            [f"{seed_prefix}_on_marker", flag_name],
                            [f"{seed_prefix}_target"],
                            name=f"{seed_prefix}_target",
                        ),
                    ]
                )
                seed_terms.append(f"{seed_prefix}_target")
            seed_sum = _add_many(nodes, seed_terms, f"{prefix}_target_seed_sum")
            _clip01(nodes, seed_sum, f"{prefix}_target_seed")

            for offset_index, offset in enumerate(TEMPLATE_OFFSETS):
                expansion_prefix = f"{prefix}_expand{offset_index}"
                relative_positions = [
                    (-(offset[0] * size + row_delta), -(offset[1] * size + col_delta))
                    for row_delta in range(size)
                    for col_delta in range(size)
                ]
                _group_shift_conv(
                    nodes,
                    initializers,
                    f"{prefix}_target_seed",
                    f"{expansion_prefix}_raw",
                    relative_positions,
                    f"{expansion_prefix}_ExpandW",
                )
                flag_name = template_flags[(marker_color, offset)]
                nodes.append(
                    helper.make_node(
                        "Mul",
                        [f"{expansion_prefix}_raw", flag_name],
                        [f"{expansion_prefix}_flagged"],
                        name=f"{expansion_prefix}_flagged",
                    )
                )
                generated_terms.append(f"{expansion_prefix}_flagged")

    generated_sum = _add_many(nodes, generated_terms, "generated_sum")
    _clip01(nodes, generated_sum, "generated_pre_allowed")
    nodes.extend(
        [
            helper.make_node("Add", ["input", "color0_mask"], ["overwrite_allowed_raw"], name="overwrite_allowed_raw"),
        ]
    )
    _clip01(nodes, "overwrite_allowed_raw", "overwrite_allowed")
    nodes.extend(
        [
            helper.make_node("Mul", ["generated_pre_allowed", "overwrite_allowed"], ["generated"], name="generated"),
            helper.make_node("ReduceSum", ["generated"], ["generated_any_raw"], name="generated_any_raw", axes=[1], keepdims=1),
        ]
    )
    _clip01(nodes, "generated_any_raw", "generated_any")
    nodes.extend(
        [
            helper.make_node("Sub", ["OneTensor", "generated_any"], ["keep_mask"], name="keep_mask"),
            helper.make_node("Mul", ["input", "keep_mask"], ["kept_input"], name="kept_input"),
            helper.make_node("Add", ["kept_input", "generated"], ["output"], name="output"),
        ]
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        nodes=nodes,
        name="task133_mask_algebra",
        inputs=[_value_info("input")],
        outputs=[_value_info("output")],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-2026",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(output))
    onnx.checker.check_model(str(output))


def write_probe_report(task: dict[str, Any], report_path: str, candidate_path: str = "") -> dict[str, Any]:
    row = probe_task(task, candidate_path)
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-path", default="task/task133.json")
    parser.add_argument("--candidate", default="")
    parser.add_argument("--report", default="outputs/reports/task133_same_shape_mask_algebra_probe.csv")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    task = json.loads(Path(args.task_path).read_text(encoding="utf-8"))
    if args.build:
        if not args.candidate:
            raise ValueError("--candidate is required with --build")
        build_task133_mask_algebra_model(args.candidate)
    row = write_probe_report(task, args.report, args.candidate)
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
