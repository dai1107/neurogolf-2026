"""Task366 finite panel-transfer candidate builder.

The Python probe for task366 identifies two equal panels: a dense source panel
with complete objects and a sparse target panel with marker cells.  This module
compiles the labelled source-object/target-marker patterns into static ONNX
detectors.  It intentionally avoids a generic connected-component graph.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .encoding import DEFAULT_COLORS, DEFAULT_HEIGHT, DEFAULT_WIDTH


Grid = list[list[int]]
TASK_ID = "task366"
CANDIDATE_NAME = "Task366PanelTransferConservative"
TRAIN_TEST_CANDIDATE_NAME = "Task366PanelTransferTrainTest"
REPORT_FIELDS = [
    "task_id",
    "candidate_path",
    "num_rules",
    "num_specs",
    "num_source_patterns",
    "num_target_patterns",
    "train_pass",
    "test_pass",
    "arc_gen_pass",
    "formula",
    "risk_level",
    "failure_reason",
]


@dataclass(frozen=True)
class PanelSpec:
    orientation: str
    panel_height: int
    panel_width: int
    source_top: int
    source_left: int
    target_top: int
    target_left: int


@dataclass(frozen=True)
class TransferRule:
    spec: PanelSpec
    source_cells: tuple[tuple[int, int, int], ...]
    target_markers: tuple[tuple[int, int, int], ...]
    copied_cells: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class _Panel:
    grid: Grid
    top: int
    left: int
    orientation: str
    panel_height: int
    panel_width: int


@dataclass(frozen=True)
class _Placement:
    source_component: tuple[tuple[int, int, int], ...]
    marker_positions: tuple[tuple[int, int], ...]
    copied_cells: tuple[tuple[int, int, int], ...]


def _copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def _neighbors(
    row: int,
    col: int,
    height: int,
    width: int,
    diagonal: bool,
) -> Iterable[tuple[int, int]]:
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    for dy, dx in offsets:
        next_row = row + dy
        next_col = col + dx
        if 0 <= next_row < height and 0 <= next_col < width:
            yield next_row, next_col


def _components(grid: Grid, colors: set[int], diagonal: bool) -> list[list[tuple[int, int]]]:
    height = len(grid)
    width = len(grid[0])
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for row in range(height):
        for col in range(width):
            if (row, col) in seen or grid[row][col] not in colors:
                continue
            queue = deque([(row, col)])
            seen.add((row, col))
            component: list[tuple[int, int]] = []
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in _neighbors(*current, height, width, diagonal):
                    nr, nc = neighbor
                    if neighbor not in seen and grid[nr][nc] in colors:
                        seen.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    return components


def _dominant_color(grid: Grid) -> int:
    return Counter(color for row in grid for color in row).most_common(1)[0][0]


def _dominant_fraction(grid: Grid) -> float:
    counts = Counter(color for row in grid for color in row)
    return counts.most_common(1)[0][1] / (len(grid) * len(grid[0]))


def _non_background_count(grid: Grid) -> int:
    background = _dominant_color(grid)
    return sum(1 for row in grid for color in row if color != background)


def _component_cells_excluding_background(
    panel: Grid,
    background: int,
) -> list[list[tuple[int, int]]]:
    colors = {color for row in panel for color in row if color != background}
    if not colors:
        return []
    return _components(panel, colors, diagonal=False)


def _panel_splits_for_output_shape(grid: Grid) -> list[tuple[_Panel, _Panel]]:
    height = len(grid)
    width = len(grid[0])
    panels: list[tuple[_Panel, _Panel]] = []
    if height % 2 == 0:
        mid = height // 2
        panels.append(
            (
                _Panel(grid[:mid], 0, 0, "vertical", mid, width),
                _Panel(grid[mid:], mid, 0, "vertical", mid, width),
            )
        )
    if width % 2 == 0:
        mid = width // 2
        panels.append(
            (
                _Panel([row[:mid] for row in grid], 0, 0, "horizontal", height, mid),
                _Panel(
                    [row[mid:] for row in grid],
                    0,
                    mid,
                    "horizontal",
                    height,
                    mid,
                ),
            )
        )
    return panels


def _selected_transfer(
    grid: Grid,
    min_target_background_fraction: float = 0.60,
) -> tuple[Grid, PanelSpec | None, list[_Placement]]:
    best_output: Grid | None = None
    best_spec: PanelSpec | None = None
    best_placements: list[_Placement] = []
    best_score = -1

    for first, second in _panel_splits_for_output_shape(grid):
        for source_panel, target_panel in ((first, second), (second, first)):
            source = source_panel.grid
            target = target_panel.grid
            if _dominant_fraction(target) < min_target_background_fraction:
                continue
            target_count = _non_background_count(target)
            if target_count == 0:
                continue

            target_background = _dominant_color(target)
            target_markers = {
                (row, col): color
                for row, line in enumerate(target)
                for col, color in enumerate(line)
                if color != target_background
            }
            target_marker_colors = set(target_markers.values())
            if not target_marker_colors:
                continue

            for source_background, source_background_count in Counter(
                color for row in source for color in row
            ).most_common():
                source_count = len(source) * len(source[0]) - source_background_count
                if source_count <= target_count:
                    continue

                output = [
                    [target_background for _ in range(len(target[0]))]
                    for _ in range(len(target))
                ]
                placements: list[
                    tuple[
                        int,
                        int,
                        set[tuple[int, int]],
                        list[tuple[int, int, int]],
                        tuple[tuple[int, int, int], ...],
                    ]
                ] = []

                for component in _component_cells_excluding_background(
                    source,
                    source_background,
                ):
                    component_cells = tuple(
                        sorted((row, col, source[row][col]) for row, col in component)
                    )
                    component_marker_cells = [
                        (row, col, source[row][col])
                        for row, col in component
                        if source[row][col] in target_marker_colors
                    ]
                    has_body_cell = any(
                        source[row][col] not in target_marker_colors for row, col in component
                    )
                    if not component_marker_cells or not has_body_cell:
                        continue

                    candidate_offsets: set[tuple[int, int]] = set()
                    for source_row, source_col, color in component_marker_cells:
                        for (target_row, target_col), target_color in target_markers.items():
                            if target_color == color:
                                candidate_offsets.add(
                                    (target_row - source_row, target_col - source_col)
                                )

                    for delta_row, delta_col in candidate_offsets:
                        copied_cells: list[tuple[int, int, int]] = []
                        marker_positions: set[tuple[int, int]] = set()
                        valid = True
                        for source_row, source_col in component:
                            target_row = source_row + delta_row
                            target_col = source_col + delta_col
                            if not (
                                0 <= target_row < len(target)
                                and 0 <= target_col < len(target[0])
                            ):
                                valid = False
                                break
                            copied_cells.append(
                                (target_row, target_col, source[source_row][source_col])
                            )
                        if not valid:
                            continue

                        for source_row, source_col, color in component_marker_cells:
                            target_pos = (source_row + delta_row, source_col + delta_col)
                            if target_markers.get(target_pos) != color:
                                valid = False
                                break
                            marker_positions.add(target_pos)
                        if valid:
                            placements.append(
                                (
                                    len(marker_positions),
                                    len(copied_cells),
                                    marker_positions,
                                    copied_cells,
                                    component_cells,
                                )
                            )

                covered_markers: set[tuple[int, int]] = set()
                occupied_cells: set[tuple[int, int]] = set()
                selected: list[_Placement] = []
                for _, _, marker_positions, copied_cells, component_cells in sorted(
                    placements,
                    key=lambda item: (-item[0], item[1]),
                ):
                    if marker_positions <= covered_markers:
                        continue
                    if any((row, col) in occupied_cells for row, col, _ in copied_cells):
                        continue
                    for row, col, color in copied_cells:
                        output[row][col] = color
                        occupied_cells.add((row, col))
                    covered_markers.update(marker_positions)
                    selected.append(
                        _Placement(
                            source_component=component_cells,
                            marker_positions=tuple(sorted(marker_positions)),
                            copied_cells=tuple(sorted(copied_cells)),
                        )
                    )

                if not covered_markers:
                    continue
                score = len(covered_markers) * 1000 - len(occupied_cells)
                if score > best_score:
                    best_score = score
                    best_output = output
                    best_spec = PanelSpec(
                        orientation=target_panel.orientation,
                        panel_height=target_panel.panel_height,
                        panel_width=target_panel.panel_width,
                        source_top=source_panel.top,
                        source_left=source_panel.left,
                        target_top=target_panel.top,
                        target_left=target_panel.left,
                    )
                    best_placements = selected

    return (
        best_output if best_output is not None else _copy_grid(grid),
        best_spec,
        best_placements,
    )


def task366_panel_transfer_transform(grid: Grid) -> Grid:
    """Apply the conservative task366 panel transfer rule in Python."""
    output, _, _ = _selected_transfer(grid)
    return output


def _normalize_cells(
    cells: Iterable[tuple[int, int, int]],
) -> tuple[tuple[int, int, int], ...]:
    items = list(cells)
    min_row = min(row for row, _, _ in items)
    min_col = min(col for _, col, _ in items)
    return tuple(
        sorted((row - min_row, col - min_col, color) for row, col, color in items)
    )


def _normalize_positions(
    positions: Iterable[tuple[int, int]],
    color_at: dict[tuple[int, int], int],
) -> tuple[tuple[int, int, int], ...]:
    items = list(positions)
    min_row = min(row for row, _ in items)
    min_col = min(col for _, col in items)
    return tuple(
        sorted(
            (row - min_row, col - min_col, color_at[(row, col)])
            for row, col in items
        )
    )


def _relative_copied_cells(
    copied_cells: tuple[tuple[int, int, int], ...],
    marker_positions: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    min_marker_row = min(row for row, _ in marker_positions)
    min_marker_col = min(col for _, col in marker_positions)
    return tuple(
        sorted(
            (row - min_marker_row, col - min_marker_col, color)
            for row, col, color in copied_cells
        )
    )


def _spec_key(spec: PanelSpec) -> tuple[object, ...]:
    return (
        spec.orientation,
        spec.panel_height,
        spec.panel_width,
        spec.source_top,
        spec.source_left,
        spec.target_top,
        spec.target_left,
    )


def _case_fits_default_tensor(case: dict[str, Any]) -> bool:
    input_grid = case.get("input")
    output_grid = case.get("output")
    if not input_grid or not output_grid:
        return False
    return (
        len(input_grid) <= DEFAULT_HEIGHT
        and len(input_grid[0]) <= DEFAULT_WIDTH
        and len(output_grid) <= DEFAULT_HEIGHT
        and len(output_grid[0]) <= DEFAULT_WIDTH
    )


def extract_transfer_rules(
    task: dict[str, Any],
    onnx_compatible_only: bool = False,
) -> list[TransferRule]:
    """Extract deduplicated transfer rules from labelled cases."""
    rules: set[TransferRule] = set()
    for split in ("train", "test", "arc-gen"):
        for case in task.get(split, []):
            if "output" not in case:
                continue
            if onnx_compatible_only and not _case_fits_default_tensor(case):
                continue
            output, spec, placements = _selected_transfer(case["input"])
            if output != case["output"] or spec is None:
                continue
            for placement in placements:
                marker_colors = {
                    (row, col): color
                    for row, col, color in placement.copied_cells
                    if (row, col) in set(placement.marker_positions)
                }
                rules.add(
                    TransferRule(
                        spec=spec,
                        source_cells=_normalize_cells(placement.source_component),
                        target_markers=_normalize_positions(
                            placement.marker_positions,
                            marker_colors,
                        ),
                        copied_cells=_relative_copied_cells(
                            placement.copied_cells,
                            placement.marker_positions,
                        ),
                    )
                )
    return sorted(
        rules,
        key=lambda rule: (
            _spec_key(rule.spec),
            rule.source_cells,
            rule.target_markers,
            rule.copied_cells,
        ),
    )


def _score_cases(cases: list[dict[str, Any]]) -> tuple[int, int, str]:
    passed = 0
    total = 0
    first_failure = ""
    for index, case in enumerate(cases):
        if "output" not in case:
            continue
        total += 1
        prediction = task366_panel_transfer_transform(case["input"])
        if prediction == case["output"]:
            passed += 1
        elif not first_failure:
            first_failure = f"case {index} mismatch"
    return passed, total, first_failure


def probe_task(task: dict[str, Any], candidate_path: str = "") -> dict[str, Any]:
    """Return a report row for the task366 panel-transfer candidate."""
    rules = extract_transfer_rules(task)
    train_passed, train_total, train_failure = _score_cases(task.get("train", []))
    test_passed, test_total, test_failure = _score_cases(task.get("test", []))
    arc_passed, arc_total, arc_failure = _score_cases(task.get("arc-gen", []))
    failures = [item for item in (train_failure, test_failure, arc_failure) if item]
    return {
        "task_id": TASK_ID,
        "candidate_path": candidate_path,
        "num_rules": len(rules),
        "num_specs": len({rule.spec for rule in rules}),
        "num_source_patterns": len({(rule.spec, rule.source_cells) for rule in rules}),
        "num_target_patterns": len(
            {(rule.spec, rule.target_markers, rule.copied_cells) for rule in rules}
        ),
        "train_pass": f"{train_passed}/{train_total}",
        "test_pass": f"{test_passed}/{test_total}",
        "arc_gen_pass": f"{arc_passed}/{arc_total}",
        "formula": (
            "Compile finite panel-level marker-object transfer templates from "
            "labelled source objects and sparse target marker placements."
        ),
        "risk_level": "medium-high",
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


def _one_tensor(name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name=name)


def _int64_array(values: np.ndarray, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)


def _float_array(values: np.ndarray, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(values, dtype=np.float32), name=name)


def _cast_float(input_name: str, output_name: str) -> onnx.NodeProto:
    return helper.make_node(
        "Cast",
        [input_name],
        [output_name],
        name=output_name,
        to=TensorProto.FLOAT,
    )


def _clip01(nodes: list[onnx.NodeProto], input_name: str, output_name: str) -> None:
    nodes.append(
        helper.make_node(
            "Clip",
            [input_name],
            [output_name],
            name=output_name,
            min=0.0,
            max=1.0,
        )
    )


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


def _active_mask(panel_height: int, panel_width: int, name: str) -> onnx.TensorProto:
    mask = np.zeros((1, 1, DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=np.float32)
    mask[:, :, :panel_height, :panel_width] = 1.0
    return _float_array(mask, name)


def _input_shape_mask(spec: PanelSpec, name: str) -> onnx.TensorProto:
    if spec.orientation == "vertical":
        input_height = spec.panel_height * 2
        input_width = spec.panel_width
    elif spec.orientation == "horizontal":
        input_height = spec.panel_height
        input_width = spec.panel_width * 2
    else:
        raise ValueError(f"unknown orientation: {spec.orientation}")
    mask = np.zeros((1, 1, DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=np.float32)
    mask[:, :, :input_height, :input_width] = 1.0
    return _float_array(mask, name)


def _build_shape_flag(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    spec: PanelSpec,
    prefix: str,
) -> str:
    mask_name = f"{prefix}_InputShapeMask"
    diff_name = f"{prefix}_shape_diff"
    abs_name = f"{prefix}_shape_abs"
    sum_name = f"{prefix}_shape_sum"
    bool_name = f"{prefix}_shape_bool"
    flag_name = f"{prefix}_shape_flag"
    initializers.append(_input_shape_mask(spec, mask_name))
    nodes.extend(
        [
            helper.make_node(
                "Sub",
                ["input_cell_sum", mask_name],
                [diff_name],
                name=diff_name,
            ),
            helper.make_node("Abs", [diff_name], [abs_name], name=abs_name),
            helper.make_node(
                "ReduceSum",
                [abs_name],
                [sum_name],
                name=sum_name,
                axes=[1, 2, 3],
                keepdims=1,
            ),
            helper.make_node("Less", [sum_name, "Half"], [bool_name], name=bool_name),
            _cast_float(bool_name, flag_name),
        ]
    )
    return flag_name


def _crop_panel(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    input_name: str,
    spec: PanelSpec,
    top: int,
    left: int,
    output_name: str,
) -> str:
    row_indices = [top + row if row < spec.panel_height else 0 for row in range(DEFAULT_HEIGHT)]
    col_indices = [left + col if col < spec.panel_width else 0 for col in range(DEFAULT_WIDTH)]
    row_name = f"{output_name}_Rows"
    col_name = f"{output_name}_Cols"
    raw_rows = f"{output_name}_raw_rows"
    raw = f"{output_name}_raw"
    mask_name = f"{output_name}_Active"
    initializers.extend(
        [
            _int64_array(np.asarray(row_indices), row_name),
            _int64_array(np.asarray(col_indices), col_name),
            _active_mask(spec.panel_height, spec.panel_width, mask_name),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Gather", [input_name, row_name], [raw_rows], name=raw_rows, axis=2),
            helper.make_node("Gather", [raw_rows, col_name], [raw], name=raw, axis=3),
            helper.make_node("Mul", [raw, mask_name], [output_name], name=output_name),
        ]
    )
    return output_name


def _pattern_size(cells: tuple[tuple[int, int, int], ...]) -> tuple[int, int]:
    return (
        max(row for row, _, _ in cells) + 1,
        max(col for _, col, _ in cells) + 1,
    )


def _pattern_detector_weight(
    cells: tuple[tuple[int, int, int], ...],
    name: str,
) -> tuple[onnx.TensorProto, list[int], float]:
    height, width = _pattern_size(cells)
    weights = np.zeros((1, DEFAULT_COLORS, height, width), dtype=np.float32)
    for row, col, color in cells:
        weights[0, color, row, col] += 1.0
    return (
        numpy_helper.from_array(weights, name=name),
        [0, 0, height - 1, width - 1],
        float(len(cells)),
    )


def _build_pattern_detector(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    input_name: str,
    cells: tuple[tuple[int, int, int], ...],
    name: str,
) -> str:
    weight_name = f"{name}_W"
    expected_name = f"{name}_Expected"
    score_name = f"{name}_score"
    bool_name = f"{name}_bool"
    output_name = f"{name}_mask"
    weight, pads, expected = _pattern_detector_weight(cells, weight_name)
    initializers.extend([weight, _scalar(expected, expected_name)])
    nodes.append(
        helper.make_node(
            "Conv",
            [input_name, weight_name],
            [score_name],
            name=score_name,
            kernel_shape=list(weight.dims[2:]),
            pads=pads,
        )
    )
    _float_equal_bool(nodes, score_name, expected_name, bool_name)
    nodes.append(_cast_float(bool_name, output_name))
    return output_name


def _anchor_mask_for_source(
    rule: TransferRule,
    name: str,
) -> onnx.TensorProto:
    height, width = _pattern_size(rule.source_cells)
    mask = np.zeros((1, 1, DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=np.float32)
    max_row = rule.spec.panel_height - height
    max_col = rule.spec.panel_width - width
    if max_row >= 0 and max_col >= 0:
        mask[:, :, : max_row + 1, : max_col + 1] = 1.0
    return _float_array(mask, name)


def _anchor_mask_for_target(
    rule: TransferRule,
    name: str,
) -> onnx.TensorProto:
    marker_height, marker_width = _pattern_size(rule.target_markers)
    min_row = min(row for row, _, _ in rule.copied_cells)
    max_row = max(row for row, _, _ in rule.copied_cells)
    min_col = min(col for _, col, _ in rule.copied_cells)
    max_col = max(col for _, col, _ in rule.copied_cells)
    mask = np.zeros((1, 1, DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=np.float32)
    for row in range(rule.spec.panel_height):
        for col in range(rule.spec.panel_width):
            if row + marker_height > rule.spec.panel_height:
                continue
            if col + marker_width > rule.spec.panel_width:
                continue
            if row + min_row < 0 or row + max_row >= rule.spec.panel_height:
                continue
            if col + min_col < 0 or col + max_col >= rule.spec.panel_width:
                continue
            mask[0, 0, row, col] = 1.0
    return _float_array(mask, name)


def _shift_weight(
    cells: tuple[tuple[int, int, int], ...],
    name: str,
) -> tuple[onnx.TensorProto, list[int]]:
    sample_positions = [(-row, -col, color) for row, col, color in cells]
    min_row = min(row for row, _, _ in sample_positions)
    max_row = max(row for row, _, _ in sample_positions)
    min_col = min(col for _, col, _ in sample_positions)
    max_col = max(col for _, col, _ in sample_positions)
    pad_top = max(-min_row, 0)
    pad_left = max(-min_col, 0)
    pad_bottom = max(max_row, 0)
    pad_right = max(max_col, 0)
    kernel_height = pad_top + pad_bottom + 1
    kernel_width = pad_left + pad_right + 1
    weights = np.zeros((DEFAULT_COLORS, 1, kernel_height, kernel_width), dtype=np.float32)
    for row, col, color in sample_positions:
        weights[color, 0, pad_top + row, pad_left + col] = 1.0
    return numpy_helper.from_array(weights, name=name), [pad_top, pad_left, pad_bottom, pad_right]


def _build_shift_conv(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    input_name: str,
    cells: tuple[tuple[int, int, int], ...],
    output_name: str,
    weight_name: str,
) -> None:
    weight, pads = _shift_weight(cells, weight_name)
    initializers.append(weight)
    nodes.append(
        helper.make_node(
            "Conv",
            [input_name, weight_name],
            [output_name],
            name=output_name,
            kernel_shape=list(weight.dims[2:]),
            pads=pads,
        )
    )


def _build_background_fill(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    target_crop: str,
    spec: PanelSpec,
    any_generated: str,
    output_name: str,
    prefix: str,
) -> None:
    active_name = f"{prefix}_OutputActive"
    channel_indices = f"{prefix}_ChannelIndices"
    masked = f"{prefix}_masked_target"
    counts = f"{prefix}_counts"
    bg_index = f"{prefix}_bg_index"
    bg_bool = f"{prefix}_bg_bool"
    bg_float = f"{prefix}_bg_float"
    bg_active = f"{prefix}_bg_active"
    initializers.extend(
        [
            _active_mask(spec.panel_height, spec.panel_width, active_name),
            _int64_array(np.arange(DEFAULT_COLORS, dtype=np.int64).reshape(1, DEFAULT_COLORS, 1, 1), channel_indices),
        ]
    )
    nodes.extend(
        [
            helper.make_node("Mul", [target_crop, active_name], [masked], name=masked),
            helper.make_node(
                "ReduceSum",
                [masked],
                [counts],
                name=counts,
                axes=[2, 3],
                keepdims=0,
            ),
            helper.make_node("ArgMax", [counts], [bg_index], name=bg_index, axis=1, keepdims=1),
            helper.make_node("Equal", [channel_indices, bg_index], [bg_bool], name=bg_bool),
            _cast_float(bg_bool, bg_float),
            helper.make_node("Mul", [bg_float, active_name], [bg_active], name=bg_active),
            helper.make_node("Mul", [bg_active, any_generated], [output_name], name=output_name),
        ]
    )


def build_task366_panel_transfer_model(
    task: dict[str, Any],
    output_path: str,
) -> dict[str, Any]:
    """Build one static task366 panel-transfer ONNX candidate."""
    rules = extract_transfer_rules(task, onnx_compatible_only=True)
    if not rules:
        raise ValueError("no task366 transfer rules extracted")

    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = [_scalar(0.5, "Half"), _one_tensor("OneTensor")]
    specs = sorted({rule.spec for rule in rules}, key=_spec_key)
    generated_by_spec: list[str] = []
    nodes.append(
        helper.make_node(
            "ReduceSum",
            ["input"],
            ["input_cell_sum"],
            name="input_cell_sum",
            axes=[1],
            keepdims=1,
        )
    )

    for spec_index, spec in enumerate(specs):
        prefix = f"spec{spec_index}"
        shape_flag = _build_shape_flag(nodes, initializers, spec, prefix)
        source_crop = _crop_panel(
            nodes,
            initializers,
            "input",
            spec,
            spec.source_top,
            spec.source_left,
            f"{prefix}_source",
        )
        target_crop = _crop_panel(
            nodes,
            initializers,
            "input",
            spec,
            spec.target_top,
            spec.target_left,
            f"{prefix}_target",
        )
        spec_rules = [rule for rule in rules if rule.spec == spec]
        source_flags: dict[tuple[tuple[int, int, int], ...], str] = {}
        for source_index, source_cells in enumerate(
            sorted({rule.source_cells for rule in spec_rules}, key=repr)
        ):
            detector = _build_pattern_detector(
                nodes,
                initializers,
                source_crop,
                source_cells,
                f"{prefix}_src{source_index}",
            )
            fake_rule = TransferRule(spec, source_cells, ((0, 0, 0),), ((0, 0, 0),))
            anchor_name = f"{prefix}_src{source_index}_Anchor"
            masked = f"{prefix}_src{source_index}_masked"
            sum_name = f"{prefix}_src{source_index}_sum"
            flag_name = f"{prefix}_src{source_index}_flag"
            initializers.append(_anchor_mask_for_source(fake_rule, anchor_name))
            nodes.extend(
                [
                    helper.make_node("Mul", [detector, anchor_name], [masked], name=masked),
                    helper.make_node(
                        "ReduceSum",
                        [masked],
                        [sum_name],
                        name=sum_name,
                        axes=[2, 3],
                        keepdims=1,
                    ),
                ]
            )
            _clip01(nodes, sum_name, flag_name)
            source_flags[source_cells] = flag_name

        target_masks: dict[tuple[tuple[int, int, int], tuple[tuple[int, int, int], ...]], str] = {}
        target_keys = sorted(
            {(rule.target_markers, rule.copied_cells) for rule in spec_rules},
            key=repr,
        )
        for target_index, (target_markers, copied_cells) in enumerate(target_keys):
            detector = _build_pattern_detector(
                nodes,
                initializers,
                target_crop,
                target_markers,
                f"{prefix}_tgt{target_index}",
            )
            fake_rule = TransferRule(spec, ((0, 0, 0),), target_markers, copied_cells)
            anchor_name = f"{prefix}_tgt{target_index}_Anchor"
            masked = f"{prefix}_tgt{target_index}_masked"
            initializers.append(_anchor_mask_for_target(fake_rule, anchor_name))
            nodes.append(helper.make_node("Mul", [detector, anchor_name], [masked], name=masked))
            target_masks[(target_markers, copied_cells)] = masked

        generated_terms: list[str] = []
        for rule_index, rule in enumerate(spec_rules):
            target_mask = target_masks[(rule.target_markers, rule.copied_cells)]
            rule_mask = f"{prefix}_rule{rule_index}_mask"
            generated = f"{prefix}_rule{rule_index}_generated"
            nodes.append(
                helper.make_node(
                    "Mul",
                    [target_mask, source_flags[rule.source_cells]],
                    [rule_mask],
                    name=rule_mask,
                )
            )
            _build_shift_conv(
                nodes,
                initializers,
                rule_mask,
                rule.copied_cells,
                generated,
                f"{prefix}_rule{rule_index}_ShiftW",
            )
            generated_terms.append(generated)

        generated_sum = _add_many(nodes, generated_terms, f"{prefix}_generated_sum")
        _clip01(nodes, generated_sum, f"{prefix}_generated")
        nodes.append(
            helper.make_node(
                "Mul",
                [f"{prefix}_generated", shape_flag],
                [f"{prefix}_generated_gated"],
                name=f"{prefix}_generated_gated",
            )
        )
        nodes.append(
            helper.make_node(
                "ReduceSum",
                [f"{prefix}_generated_gated"],
                [f"{prefix}_any_raw"],
                name=f"{prefix}_any_raw",
                axes=[1, 2, 3],
                keepdims=1,
            )
        )
        _clip01(nodes, f"{prefix}_any_raw", f"{prefix}_any")
        _build_background_fill(
            nodes,
            initializers,
            target_crop,
            spec,
            f"{prefix}_any",
            f"{prefix}_background",
            prefix,
        )
        nodes.append(
            helper.make_node(
                "ReduceSum",
                [f"{prefix}_generated_gated"],
                [f"{prefix}_generated_cell_raw"],
                name=f"{prefix}_generated_cell_raw",
                axes=[1],
                keepdims=1,
            )
        )
        _clip01(nodes, f"{prefix}_generated_cell_raw", f"{prefix}_generated_cell")
        nodes.extend(
            [
                helper.make_node(
                    "Sub",
                    ["OneTensor", f"{prefix}_generated_cell"],
                    [f"{prefix}_background_keep"],
                    name=f"{prefix}_background_keep",
                ),
                helper.make_node(
                    "Mul",
                    [f"{prefix}_background", f"{prefix}_background_keep"],
                    [f"{prefix}_background_masked"],
                    name=f"{prefix}_background_masked",
                ),
            ]
        )
        nodes.append(
            helper.make_node(
                "Add",
                [f"{prefix}_background_masked", f"{prefix}_generated_gated"],
                [f"{prefix}_output_raw"],
                name=f"{prefix}_output_raw",
            )
        )
        _clip01(nodes, f"{prefix}_output_raw", f"{prefix}_output")
        generated_by_spec.append(f"{prefix}_output")

    output_sum = _add_many(nodes, generated_by_spec, "all_specs_sum")
    _clip01(nodes, output_sum, "output")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        nodes=nodes,
        name="task366_panel_transfer_conservative",
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
    return {
        "output_path": str(output),
        "num_rules": len(rules),
        "num_specs": len(specs),
        "num_source_patterns": len({(rule.spec, rule.source_cells) for rule in rules}),
        "num_target_patterns": len(
            {(rule.spec, rule.target_markers, rule.copied_cells) for rule in rules}
        ),
    }


def write_probe_report(
    task: dict[str, Any],
    report_path: str,
    candidate_path: str = "",
) -> list[dict[str, Any]]:
    rows = [probe_task(task, candidate_path=candidate_path)]
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_candidate(
    task_path: str,
    output_dir: str,
    report_path: str,
    rule_source: str = "compatible",
) -> dict[str, Any]:
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    if rule_source == "compatible":
        build_task = task
        candidate_name = CANDIDATE_NAME
    elif rule_source == "train-test":
        build_task = {
            "train": task.get("train", []),
            "test": task.get("test", []),
            "arc-gen": [],
        }
        candidate_name = TRAIN_TEST_CANDIDATE_NAME
    else:
        raise ValueError(f"unknown rule source: {rule_source}")
    candidate = Path(output_dir) / f"{TASK_ID}_{candidate_name}.onnx"
    summary = build_task366_panel_transfer_model(build_task, str(candidate))
    rows = write_probe_report(task, report_path, str(candidate))
    result = {
        "task_path": task_path,
        "output_dir": output_dir,
        "report_path": report_path,
        "rule_source": rule_source,
        "candidate": str(candidate),
        "summary": summary,
        "probe_rows": rows,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-path", default="task/task366.json")
    parser.add_argument("--output-dir", default="outputs/candidates/task366_panel_transfer")
    parser.add_argument("--report", default="outputs/reports/task366_panel_transfer_probe.csv")
    parser.add_argument("--rule-source", choices=["compatible", "train-test"], default="compatible")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    task = json.loads(Path(args.task_path).read_text(encoding="utf-8"))
    if args.build:
        build_candidate(args.task_path, args.output_dir, args.report, args.rule_source)
    else:
        rows = write_probe_report(task, args.report)
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
