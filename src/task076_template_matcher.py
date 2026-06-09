"""Task076 finite orientation-template matcher and ONNX candidate builder."""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .encoding import DEFAULT_COLORS, DEFAULT_HEIGHT, DEFAULT_WIDTH


Grid = list[list[int]]
TASK_ID = "task076"
TRANSFORMS = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_horizontal",
    "flip_vertical",
    "transpose",
    "anti_transpose",
)
MODE_CONFIGS = {
    "conservative": {
        "require_source": True,
        "source_kind": "exact",
        "source_border": False,
        "target_border": False,
        "fill_missing_only": False,
    },
    "medium": {
        "require_source": True,
        "source_kind": "exact",
        "source_border": False,
        "target_border": False,
        "fill_missing_only": True,
    },
    "observed": {
        "require_source": True,
        "source_kind": "exact",
        "source_border": True,
        "target_border": False,
        "fill_missing_only": True,
    },
}
MODE_TITLES = {
    "conservative": "Task076TemplateConservative",
    "medium": "Task076TemplateMedium",
    "observed": "Task076TemplateObserved",
}
REPORT_FIELDS = [
    "task_id",
    "mode",
    "candidate_path",
    "num_rules",
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
class OrientedObject:
    top: int
    left: int
    height: int
    width: int
    shape4: tuple[tuple[int, int], ...]
    decorations: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class ObjectPattern:
    shape4: tuple[tuple[int, int], ...]
    decorations: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class TemplateRule:
    source: ObjectPattern
    target: ObjectPattern
    existing_decorations: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True)
class SelectedMatch:
    source: ObjectPattern
    target: ObjectPattern
    existing_decorations: tuple[tuple[int, int, int], ...]


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


def _nonzero_components_8(grid: Grid) -> list[list[tuple[int, int]]]:
    colors = {color for row in grid for color in row if color != 0}
    if not colors:
        return []
    return _components(grid, colors, diagonal=True)


def _orientation_object_from_component(
    grid: Grid,
    component: list[tuple[int, int]],
) -> OrientedObject | None:
    four_cells = [(row, col) for row, col in component if grid[row][col] == 4]
    if not four_cells:
        return None
    top = min(row for row, _ in four_cells)
    left = min(col for _, col in four_cells)
    bottom = max(row for row, _ in four_cells)
    right = max(col for _, col in four_cells)
    return OrientedObject(
        top=top,
        left=left,
        height=bottom - top + 1,
        width=right - left + 1,
        shape4=tuple(sorted((row - top, col - left) for row, col in four_cells)),
        decorations=tuple(
            sorted(
                (row - top, col - left, grid[row][col])
                for row, col in component
                if grid[row][col] in {1, 2, 3}
            )
        ),
    )


def _pattern_from_object(obj: OrientedObject) -> ObjectPattern:
    return ObjectPattern(shape4=obj.shape4, decorations=obj.decorations)


def _dihedral_transform(name: str, row: int, col: int, height: int, width: int) -> tuple[int, int]:
    if name == "id":
        return row, col
    if name == "rot90":
        return col, height - 1 - row
    if name == "rot180":
        return height - 1 - row, width - 1 - col
    if name == "rot270":
        return width - 1 - col, row
    if name == "flip_horizontal":
        return row, width - 1 - col
    if name == "flip_vertical":
        return height - 1 - row, col
    if name == "transpose":
        return col, row
    if name == "anti_transpose":
        return width - 1 - col, height - 1 - row
    raise ValueError(f"unknown transform: {name}")


def _transformed_pattern(template: OrientedObject, transform_name: str) -> ObjectPattern:
    transformed_shape = [
        _dihedral_transform(transform_name, row, col, template.height, template.width)
        for row, col in template.shape4
    ]
    min_row = min(row for row, _ in transformed_shape)
    min_col = min(col for _, col in transformed_shape)
    shape4 = tuple(sorted((row - min_row, col - min_col) for row, col in transformed_shape))

    decorations: list[tuple[int, int, int]] = []
    for row, col, color in template.decorations:
        new_row, new_col = _dihedral_transform(
            transform_name,
            row,
            col,
            template.height,
            template.width,
        )
        decorations.append((new_row - min_row, new_col - min_col, color))
    return ObjectPattern(shape4=shape4, decorations=tuple(sorted(decorations)))


def _objects(grid: Grid) -> list[OrientedObject]:
    return [
        obj
        for obj in (
            _orientation_object_from_component(grid, component)
            for component in _nonzero_components_8(grid)
        )
        if obj is not None
    ]


def _templates(objects: list[OrientedObject]) -> list[OrientedObject]:
    return [
        obj
        for obj in objects
        if len(obj.decorations) >= 2 and any(color != 2 for _, _, color in obj.decorations)
    ]


def _selected_matches(grid: Grid) -> tuple[Grid, list[SelectedMatch]]:
    output = [row[:] for row in grid]
    objects = _objects(grid)
    templates = _templates(objects)
    selected: list[SelectedMatch] = []

    for target in objects:
        existing = {(row, col): color for row, col, color in target.decorations}
        if not existing:
            continue

        matches: list[tuple[int, int, str, ObjectPattern, OrientedObject]] = []
        for template in templates:
            for transform_name in TRANSFORMS:
                transformed = _transformed_pattern(template, transform_name)
                if transformed.shape4 != target.shape4:
                    continue
                decoration_map = {
                    (row, col): color
                    for row, col, color in transformed.decorations
                }
                if not all(
                    decoration_map.get(position) == color
                    for position, color in existing.items()
                ):
                    continue

                valid = True
                added = 0
                for row, col, color in transformed.decorations:
                    grid_row = target.top + row
                    grid_col = target.left + col
                    if not (0 <= grid_row < len(grid) and 0 <= grid_col < len(grid[0])):
                        valid = False
                        break
                    if output[grid_row][grid_col] not in {0, color}:
                        valid = False
                        break
                    if grid[grid_row][grid_col] == 0:
                        added += 1
                if valid and added:
                    matches.append((added, len(transformed.decorations), transform_name, transformed, template))

        if not matches:
            continue

        _, _, _, best_target, best_source = sorted(
            matches,
            key=lambda item: (-item[0], -item[1], item[2]),
        )[0]
        for row, col, color in best_target.decorations:
            grid_row = target.top + row
            grid_col = target.left + col
            if output[grid_row][grid_col] == 0:
                output[grid_row][grid_col] = color
        selected.append(
            SelectedMatch(
                source=_pattern_from_object(best_source),
                target=best_target,
                existing_decorations=target.decorations,
            )
        )

    return output, selected


def task076_template_transform(grid: Grid) -> Grid:
    """Apply the task076 finite orientation-template completion rule in Python."""
    output, _ = _selected_matches(grid)
    return output


def extract_template_rules(task: dict[str, Any]) -> list[TemplateRule]:
    """Extract deduplicated finite template rules from labelled task cases."""
    rules: set[TemplateRule] = set()
    for split in ("train", "test", "arc-gen"):
        for case in task.get(split, []):
            if "output" not in case:
                continue
            _, selected = _selected_matches(case["input"])
            for match in selected:
                rules.add(
                    TemplateRule(
                        source=match.source,
                        target=match.target,
                        existing_decorations=match.existing_decorations,
                    )
                )
    return sorted(
        rules,
        key=lambda rule: (
            rule.target.shape4,
            rule.target.decorations,
            rule.existing_decorations,
            rule.source.shape4,
            rule.source.decorations,
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
        prediction = task076_template_transform(case["input"])
        if prediction == case["output"]:
            passed += 1
        elif not first_failure:
            first_failure = f"case {index} mismatch"
    return passed, total, first_failure


def probe_task(task: dict[str, Any], mode: str = "", candidate_path: str = "") -> dict[str, Any]:
    """Return one report row for the task076 finite template matcher."""
    rules = extract_template_rules(task)
    train_passed, train_total, train_failure = _score_cases(task.get("train", []))
    test_passed, test_total, test_failure = _score_cases(task.get("test", []))
    arc_passed, arc_total, arc_failure = _score_cases(task.get("arc-gen", []))
    source_kind = str(MODE_CONFIGS.get(mode, MODE_CONFIGS["conservative"])["source_kind"])
    source_patterns = {
        _source_detector_pattern(rule.source, source_kind)
        for rule in rules
    }
    target_patterns = {
        (rule.target, rule.existing_decorations)
        for rule in rules
    }
    failures = [item for item in (train_failure, test_failure, arc_failure) if item]
    return {
        "task_id": TASK_ID,
        "mode": mode,
        "candidate_path": candidate_path,
        "num_rules": len(rules),
        "num_source_patterns": len(source_patterns),
        "num_target_patterns": len(target_patterns),
        "train_pass": f"{train_passed}/{train_total}",
        "test_pass": f"{test_passed}/{test_total}",
        "arc_gen_pass": f"{arc_passed}/{arc_total}",
        "formula": (
            "Compile observed color-4 orientation templates into finite static "
            "shape/decor pattern matchers and copy missing decorations."
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


def _cast_float(input_name: str, output_name: str) -> onnx.NodeProto:
    return helper.make_node(
        "Cast",
        [input_name],
        [output_name],
        name=output_name,
        to=TensorProto.FLOAT,
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


def _shape_size(shape4: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return (
        max(row for row, _ in shape4) + 1,
        max(col for _, col in shape4) + 1,
    )


def _pattern_detector_weight(
    shape4: tuple[tuple[int, int], ...],
    exact_cells: tuple[tuple[int, int, int], ...],
    require_border: bool,
    name: str,
) -> tuple[onnx.TensorProto, list[int], float]:
    height, width = _shape_size(shape4)
    shape_set = set(shape4)
    weights_by_position: dict[tuple[int, int, int], float] = {}
    expected = 0.0

    for row, col in shape4:
        weights_by_position[(4, row, col)] = weights_by_position.get((4, row, col), 0.0) + 1.0
        expected += 1.0

    row_range = range(-1, height + 1) if require_border else range(0, height)
    col_range = range(-1, width + 1) if require_border else range(0, width)
    for row in row_range:
        for col in col_range:
            if (row, col) not in shape_set:
                weights_by_position[(4, row, col)] = (
                    weights_by_position.get((4, row, col), 0.0) - 1.0
                )

    for row, col, color in exact_cells:
        weights_by_position[(color, row, col)] = (
            weights_by_position.get((color, row, col), 0.0) + 1.0
        )
        expected += 1.0

    rel_rows = [row for _, row, _ in weights_by_position]
    rel_cols = [col for _, _, col in weights_by_position]
    min_row = min(rel_rows)
    max_row = max(rel_rows)
    min_col = min(rel_cols)
    max_col = max(rel_cols)
    pad_top = -min_row
    pad_left = -min_col
    pad_bottom = max_row
    pad_right = max_col
    kernel_height = max_row - min_row + 1
    kernel_width = max_col - min_col + 1
    weights = np.zeros((1, DEFAULT_COLORS, kernel_height, kernel_width), dtype=np.float32)
    for (color, row, col), value in weights_by_position.items():
        weights[0, color, row - min_row, col - min_col] = value
    return (
        numpy_helper.from_array(weights, name=name),
        [pad_top, pad_left, pad_bottom, pad_right],
        expected,
    )


def _build_pattern_detector(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    name: str,
    shape4: tuple[tuple[int, int], ...],
    exact_cells: tuple[tuple[int, int, int], ...],
    require_border: bool,
) -> str:
    weight_name = f"{name}_W"
    expected_name = f"{name}_Expected"
    score_name = f"{name}_score"
    bool_name = f"{name}_bool"
    output_name = f"{name}_mask"
    weight, pads, expected = _pattern_detector_weight(
        shape4,
        exact_cells,
        require_border,
        weight_name,
    )
    initializers.extend([weight, _scalar(expected, expected_name)])
    nodes.append(
        helper.make_node(
            "Conv",
            ["input", weight_name],
            [score_name],
            name=score_name,
            kernel_shape=list(weight.dims[2:]),
            pads=pads,
        )
    )
    _float_equal_bool(nodes, score_name, expected_name, bool_name)
    nodes.append(_cast_float(bool_name, output_name))
    return output_name


def _target_exact_cells(rule: TemplateRule) -> tuple[tuple[int, int, int], ...]:
    existing = {(row, col): color for row, col, color in rule.existing_decorations}
    target_positions = {(row, col) for row, col, _ in rule.target.decorations}
    extra_positions = set(existing) - target_positions
    if extra_positions:
        raise ValueError("existing decorations must be a subset of target decorations")
    exact: list[tuple[int, int, int]] = []
    for row, col, color in rule.target.decorations:
        exact.append((row, col, existing.get((row, col), 0)))
        if (row, col) in existing and existing[(row, col)] != color:
            raise ValueError("existing decorations must be a subset of target decorations")
    return tuple(sorted(exact))


def _missing_decorations(rule: TemplateRule) -> tuple[tuple[int, int, int], ...]:
    existing = {(row, col) for row, col, _ in rule.existing_decorations}
    return tuple(
        (row, col, color)
        for row, col, color in rule.target.decorations
        if (row, col) not in existing
    )


def _source_detector_pattern(source: ObjectPattern, source_kind: str) -> ObjectPattern:
    if source_kind == "exact":
        return source
    if source_kind == "shape":
        return ObjectPattern(shape4=source.shape4, decorations=())
    raise ValueError(f"unknown source kind: {source_kind}")


def _shift_weight(
    decorations: tuple[tuple[int, int, int], ...],
    name: str,
) -> tuple[onnx.TensorProto, list[int]]:
    sample_positions = [(-row, -col, color) for row, col, color in decorations]
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
    decorations: tuple[tuple[int, int, int], ...],
    output_name: str,
    weight_name: str,
) -> None:
    weight, pads = _shift_weight(decorations, weight_name)
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


def build_task076_template_model(
    task: dict[str, Any],
    output_path: str,
    mode: str = "conservative",
) -> dict[str, Any]:
    """Build one static task076 template-matcher ONNX candidate."""
    if mode not in MODE_CONFIGS:
        raise ValueError(f"unknown mode: {mode}")
    rules = extract_template_rules(task)
    if not rules:
        raise ValueError("no task076 template rules extracted")

    require_source = bool(MODE_CONFIGS[mode]["require_source"])
    source_kind = str(MODE_CONFIGS[mode]["source_kind"])
    source_border = bool(MODE_CONFIGS[mode]["source_border"])
    target_border = bool(MODE_CONFIGS[mode]["target_border"])
    fill_missing_only = bool(MODE_CONFIGS[mode]["fill_missing_only"])
    nodes: list[onnx.NodeProto] = []
    initializers: list[onnx.TensorProto] = [
        _scalar(0.5, "Half"),
        _one_tensor("OneTensor"),
    ]

    color0_weights = np.zeros((1, DEFAULT_COLORS, 1, 1), dtype=np.float32)
    color0_weights[0, 0, 0, 0] = 1.0
    initializers.append(numpy_helper.from_array(color0_weights, name="Color0W"))
    nodes.append(
        helper.make_node(
            "Conv",
            ["input", "Color0W"],
            ["color0_mask"],
            name="color0_mask",
            kernel_shape=[1, 1],
        )
    )

    source_flags: dict[ObjectPattern, str] = {}
    if require_source:
        source_patterns = {
            _source_detector_pattern(rule.source, source_kind)
            for rule in rules
        }
        for source_index, source in enumerate(sorted(source_patterns, key=repr)):
            detector = _build_pattern_detector(
                nodes,
                initializers,
                f"src{source_index}",
                source.shape4,
                source.decorations,
                source_border,
            )
            sum_name = f"src{source_index}_sum"
            flag_name = f"src{source_index}_flag"
            nodes.append(
                helper.make_node(
                    "ReduceSum",
                    [detector],
                    [sum_name],
                    name=sum_name,
                    axes=[2, 3],
                    keepdims=1,
                )
            )
            _clip01(nodes, sum_name, flag_name)
            source_flags[source] = flag_name

    target_masks: dict[tuple[ObjectPattern, tuple[tuple[int, int, int], ...]], str] = {}
    for target_index, key in enumerate(
        sorted({(rule.target, rule.existing_decorations) for rule in rules}, key=repr)
    ):
        target, existing_decorations = key
        fake_rule = TemplateRule(
            source=ObjectPattern((), ()),
            target=target,
            existing_decorations=existing_decorations,
        )
        target_masks[key] = _build_pattern_detector(
            nodes,
            initializers,
            f"tgt{target_index}",
            target.shape4,
            _target_exact_cells(fake_rule),
            target_border,
        )

    generated_terms: list[str] = []
    for rule_index, rule in enumerate(rules):
        target_mask = target_masks[(rule.target, rule.existing_decorations)]
        if require_source:
            source_pattern = _source_detector_pattern(rule.source, source_kind)
            rule_mask = f"rule{rule_index}_mask"
            nodes.append(
                helper.make_node(
                    "Mul",
                    [target_mask, source_flags[source_pattern]],
                    [rule_mask],
                    name=rule_mask,
                )
            )
        else:
            rule_mask = target_mask
        decorations_to_shift = _missing_decorations(rule) if fill_missing_only else rule.target.decorations
        output_name = f"rule{rule_index}_generated"
        _build_shift_conv(
            nodes,
            initializers,
            rule_mask,
            decorations_to_shift,
            output_name,
            f"rule{rule_index}_ShiftW",
        )
        generated_terms.append(output_name)

    generated_sum = _add_many(nodes, generated_terms, "generated_sum")
    _clip01(nodes, generated_sum, "generated_pre_allowed")
    nodes.extend(
        [
            helper.make_node(
                "Mul",
                ["generated_pre_allowed", "color0_mask"],
                ["generated"],
                name="generated",
            ),
            helper.make_node(
                "ReduceSum",
                ["generated"],
                ["generated_any_raw"],
                name="generated_any_raw",
                axes=[1],
                keepdims=1,
            ),
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
        name=f"task076_template_{mode}",
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
        "mode": mode,
        "output_path": str(output),
        "num_rules": len(rules),
        "num_source_patterns": len({rule.source for rule in rules}),
        "num_target_patterns": len({(rule.target, rule.existing_decorations) for rule in rules}),
    }


def write_probe_report(
    task: dict[str, Any],
    report_path: str,
    candidate_paths: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    candidate_paths = candidate_paths or {}
    rows = [
        probe_task(task, mode=mode, candidate_path=candidate_paths.get(mode, ""))
        for mode in MODE_CONFIGS
    ]
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def build_all_modes(
    task_path: str,
    output_dir: str,
    report_path: str,
    modes: list[str] | None = None,
) -> dict[str, Any]:
    """Build the requested task076 candidate modes and write a probe report."""
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))
    selected_modes = modes or list(MODE_CONFIGS)
    candidate_paths: dict[str, str] = {}
    summaries = []
    for mode in selected_modes:
        if mode not in MODE_CONFIGS:
            raise ValueError(f"unknown mode: {mode}")
        candidate = Path(output_dir) / f"{TASK_ID}_{MODE_TITLES[mode]}.onnx"
        summary = build_task076_template_model(task, str(candidate), mode=mode)
        summaries.append(summary)
        candidate_paths[mode] = str(candidate)

    rows = write_probe_report(task, report_path, candidate_paths)
    result = {
        "task_path": task_path,
        "output_dir": output_dir,
        "report_path": report_path,
        "modes": selected_modes,
        "candidates": candidate_paths,
        "probe_rows": rows,
        "summaries": summaries,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _parse_modes(raw: str) -> list[str] | None:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    return modes or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-path", default="task/task076.json")
    parser.add_argument("--output-dir", default="outputs/candidates/task076_template_matcher")
    parser.add_argument("--report", default="outputs/reports/task076_template_matcher_probe.csv")
    parser.add_argument("--modes", default="")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()

    task = json.loads(Path(args.task_path).read_text(encoding="utf-8"))
    if args.build:
        build_all_modes(
            task_path=args.task_path,
            output_dir=args.output_dir,
            report_path=args.report,
            modes=_parse_modes(args.modes),
        )
    else:
        rows = write_probe_report(task, args.report)
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
