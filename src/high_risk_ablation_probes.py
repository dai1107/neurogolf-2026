"""Probe medium/high-risk task rules without building ONNX candidates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, deque
from pathlib import Path
from typing import Callable


Grid = list[list[int]]

FIELDS = [
    "task_id",
    "probe_name",
    "matched_train",
    "matched_test",
    "matched_arc_gen",
    "total_train",
    "total_test",
    "total_arc_gen",
    "formula",
    "uncertainty",
    "builder_possible",
    "risk_level",
    "failure_reason",
]


def _neighbors(row: int, col: int, height: int, width: int, diagonal: bool) -> list[tuple[int, int]]:
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        offsets.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])
    result = []
    for dy, dx in offsets:
        next_row = row + dy
        next_col = col + dx
        if 0 <= next_row < height and 0 <= next_col < width:
            result.append((next_row, next_col))
    return result


def _copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


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


def _fill_components_containing_marker(grid: Grid, diagonal: bool) -> Grid:
    output = _copy_grid(grid)
    for component in _components(grid, {0, 2}, diagonal):
        if any(grid[row][col] == 2 for row, col in component):
            for row, col in component:
                if output[row][col] == 0:
                    output[row][col] = 2
    return output


def _fill_zero_components_same_size_as_marker_component(grid: Grid, diagonal: bool) -> Grid:
    output = _copy_grid(grid)
    marker_components = [
        component
        for component in _components(grid, {0, 2}, diagonal)
        if any(grid[row][col] == 2 for row, col in component)
    ]
    marker_sizes = {len(component) for component in marker_components}
    for component in _components(grid, {0}, diagonal):
        if len(component) in marker_sizes:
            for row, col in component:
                output[row][col] = 2
    return output


def _component_signature(component: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_row = min(row for row, _ in component)
    min_col = min(col for _, col in component)
    return tuple(sorted((row - min_row, col - min_col) for row, col in component))


def _fill_zero_components_same_shape_as_marker_component(grid: Grid, diagonal: bool) -> Grid:
    output = _copy_grid(grid)
    marker_signatures = {
        _component_signature(component)
        for component in _components(grid, {0, 2}, diagonal)
        if any(grid[row][col] == 2 for row, col in component)
    }
    for component in _components(grid, {0}, diagonal):
        if _component_signature(component) in marker_signatures:
            for row, col in component:
                output[row][col] = 2
    return output


def _fill_horizontal_zero_runs_matching_marker_lengths(grid: Grid) -> Grid:
    output = _copy_grid(grid)
    lengths: set[int] = set()
    for row in grid:
        col = 0
        while col < len(row):
            if row[col] != 2:
                col += 1
                continue
            start = col
            while col < len(row) and row[col] == 2:
                col += 1
            lengths.add(col - start)
    for row_index, row in enumerate(grid):
        col = 0
        while col < len(row):
            if row[col] != 0:
                col += 1
                continue
            start = col
            while col < len(row) and row[col] == 0:
                col += 1
            if col - start in lengths:
                for fill_col in range(start, col):
                    output[row_index][fill_col] = 2
    return output


def _line_directions() -> tuple[tuple[int, int], ...]:
    return ((0, 1), (1, 0), (1, 1), (1, -1))


def _line_run_lengths(grid: Grid, marker_color: int = 2) -> set[tuple[int, tuple[int, int]]]:
    height = len(grid)
    width = len(grid[0])
    runs: set[tuple[int, tuple[int, int]]] = set()
    for delta_row, delta_col in _line_directions():
        for row in range(height):
            for col in range(width):
                prev_row = row - delta_row
                prev_col = col - delta_col
                if (
                    0 <= prev_row < height
                    and 0 <= prev_col < width
                    and grid[prev_row][prev_col] == marker_color
                ):
                    continue
                if grid[row][col] != marker_color:
                    continue
                length = 0
                run_row = row
                run_col = col
                while (
                    0 <= run_row < height
                    and 0 <= run_col < width
                    and grid[run_row][run_col] == marker_color
                ):
                    length += 1
                    run_row += delta_row
                    run_col += delta_col
                if length >= 2:
                    runs.add((length, (delta_row, delta_col)))
    return runs


def _complete_line_patterns(grid: Grid) -> Grid:
    """Probe-only completion of blank runs matching observed color-2 line lengths."""
    output = _copy_grid(grid)
    observed_runs = _line_run_lengths(grid, marker_color=2)
    if not observed_runs:
        return output

    height = len(grid)
    width = len(grid[0])
    for length, (delta_row, delta_col) in observed_runs:
        for row in range(height):
            for col in range(width):
                end_row = row + (length - 1) * delta_row
                end_col = col + (length - 1) * delta_col
                if not (0 <= end_row < height and 0 <= end_col < width):
                    continue
                cells = [
                    (row + offset * delta_row, col + offset * delta_col)
                    for offset in range(length)
                ]
                values = [grid[cell_row][cell_col] for cell_row, cell_col in cells]
                if all(value == 0 for value in values):
                    for cell_row, cell_col in cells:
                        output[cell_row][cell_col] = 2
    return output


def _dominant_color(grid: Grid) -> int:
    return Counter(color for row in grid for color in row).most_common(1)[0][0]


def _dominant_fraction(grid: Grid) -> float:
    counts = Counter(color for row in grid for color in row)
    return counts.most_common(1)[0][1] / (len(grid) * len(grid[0]))


def _non_background_count(grid: Grid) -> int:
    background = _dominant_color(grid)
    return sum(1 for row in grid for color in row if color != background)


def _panel_splits_for_output_shape(grid: Grid) -> list[tuple[Grid, Grid]]:
    height = len(grid)
    width = len(grid[0])
    panels: list[tuple[Grid, Grid]] = []
    if height % 2 == 0:
        mid = height // 2
        panels.append((grid[:mid], grid[mid:]))
    if width % 2 == 0:
        mid = width // 2
        panels.append(([row[:mid] for row in grid], [row[mid:] for row in grid]))
    return panels


def _component_cells_excluding_background(panel: Grid, background: int) -> list[list[tuple[int, int]]]:
    colors = {color for row in panel for color in row if color != background}
    if not colors:
        return []
    return _components(panel, colors, diagonal=False)


def _copy_marker_matched_source_objects_to_sparse_panel(
    grid: Grid,
    min_target_background_fraction: float = 0.60,
) -> Grid:
    best_output: Grid | None = None
    best_score = -1

    for first, second in _panel_splits_for_output_shape(grid):
        panel_pairs = ((first, second), (second, first))
        for source, target in panel_pairs:
            # Conservative guard: the target panel must be clearly sparse.
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

            # Multi-source-background guard: try every plausible source background.
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
                    tuple[int, int, set[tuple[int, int]], list[tuple[int, int, int]]]
                ] = []

                for component in _component_cells_excluding_background(source, source_background):
                    component_marker_cells = [
                        (row, col, source[row][col])
                        for row, col in component
                        if source[row][col] in target_marker_colors
                    ]
                    has_body_cell = any(
                        source[row][col] not in target_marker_colors for row, col in component
                    )
                    # Degenerate marker-only guard: do not copy components that
                    # contain only marker cells and no object body.
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
                                )
                            )

                covered_markers: set[tuple[int, int]] = set()
                occupied_cells: set[tuple[int, int]] = set()
                for _, _, marker_positions, copied_cells in sorted(
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

                if not covered_markers:
                    continue
                score = len(covered_markers) * 1000 - len(occupied_cells)
                if score > best_score:
                    best_score = score
                    best_output = output

    return best_output if best_output is not None else _copy_grid(grid)


def _nonzero_components_8(grid: Grid) -> list[list[tuple[int, int]]]:
    colors = {color for row in grid for color in row if color != 0}
    if not colors:
        return []
    return _components(grid, colors, diagonal=True)


def _orientation_object_from_component(
    grid: Grid,
    component: list[tuple[int, int]],
) -> dict[str, object] | None:
    four_cells = [(row, col) for row, col in component if grid[row][col] == 4]
    if not four_cells:
        return None

    top = min(row for row, _ in four_cells)
    left = min(col for _, col in four_cells)
    bottom = max(row for row, _ in four_cells)
    right = max(col for _, col in four_cells)
    return {
        "top": top,
        "left": left,
        "height": bottom - top + 1,
        "width": right - left + 1,
        "shape4": frozenset((row - top, col - left) for row, col in four_cells),
        "decorations": tuple(
            sorted(
                (row - top, col - left, grid[row][col])
                for row, col in component
                if grid[row][col] in {1, 2, 3}
            )
        ),
    }


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


def _transformed_orientation_template(
    template: dict[str, object],
    transform_name: str,
) -> tuple[frozenset[tuple[int, int]], tuple[tuple[int, int, int], ...]]:
    height = int(template["height"])
    width = int(template["width"])
    transformed_shape = [
        _dihedral_transform(transform_name, row, col, height, width)
        for row, col in template["shape4"]  # type: ignore[index]
    ]
    min_row = min(row for row, _ in transformed_shape)
    min_col = min(col for _, col in transformed_shape)
    normalized_shape = frozenset((row - min_row, col - min_col) for row, col in transformed_shape)

    decorations = []
    for row, col, color in template["decorations"]:  # type: ignore[index]
        new_row, new_col = _dihedral_transform(transform_name, row, col, height, width)
        decorations.append((new_row - min_row, new_col - min_col, color))
    return normalized_shape, tuple(sorted(decorations))


def _complete_orientation_aware_marker_copy(grid: Grid) -> Grid:
    """Complete sparse marker objects from same-shape decorated color-4 templates."""
    output = _copy_grid(grid)
    objects = [
        obj
        for obj in (
            _orientation_object_from_component(grid, component)
            for component in _nonzero_components_8(grid)
        )
        if obj is not None
    ]
    templates = [
        obj
        for obj in objects
        if len(obj["decorations"]) >= 2  # type: ignore[arg-type]
        and any(color != 2 for _, _, color in obj["decorations"])  # type: ignore[index]
    ]

    transform_names = (
        "id",
        "rot90",
        "rot180",
        "rot270",
        "flip_horizontal",
        "flip_vertical",
        "transpose",
        "anti_transpose",
    )
    for target in objects:
        existing = {
            (row, col): color
            for row, col, color in target["decorations"]  # type: ignore[index]
        }
        if not existing:
            continue

        matches: list[tuple[int, int, str, tuple[tuple[int, int, int], ...]]] = []
        for template in templates:
            for transform_name in transform_names:
                shape, decorations = _transformed_orientation_template(template, transform_name)
                if shape != target["shape4"]:
                    continue
                decoration_map = {(row, col): color for row, col, color in decorations}
                if not all(decoration_map.get(position) == color for position, color in existing.items()):
                    continue

                valid = True
                added = 0
                for row, col, color in decorations:
                    grid_row = int(target["top"]) + row
                    grid_col = int(target["left"]) + col
                    if not (0 <= grid_row < len(grid) and 0 <= grid_col < len(grid[0])):
                        valid = False
                        break
                    if output[grid_row][grid_col] not in {0, color}:
                        valid = False
                        break
                    if grid[grid_row][grid_col] == 0:
                        added += 1
                if valid and added:
                    matches.append((added, len(decorations), transform_name, decorations))

        if not matches:
            continue
        _, _, _, best_decorations = sorted(matches, key=lambda item: (-item[0], -item[1], item[2]))[0]
        for row, col, color in best_decorations:
            grid_row = int(target["top"]) + row
            grid_col = int(target["left"]) + col
            if output[grid_row][grid_col] == 0:
                output[grid_row][grid_col] = color

    return output


PROBES: dict[str, tuple[str, Callable[[Grid], Grid], str, str]] = {
    "component_fill_4": (
        "Fill every 0 cell in 4-connected {0,2} components that contain color 2.",
        lambda grid: _fill_components_containing_marker(grid, diagonal=False),
        "maybe",
        "high",
    ),
    "component_fill_8": (
        "Fill every 0 cell in 8-connected {0,2} components that contain color 2.",
        lambda grid: _fill_components_containing_marker(grid, diagonal=True),
        "maybe",
        "high",
    ),
    "same_size_zero_component_4": (
        "Fill 4-connected 0 components whose size equals a {0,2} marker component size.",
        lambda grid: _fill_zero_components_same_size_as_marker_component(grid, diagonal=False),
        "maybe",
        "high",
    ),
    "same_size_zero_component_8": (
        "Fill 8-connected 0 components whose size equals a {0,2} marker component size.",
        lambda grid: _fill_zero_components_same_size_as_marker_component(grid, diagonal=True),
        "maybe",
        "high",
    ),
    "same_shape_zero_component_4": (
        "Fill 4-connected 0 components matching a {0,2} marker component normalized shape.",
        lambda grid: _fill_zero_components_same_shape_as_marker_component(grid, diagonal=False),
        "maybe",
        "high",
    ),
    "same_shape_zero_component_8": (
        "Fill 8-connected 0 components matching a {0,2} marker component normalized shape.",
        lambda grid: _fill_zero_components_same_shape_as_marker_component(grid, diagonal=True),
        "maybe",
        "high",
    ),
    "line_pattern_completion": (
        "Probe-only completion of horizontal, vertical, and diagonal 0-runs matching observed color-2 line lengths.",
        _complete_line_patterns,
        "no",
        "probe",
    ),
    "two_panel_marker_object_transfer_conservative": (
        "Split equal panels; require target background dominance, try multiple source backgrounds, and reject marker-only source components before copying matched source objects.",
        _copy_marker_matched_source_objects_to_sparse_panel,
        "hard",
        "medium",
    ),
    "orientation_aware_marker_copy": (
        "Complete sparse marker objects by copying decorations from same-shape 8-connected color-4 templates under a dihedral transform.",
        _complete_orientation_aware_marker_copy,
        "hard",
        "high",
    ),
}


def _score_cases(cases: list[dict], transform: Callable[[Grid], Grid]) -> tuple[int, int, str]:
    if not cases:
        return 0, 0, ""
    passed = 0
    first_failure = ""
    for index, case in enumerate(cases):
        if "output" not in case:
            continue
        output = transform(case["input"])
        if output == case["output"]:
            passed += 1
        elif not first_failure:
            first_failure = f"case {index} mismatch"
    return passed, len(cases), first_failure


def probe_task(task_id: str, task: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, (formula, transform, builder_possible, risk_level) in PROBES.items():
        train_passed, train_total, train_failure = _score_cases(task.get("train", []), transform)
        test_passed, test_total, test_failure = _score_cases(task.get("test", []), transform)
        arc_passed, arc_total, arc_failure = _score_cases(task.get("arc-gen", []), transform)
        failures = [item for item in (train_failure, test_failure, arc_failure) if item]
        rows.append(
            {
                "task_id": task_id,
                "probe_name": name,
                "matched_train": train_passed,
                "matched_test": test_passed,
                "matched_arc_gen": arc_passed,
                "total_train": train_total,
                "total_test": test_total,
                "total_arc_gen": arc_total,
                "formula": formula,
                "uncertainty": "POSSIBLE" if train_passed == train_total else "REJECT",
                "builder_possible": builder_possible,
                "risk_level": risk_level,
                "failure_reason": "; ".join(failures),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="task")
    parser.add_argument("--task-ids", required=True)
    parser.add_argument("--report", default="outputs/reports/high_risk_ablation_probe_report.csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for task_id in [item.strip() for item in args.task_ids.split(",") if item.strip()]:
        task_path = Path(args.data_dir) / f"{task_id}.json"
        task = json.loads(task_path.read_text(encoding="utf-8"))
        rows.extend(probe_task(task_id, task))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['task_id']} {row['probe_name']}: "
            f"train {row['matched_train']}/{row['total_train']}, "
            f"test {row['matched_test']}/{row['total_test']}, "
            f"arc-gen {row['matched_arc_gen']}/{row['total_arc_gen']}"
        )
    print(f"probe_report = {report_path}")


if __name__ == "__main__":
    main()
