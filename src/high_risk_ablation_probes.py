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


def _copy_marker_matched_source_objects_to_sparse_panel(grid: Grid) -> Grid:
    best_output: Grid | None = None
    best_score = -1

    for first, second in _panel_splits_for_output_shape(grid):
        panel_pairs = ((first, second), (second, first))
        for source, target in panel_pairs:
            if _dominant_fraction(target) < 0.55:
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

                score = len(covered_markers) * 1000 - len(occupied_cells)
                if score > best_score:
                    best_score = score
                    best_output = output

    return best_output if best_output is not None else _copy_grid(grid)


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
    "horizontal_zero_runs_by_marker_length": (
        "Fill horizontal 0 runs whose length equals any horizontal color-2 run length.",
        _fill_horizontal_zero_runs_matching_marker_lengths,
        "yes",
        "medium",
    ),
    "two_panel_marker_object_transfer": (
        "Split the input into two equal panels; copy source-panel objects onto the sparse marker panel when marker-color layouts match.",
        _copy_marker_matched_source_objects_to_sparse_panel,
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
