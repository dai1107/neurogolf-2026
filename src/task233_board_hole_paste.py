"""Probe task233 board-hole paste semantics before ONNX generation."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, deque
from pathlib import Path
from typing import Iterable


Grid = list[list[int]]

FIELDS = [
    "task_id",
    "split",
    "case_index",
    "passed",
    "input_shape",
    "output_shape",
    "board_bbox",
    "num_holes",
    "num_templates",
    "failure_reason",
]


def _neighbors(row: int, col: int, height: int, width: int) -> Iterable[tuple[int, int]]:
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = row + dr
        nc = col + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _components_where(grid: Grid, predicate) -> list[list[tuple[int, int]]]:
    height = len(grid)
    width = len(grid[0])
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    for row in range(height):
        for col in range(width):
            if (row, col) in seen or not predicate(row, col):
                continue
            queue = deque([(row, col)])
            seen.add((row, col))
            component: list[tuple[int, int]] = []
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in _neighbors(*current, height, width):
                    nr, nc = neighbor
                    if neighbor not in seen and predicate(nr, nc):
                        seen.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
    return components


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    materialized = list(cells)
    return (
        min(row for row, _ in materialized),
        min(col for _, col in materialized),
        max(row for row, _ in materialized),
        max(col for _, col in materialized),
    )


def _crop(grid: Grid, bbox: tuple[int, int, int, int]) -> Grid:
    top, left, bottom, right = bbox
    return [row[left : right + 1] for row in grid[top : bottom + 1]]


def _largest_board_bbox(grid: Grid, board_color: int) -> tuple[int, int, int, int] | None:
    components = _components_where(grid, lambda row, col: grid[row][col] == board_color)
    if not components:
        return None
    components.sort(key=lambda comp: (-len(comp), _bbox(comp)))
    board = components[0]
    if len(board) < 12:
        return None
    return _bbox(board)


def _hole_components(board: Grid, board_color: int) -> list[dict[str, object]]:
    holes: list[dict[str, object]] = []
    height = len(board)
    width = len(board[0])
    zero_components = _components_where(board, lambda row, col: board[row][col] == 0)
    for component in zero_components:
        if any(row in {0, height - 1} or col in {0, width - 1} for row, col in component):
            continue
        top, left, bottom, right = _bbox(component)
        holes.append(
            {
                "bbox": (top, left, bottom, right),
                "shape": frozenset((row - top, col - left) for row, col in component),
                "height": bottom - top + 1,
                "width": right - left + 1,
            }
        )
    return holes


def _outside_board(
    row: int,
    col: int,
    board_bbox: tuple[int, int, int, int],
) -> bool:
    top, left, bottom, right = board_bbox
    return row < top or row > bottom or col < left or col > right


def _template_components(
    grid: Grid,
    board_bbox: tuple[int, int, int, int],
    board_color: int,
) -> list[dict[str, object]]:
    components = _components_where(
        grid,
        lambda row, col: grid[row][col] != 0 and _outside_board(row, col, board_bbox),
    )
    templates: list[dict[str, object]] = []
    for component in components:
        colors = {grid[row][col] for row, col in component}
        if board_color not in colors or colors <= {board_color}:
            continue
        top, left, bottom, right = _bbox(component)
        height = bottom - top + 1
        width = right - left + 1
        component_cells = set(component)
        matrix = [
            [
                grid[top + row][left + col]
                if (top + row, left + col) in component_cells
                else 0
                for col in range(width)
            ]
            for row in range(height)
        ]
        templates.append(
            {
                "bbox": (top, left, bottom, right),
                "height": height,
                "width": width,
                "matrix": matrix,
            }
        )
    return templates


def _transform_cell(name: str, row: int, col: int, height: int, width: int) -> tuple[int, int]:
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


def _transform_matrix(matrix: Grid, name: str) -> Grid:
    height = len(matrix)
    width = len(matrix[0])
    positions = [
        _transform_cell(name, row, col, height, width)
        for row in range(height)
        for col in range(width)
    ]
    min_row = min(row for row, _ in positions)
    min_col = min(col for _, col in positions)
    max_row = max(row for row, _ in positions)
    max_col = max(col for _, col in positions)
    output = [[0 for _ in range(max_col - min_col + 1)] for _ in range(max_row - min_row + 1)]
    for row in range(height):
        for col in range(width):
            new_row, new_col = _transform_cell(name, row, col, height, width)
            output[new_row - min_row][new_col - min_col] = matrix[row][col]
    return output


def _board_color_candidates(grid: Grid) -> list[int]:
    counts = Counter(color for row in grid for color in row if color != 0)
    return [color for color, _ in counts.most_common()]


def board_hole_paste_transform(grid: Grid) -> Grid:
    """Return task233 output by pasting external templates into board holes."""
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

    best: tuple[int, Grid] | None = None
    for board_color in _board_color_candidates(grid):
        board_bbox = _largest_board_bbox(grid, board_color)
        if board_bbox is None:
            continue
        board = _crop(grid, board_bbox)
        holes = _hole_components(board, board_color)
        templates = _template_components(grid, board_bbox, board_color)
        if not holes or not templates:
            continue

        output = [[board_color if color == 0 else color for color in row] for row in board]
        placements = _candidate_template_placements(board, templates, board_color, transform_names)
        used_templates: set[int] = set()
        covered_zero_cells: set[tuple[int, int]] = set()
        occupied_cells: set[tuple[int, int]] = set()
        accepted = 0
        for placement in placements:
            template_index = int(placement["template_index"])
            zero_cells = placement["zero_cells"]  # type: ignore[assignment]
            non_board_cells = placement["non_board_cells"]  # type: ignore[assignment]
            if template_index in used_templates:
                continue
            if zero_cells & covered_zero_cells:  # type: ignore[operator]
                continue
            if any((row, col) in occupied_cells for row, col, _ in non_board_cells):  # type: ignore[union-attr]
                continue
            used_templates.add(template_index)
            covered_zero_cells.update(zero_cells)  # type: ignore[arg-type]
            for row, col, color in non_board_cells:  # type: ignore[union-attr]
                output[row][col] = color
                occupied_cells.add((row, col))
            accepted += 1

        score = accepted * 1000 + len(covered_zero_cells) * 10 + len(occupied_cells)
        if best is None or score > best[0]:
            best = (score, output)

    return best[1] if best is not None else [row[:] for row in grid]


def _candidate_template_placements(
    board: Grid,
    templates: list[dict[str, object]],
    board_color: int,
    transform_names: tuple[str, ...],
) -> list[dict[str, object]]:
    board_height = len(board)
    board_width = len(board[0])
    placements: list[dict[str, object]] = []
    for template_index, template in enumerate(templates):
        for transform_name in transform_names:
            matrix = _transform_matrix(template["matrix"], transform_name)  # type: ignore[arg-type]
            height = len(matrix)
            width = len(matrix[0])
            if height > board_height or width > board_width:
                continue
            for top in range(board_height - height + 1):
                for left in range(board_width - width + 1):
                    zero_cells: set[tuple[int, int]] = set()
                    non_board_cells: list[tuple[int, int, int]] = []
                    valid = True
                    for row, line in enumerate(matrix):
                        for col, color in enumerate(line):
                            board_row = top + row
                            board_col = left + col
                            board_cell = board[board_row][board_col]
                            if color == board_color:
                                if board_cell != 0:
                                    valid = False
                                    break
                                zero_cells.add((board_row, board_col))
                            else:
                                if board_cell == 0:
                                    valid = False
                                    break
                                if color != 0:
                                    non_board_cells.append((board_row, board_col, color))
                        if not valid:
                            break
                    if not valid or not zero_cells or not non_board_cells:
                        continue
                    placements.append(
                        {
                            "template_index": template_index,
                            "transform_name": transform_name,
                            "top": top,
                            "left": left,
                            "height": height,
                            "width": width,
                            "border_penalty": int(top == 0)
                            + int(left == 0)
                            + int(top + height == board_height)
                            + int(left + width == board_width),
                            "zero_cells": zero_cells,
                            "non_board_cells": non_board_cells,
                        }
                    )

    return sorted(
        placements,
        key=lambda item: (
            -len(item["zero_cells"]),  # type: ignore[arg-type]
            -len(item["non_board_cells"]),  # type: ignore[arg-type]
            item["border_penalty"],
            item["template_index"],
            item["transform_name"],
            item["top"],
            item["left"],
        ),
    )


def _case_shape(grid: Grid) -> str:
    return f"{len(grid)}x{len(grid[0])}"


def _probe_case(task_id: str, split: str, index: int, case: dict) -> dict[str, object]:
    actual = board_hole_paste_transform(case["input"])
    expected = case.get("output")
    passed = expected == actual if expected is not None else False
    board_bbox = _largest_board_bbox(case["input"], 2)
    holes = []
    templates = []
    if board_bbox is not None:
        board = _crop(case["input"], board_bbox)
        holes = _hole_components(board, 2)
        templates = _template_components(case["input"], board_bbox, 2)
    return {
        "task_id": task_id,
        "split": split,
        "case_index": index,
        "passed": passed,
        "input_shape": _case_shape(case["input"]),
        "output_shape": _case_shape(expected) if expected else "",
        "board_bbox": board_bbox or "",
        "num_holes": len(holes),
        "num_templates": len(templates),
        "failure_reason": "" if passed else "mismatch",
    }


def probe_task(task_id: str, task: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("train", "test", "arc-gen"):
        for index, case in enumerate(task.get(split, [])):
            if "output" not in case:
                continue
            rows.append(_probe_case(task_id, split, index, case))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="task/task233.json")
    parser.add_argument("--task-id", default="task233")
    parser.add_argument("--report", default="outputs/reports/task233_board_hole_paste_probe.csv")
    args = parser.parse_args()

    task = json.loads(Path(args.task).read_text(encoding="utf-8"))
    rows = probe_task(args.task_id, task)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.report).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    passed = sum(1 for row in rows if row["passed"])
    print(f"passed = {passed}/{len(rows)}")
    print(f"report_path = {args.report}")


if __name__ == "__main__":
    main()
