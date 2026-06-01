"""Small task-analysis helpers used in logs and reports."""

from __future__ import annotations


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    """Return a rectangular grid's height and width."""
    return len(grid), len(grid[0])


def analyze_task(task: dict) -> dict[str, object]:
    """Return compact train-set facts useful for debugging rule decisions."""
    train = task.get("train", [])
    input_shapes = [grid_shape(case["input"]) for case in train]
    output_shapes = [grid_shape(case["output"]) for case in train]
    colors = sorted(
        {
            color
            for case in train
            for grid in (case["input"], case["output"])
            for row in grid
            for color in row
        }
    )
    return {
        "num_train_cases": len(train),
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "colors": colors,
        "all_input_output_shapes_equal": input_shapes == output_shapes,
    }
