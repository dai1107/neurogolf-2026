"""Grid and one-hot tensor encoding utilities for ARC-style tasks."""

from __future__ import annotations

from typing import Sequence

import numpy as np


DEFAULT_BATCH = 1
DEFAULT_COLORS = 10
DEFAULT_HEIGHT = 30
DEFAULT_WIDTH = 30
DEFAULT_SHAPE = (DEFAULT_BATCH, DEFAULT_COLORS, DEFAULT_HEIGHT, DEFAULT_WIDTH)


def _validate_grid(grid: Sequence[Sequence[int]]) -> tuple[int, int]:
    """Return grid dimensions after checking the grid is rectangular."""
    if not grid:
        raise ValueError("grid must contain at least one row")
    width = len(grid[0])
    if width == 0:
        raise ValueError("grid rows must contain at least one cell")
    for row_index, row in enumerate(grid):
        if len(row) != width:
            raise ValueError(
                f"grid must be rectangular: row 0 has width {width}, "
                f"row {row_index} has width {len(row)}"
            )
    return len(grid), width


def grid_to_onehot(
    grid: list[list[int]],
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_colors: int = DEFAULT_COLORS,
) -> np.ndarray:
    """Encode an ARC grid as a padded NCHW one-hot tensor.

    The source grid is placed in the top-left corner. Padding cells remain all
    zero, which is distinct from real color 0 cells.
    """
    grid_height, grid_width = _validate_grid(grid)
    if height <= 0 or width <= 0 or num_colors <= 0:
        raise ValueError("height, width, and num_colors must be positive")
    if grid_height > height or grid_width > width:
        raise ValueError(
            f"grid shape {grid_height}x{grid_width} exceeds target shape "
            f"{height}x{width}"
        )

    tensor = np.zeros((1, num_colors, height, width), dtype=np.float32)
    for row_index, row in enumerate(grid):
        for col_index, color in enumerate(row):
            if not isinstance(color, (int, np.integer)):
                raise ValueError(
                    f"grid color at ({row_index}, {col_index}) is not an integer: "
                    f"{color!r}"
                )
            color_int = int(color)
            if color_int < 0 or color_int >= num_colors:
                raise ValueError(
                    f"grid color at ({row_index}, {col_index}) is {color_int}; "
                    f"expected 0..{num_colors - 1}"
                )
            tensor[0, color_int, row_index, col_index] = 1.0
    return tensor


def onehot_to_grid(tensor: np.ndarray, height: int, width: int) -> list[list[int]]:
    """Decode a model output tensor to a grid by channel-wise argmax."""
    array = np.asarray(tensor)
    if array.shape != DEFAULT_SHAPE:
        raise ValueError(f"tensor shape must be {DEFAULT_SHAPE}, got {array.shape}")
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if height > DEFAULT_HEIGHT or width > DEFAULT_WIDTH:
        raise ValueError(
            f"requested grid shape {height}x{width} exceeds tensor shape "
            f"{DEFAULT_HEIGHT}x{DEFAULT_WIDTH}"
        )
    if not np.isfinite(array).all():
        raise ValueError("tensor contains NaN or Inf")

    decoded = np.argmax(array[0, :, :height, :width], axis=0)
    return decoded.astype(int).tolist()


def find_zero_confidence_cells(
    tensor: np.ndarray,
    height: int,
    width: int,
    tolerance: float = 1e-6,
) -> list[dict[str, int]]:
    """Report cells whose channels are all effectively zero before argmax."""
    array = np.asarray(tensor)
    if array.shape != DEFAULT_SHAPE:
        raise ValueError(f"tensor shape must be {DEFAULT_SHAPE}, got {array.shape}")
    if height > DEFAULT_HEIGHT or width > DEFAULT_WIDTH:
        raise ValueError(
            f"requested grid shape {height}x{width} exceeds tensor shape "
            f"{DEFAULT_HEIGHT}x{DEFAULT_WIDTH}"
        )
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    cell_max = np.max(np.abs(array[0, :, :height, :width]), axis=0)
    rows, cols = np.where(cell_max <= tolerance)
    return [
        {"row": int(row), "col": int(col)}
        for row, col in zip(rows.tolist(), cols.tolist())
    ]
