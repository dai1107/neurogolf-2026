"""Runtime validation helpers for generated ONNX models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .encoding import (
    DEFAULT_HEIGHT,
    DEFAULT_SHAPE,
    DEFAULT_WIDTH,
    find_zero_confidence_cells,
    grid_to_onehot,
    onehot_to_grid,
)


def run_model(model_path: str, input_tensor: np.ndarray) -> np.ndarray:
    """Run a generated ONNX model and return its only output tensor."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"model file does not exist: {model_path}")

    input_array = np.asarray(input_tensor)
    if input_array.shape != DEFAULT_SHAPE:
        raise ValueError(f"input tensor shape must be {DEFAULT_SHAPE}, got {input_array.shape}")
    if not np.isfinite(input_array).all():
        raise ValueError("input tensor contains NaN or Inf")

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise ValueError(f"model must have exactly one input, got {len(inputs)}")
    if len(outputs) != 1:
        raise ValueError(f"model must have exactly one output, got {len(outputs)}")

    result = session.run([outputs[0].name], {inputs[0].name: input_array.astype(np.float32, copy=False)})
    output = result[0]
    if output.shape != DEFAULT_SHAPE:
        raise ValueError(f"output tensor shape must be {DEFAULT_SHAPE}, got {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError("output tensor contains NaN or Inf")
    return output


def _grid_size(grid: list[list[int]]) -> tuple[int, int]:
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


def validate_case(
    model_path: str,
    input_grid: list[list[int]],
    expected_grid: list[list[int]],
) -> dict[str, object]:
    """Validate one ARC training case with exact grid equality."""
    expected_height, expected_width = _grid_size(expected_grid)
    input_tensor = grid_to_onehot(input_grid)
    output_tensor = run_model(model_path, input_tensor)
    actual_grid = onehot_to_grid(output_tensor, expected_height, expected_width)

    mismatches: list[dict[str, int]] = []
    for row_index, row in enumerate(expected_grid):
        for col_index, expected_color in enumerate(row):
            actual_color = actual_grid[row_index][col_index]
            if actual_color != expected_color:
                mismatches.append(
                    {
                        "row": row_index,
                        "col": col_index,
                        "expected": expected_color,
                        "actual": actual_color,
                    }
                )

    zero_confidence_cells = find_zero_confidence_cells(
        output_tensor,
        expected_height,
        expected_width,
    )
    return {
        "passed": not mismatches,
        "num_mismatched_cells": len(mismatches),
        "mismatches": mismatches,
        "zero_confidence_cells": zero_confidence_cells,
        "nonzero_padding_cells": find_nonzero_padding_cells(
            output_tensor,
            expected_height,
            expected_width,
        ),
    }


def find_nonzero_padding_cells(
    tensor: np.ndarray,
    height: int,
    width: int,
    tolerance: float = 1e-6,
) -> list[dict[str, int]]:
    """Report cells outside the expected grid rectangle with nonzero output."""
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

    active = np.zeros((DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=bool)
    active[:height, :width] = True
    cell_max = np.max(np.abs(array[0]), axis=0)
    rows, cols = np.where((~active) & (cell_max > tolerance))
    return [
        {"row": int(row), "col": int(col)}
        for row, col in zip(rows.tolist(), cols.tolist())
    ]


def validate_cases(
    model_path: str,
    cases: list[dict[str, list[list[int]]]],
) -> dict[str, object]:
    """Validate multiple ARC cases and keep per-case failure details."""
    case_results = []
    for case_index, case in enumerate(cases):
        result = validate_case(model_path, case["input"], case["output"])
        case_results.append({"case_index": case_index, **result})

    failed_cases = [case for case in case_results if not case["passed"]]
    return {
        "passed": not failed_cases,
        "num_cases": len(cases),
        "num_failed_cases": len(failed_cases),
        "failed_cases": failed_cases,
        "case_results": case_results,
    }


def validate_task(model_path: str, task: dict) -> dict[str, object]:
    """Validate an ONNX model against every train case in a task."""
    cases = task.get("train")
    if not isinstance(cases, list):
        raise ValueError("task must contain a train list")

    case_results = []
    failed_cases = []
    for case_index, case in enumerate(cases):
        result = validate_case(model_path, case["input"], case["output"])
        reason = None
        if not result["passed"]:
            reason = "grid_mismatch"
        elif result["zero_confidence_cells"]:
            reason = "zero_confidence_cells"
        elif result["nonzero_padding_cells"]:
            reason = "nonzero_padding_cells"

        case_result = {"case_index": case_index, **result}
        case_results.append(case_result)
        if reason is not None:
            failed_cases.append({"case_index": case_index, "reason": reason, **result})

    return {
        "passed": not failed_cases,
        "num_cases": len(cases),
        "failed_cases": failed_cases,
        "case_results": case_results,
    }
