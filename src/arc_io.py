"""ARC task loading and validation helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


TASK_ID_RE = re.compile(r"^task(\d{3})$")


def _validate_grid(grid: Any, context: str) -> None:
    if not isinstance(grid, list) or not grid:
        raise ValueError(f"{context} must be a non-empty 2D list")
    width = None
    for row_index, row in enumerate(grid):
        if not isinstance(row, list) or not row:
            raise ValueError(f"{context}[{row_index}] must be a non-empty list")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(
                f"{context} must be rectangular: row 0 has width {width}, "
                f"row {row_index} has width {len(row)}"
            )
        for col_index, color in enumerate(row):
            if isinstance(color, bool) or not isinstance(color, int):
                raise ValueError(f"{context}[{row_index}][{col_index}] must be an integer")
            if color < 0 or color > 9:
                raise ValueError(
                    f"{context}[{row_index}][{col_index}] color {color} is outside 0..9"
                )


def _validate_case(case: Any, context: str, require_output: bool) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"{context} must be an object")
    if "input" not in case:
        raise ValueError(f"{context} is missing input")
    if require_output and "output" not in case:
        raise ValueError(f"{context} is missing output")
    _validate_grid(case["input"], f"{context}.input")
    if "output" in case:
        _validate_grid(case["output"], f"{context}.output")


def load_task(path: str) -> dict:
    """Read and validate one ARC task JSON file."""
    task_path = Path(path)
    if not task_path.is_file():
        raise FileNotFoundError(f"task file does not exist: {path}")

    try:
        with task_path.open("r", encoding="utf-8") as handle:
            task = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc

    if not isinstance(task, dict):
        raise ValueError(f"task JSON must be an object: {path}")
    train = task.get("train")
    if not isinstance(train, list) or not train:
        raise ValueError(f"task must contain a non-empty train list: {path}")
    for index, case in enumerate(train):
        _validate_case(case, f"train[{index}]", require_output=True)

    for split in ("test", "arc-gen"):
        cases = task.get(split)
        if cases is None:
            continue
        if not isinstance(cases, list):
            raise ValueError(f"{split} must be a list in {path}")
        for index, case in enumerate(cases):
            _validate_case(case, f"{split}[{index}]", require_output=False)
    return task


def load_all_tasks(data_dir: str) -> dict[str, dict]:
    """Load all taskXXX.json files from a directory keyed by submission task id."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"task directory does not exist: {data_dir}")

    paths = sorted(root.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no JSON task files found in {data_dir}")

    tasks: dict[str, dict] = {}
    invalid_names = []
    for path in paths:
        match = TASK_ID_RE.match(path.stem)
        if match is None:
            invalid_names.append(path.name)
            continue
        task_number = int(match.group(1))
        if task_number < 1 or task_number > 400:
            invalid_names.append(path.name)
            continue
        task_id = f"task{task_number:03d}"
        tasks[task_id] = load_task(str(path))

    if invalid_names:
        names = ", ".join(invalid_names[:5])
        raise ValueError(
            "found task files without explicit taskXXX numbering; "
            f"provide an official mapping before loading them: {names}"
        )
    return dict(sorted(tasks.items()))
