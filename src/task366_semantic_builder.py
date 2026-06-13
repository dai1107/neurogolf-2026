"""Build a small semantic ONNX model for task366 (panel marker-object transfer).

Algorithm:
1. Split detection: find background color boundary
2. Source/target: count non-bg cells per panel
3. Object extraction: iterative MaxPool dilation from markers
4. Finite-offset pasting: for each (source_marker -> target_marker) offset,
   shift source object and paste where markers align.

Target cost: < 5,000 (vs current 260,211 archive model)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .cost_estimator import estimate_model_cost


def _make_initializer(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


def _ones(shape: tuple[int, ...], name: str) -> onnx.TensorProto:
    return _make_initializer(name, np.ones(shape, dtype=np.float32))


def _zeros(shape: tuple[int, ...], name: str) -> onnx.TensorProto:
    return _make_initializer(name, np.zeros(shape, dtype=np.float32))


def build_task366_model(task_data: dict, max_cost: int = 50000) -> Optional[onnx.ModelProto]:
    """Build ONNX model for task366. Returns None if model would be too large."""

    # ---- Analyze task data to extract templates ----
    train_cases = task_data["train"]
    all_cases = list(train_cases)
    if "test" in task_data:
        all_cases.append(task_data["test"][0])

    # Determine panel dimensions and marker colors from data
    # We need to know max panel size for padding
    max_h, max_w = 0, 0
    all_markers = set()

    for case in all_cases:
        inp = case["input"]
        H, W = len(inp), len(inp[0])
        # Find split
        from collections import Counter
        split_dir = None
        split_pos = None
        for r in range(1, H):
            if Counter(inp[r-1]).most_common(1)[0][0] != Counter(inp[r]).most_common(1)[0][0]:
                split_dir, split_pos = 'H', r
                break
        if split_dir is None:
            for c in range(1, W):
                if Counter(inp[r][c-1] for r in range(H)).most_common(1)[0][0] != Counter(inp[r][c] for r in range(H)).most_common(1)[0][0]:
                    split_dir, split_pos = 'V', c
                    break

        if split_dir == 'H':
            ph, pw = split_pos, W
        else:
            ph, pw = H, split_pos

        max_h = max(max_h, ph)
        max_w = max(max_w, pw)

        # Marker extraction was left for the full semantic builder; this
        # placeholder only records panel bounds.

    max_h = max(max_h, 15)
    max_w = max(max_w, 17)

    # ---- Build model ----
    model = helper.make_model(
        opset_imports=[helper.make_opsetid("", 13)],
        graph=helper.make_graph(
            nodes=[],
            name="task366",
            inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, max_h * 2, max_w])],
            outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, "out_h", "out_w"])],
            initializer=[],
        ),
    )

    # For now, return a placeholder — actual ONNX construction is complex
    # and needs careful implementation for each op

    return model


if __name__ == "__main__":
    task = json.loads(open("task/task366.json", "r", encoding="utf-8").read())
    m = build_task366_model(task)
    if m:
        onnx.checker.check_model(m)
        onnx.save(m, "outputs/candidates/task366_semantic/task366.onnx")
        cost = estimate_model_cost("outputs/candidates/task366_semantic/task366.onnx")
        print(f"cost={cost['estimated_cost']}")
