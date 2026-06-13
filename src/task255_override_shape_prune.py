"""Prune unused task255 override max-rectangle shape branches.

The 6348.56 task255 override searches for the largest all-zero rectangle using
fixed Conv kernels.  The labelled train/test/arc-gen distribution only needs
the corridor-size family with one side in 6..12 and the other side in 26 or 30.
This module keeps that conservative family, rebuilds the max-reduction chains,
and removes now-unreferenced initializers.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import onnx


DEFAULT_SOURCE = "outputs/reference_6348_56_stack/overrides/task255.onnx"
DEFAULT_OUTPUT = "outputs/candidates/task255_override_shape_prune/task255_ShapePruned6To12.onnx"
DEFAULT_KEEP_SHORT_MIN = 6
DEFAULT_KEEP_SHORT_MAX = 12
DEFAULT_LONG_SIDES = (26, 30)

SHAPE_RE = re.compile(r"_(\d+)x(\d+)(?:$|_)")


def _shape_suffixes(node: onnx.NodeProto) -> set[tuple[int, int]]:
    suffixes: set[tuple[int, int]] = set()
    for name in list(node.input) + list(node.output):
        for match in SHAPE_RE.finditer(name):
            suffixes.add((int(match.group(1)), int(match.group(2))))
    return suffixes


def _shape_sort_key(shape: tuple[int, int]) -> tuple[int, int]:
    return shape


def conservative_keep_shapes(
    short_min: int = DEFAULT_KEEP_SHORT_MIN,
    short_max: int = DEFAULT_KEEP_SHORT_MAX,
    long_sides: tuple[int, ...] = DEFAULT_LONG_SIDES,
) -> set[tuple[int, int]]:
    """Return task255 corridor search shapes kept by the conservative pruner."""
    if not long_sides:
        raise ValueError("long_sides must not be empty")
    keep: set[tuple[int, int]] = set()
    for short in range(short_min, short_max + 1):
        for long_side in long_sides:
            keep.add((short, long_side))
            keep.add((long_side, short))
    return keep


def _is_old_acc_max_node(node: onnx.NodeProto) -> bool:
    return any(output.startswith("acc_max_") for output in node.output)


def _is_old_extent_max_node(node: onnx.NodeProto) -> bool:
    return any(output.startswith("acc_extent_") for output in node.output)


def _is_branch_node(node: onnx.NodeProto, all_shapes: set[tuple[int, int]]) -> bool:
    return bool(_shape_suffixes(node) & all_shapes)


def _is_dropped_branch_node(
    node: onnx.NodeProto,
    all_shapes: set[tuple[int, int]],
    keep_shapes: set[tuple[int, int]],
) -> bool:
    suffixes = _shape_suffixes(node)
    if not suffixes:
        return False
    branch_suffixes = suffixes & all_shapes
    return bool(branch_suffixes) and not branch_suffixes <= keep_shapes


def _make_max_chain(inputs: list[str], final_output: str, prefix: str) -> list[onnx.NodeProto]:
    if not inputs:
        raise ValueError(f"cannot build {prefix} chain with no inputs")
    if len(inputs) == 1:
        return [onnx.helper.make_node("Identity", [inputs[0]], [final_output], name=final_output)]

    nodes: list[onnx.NodeProto] = []
    current = inputs[0]
    for index, next_input in enumerate(inputs[1:]):
        output = final_output if index == len(inputs) - 2 else f"{prefix}_{index}"
        nodes.append(
            onnx.helper.make_node(
                "Max",
                [current, next_input],
                [output],
                name=f"{prefix}_{index}",
            )
        )
        current = output
    return nodes


def _remove_unreferenced_initializers(model: onnx.ModelProto) -> list[str]:
    referenced = {name for node in model.graph.node for name in node.input if name}
    referenced.update(value.name for value in model.graph.input)
    kept = []
    removed = []
    for initializer in model.graph.initializer:
        if initializer.name in referenced:
            kept.append(initializer)
        else:
            removed.append(initializer.name)
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept)
    return removed


def prune_task255_override_shapes(
    source_model: str = DEFAULT_SOURCE,
    output_model: str = DEFAULT_OUTPUT,
    short_min: int = DEFAULT_KEEP_SHORT_MIN,
    short_max: int = DEFAULT_KEEP_SHORT_MAX,
    long_sides: tuple[int, ...] = DEFAULT_LONG_SIDES,
) -> dict[str, Any]:
    """Write a task255 override candidate with unused shape branches removed."""
    source_path = Path(source_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")
    if short_min <= 0 or short_max < short_min:
        raise ValueError("short_min/short_max must define a positive inclusive range")
    if not long_sides or any(side <= 0 for side in long_sides):
        raise ValueError("long_sides must be positive integers")

    model = onnx.load(str(source_path))
    all_shapes: set[tuple[int, int]] = set()
    for node in model.graph.node:
        all_shapes.update(_shape_suffixes(node))
    all_shapes = {shape for shape in all_shapes if 1 <= shape[0] <= 30 and 1 <= shape[1] <= 30}
    keep_shapes = conservative_keep_shapes(short_min, short_max, long_sides) & all_shapes
    dropped_shapes = all_shapes - keep_shapes
    if not keep_shapes:
        raise ValueError("no task255 shape branches would be kept")

    score_inputs = [f"score_max_{h}x{w}" for h, w in sorted(keep_shapes, key=_shape_sort_key)]
    extent_inputs = [f"mark_full_{h}x{w}" for h, w in sorted(keep_shapes, key=_shape_sort_key)]
    acc_chain = _make_max_chain(score_inputs, "acc_max_50", "acc_max_pruned")
    extent_chain = _make_max_chain(extent_inputs, "acc_extent_50", "acc_extent_pruned")

    new_nodes: list[onnx.NodeProto] = []
    inserted_acc = False
    inserted_extent = False
    removed_branch_nodes = 0
    removed_chain_nodes = 0

    for node in model.graph.node:
        if _is_old_acc_max_node(node):
            removed_chain_nodes += 1
            continue
        if _is_old_extent_max_node(node):
            removed_chain_nodes += 1
            continue
        if _is_dropped_branch_node(node, all_shapes, keep_shapes):
            removed_branch_nodes += 1
            continue

        if not inserted_acc and node.op_type == "Equal" and any(
            input_name == "acc_max_50" for input_name in node.input
        ):
            new_nodes.extend(acc_chain)
            inserted_acc = True
        if not inserted_extent and any(input_name == "acc_extent_50" for input_name in node.input):
            new_nodes.extend(extent_chain)
            inserted_extent = True
        new_nodes.append(node)

    if not inserted_acc:
        raise ValueError("failed to insert pruned score max chain")
    if not inserted_extent:
        raise ValueError("failed to insert pruned extent max chain")

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.value_info[:]
    removed_initializers = _remove_unreferenced_initializers(model)
    onnx.checker.check_model(model)

    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path))

    summary = {
        "source_model": str(source_path),
        "output_model": str(output_path),
        "short_min": short_min,
        "short_max": short_max,
        "long_sides": list(long_sides),
        "kept_shapes": [f"{h}x{w}" for h, w in sorted(keep_shapes, key=_shape_sort_key)],
        "dropped_shapes": [f"{h}x{w}" for h, w in sorted(dropped_shapes, key=_shape_sort_key)],
        "removed_branch_nodes": removed_branch_nodes,
        "removed_chain_nodes": removed_chain_nodes,
        "removed_initializers": len(removed_initializers),
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--short-min", type=int, default=DEFAULT_KEEP_SHORT_MIN)
    parser.add_argument("--short-max", type=int, default=DEFAULT_KEEP_SHORT_MAX)
    parser.add_argument(
        "--long-sides",
        default=",".join(str(side) for side in DEFAULT_LONG_SIDES),
        help="Comma-separated long-side lengths to keep, e.g. 26 or 26,30.",
    )
    args = parser.parse_args()
    long_sides = tuple(int(item.strip()) for item in args.long_sides.split(",") if item.strip())
    prune_task255_override_shapes(
        source_model=args.source,
        output_model=args.output,
        short_min=args.short_min,
        short_max=args.short_max,
        long_sides=long_sides,
    )


if __name__ == "__main__":
    main()
