"""Prune task101 template connected-component expansion radius.

The task101 override grows the source template component through 15 repeated
cross-convolution steps.  Labelled train/test/arc-gen validation shows radius 2
is sufficient for the generated distribution, while radius 1 is not.  This
module rewires ``template_mask`` to an earlier ``R_XX`` tensor and removes the
now-dead expansion nodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import onnx

from .hybrid_stack_optimizer import _prune_dead_graph


DEFAULT_SOURCE = "outputs/reference_6348_56_stack/overrides/task101.onnx"
DEFAULT_OUTPUT = "outputs/candidates/task101_radius_prune/task101_TemplateRadius02.onnx"
DEFAULT_RADIUS = 2


def _radius_value_name(radius: int) -> str:
    if radius < 1 or radius > 15:
        raise ValueError(f"radius must be in 1..15, got {radius}")
    return f"R_{radius:02d}"


def prune_task101_template_radius(
    source_model: str = DEFAULT_SOURCE,
    output_model: str = DEFAULT_OUTPUT,
    radius: int = DEFAULT_RADIUS,
) -> dict[str, Any]:
    """Write a task101 candidate using an earlier template expansion radius."""
    source_path = Path(source_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")

    radius_value = _radius_value_name(radius)
    model = onnx.load(str(source_path))
    rewired = False
    for node in model.graph.node:
        if list(node.output) == ["template_mask"]:
            node.input[0] = radius_value
            rewired = True
            break
    if not rewired:
        raise ValueError("source model does not contain template_mask node")

    del model.graph.value_info[:]
    stats = _prune_dead_graph(model)
    onnx.checker.check_model(model)

    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path))

    summary = {
        "source_model": str(source_path),
        "output_model": str(output_path),
        "radius": radius,
        "radius_value": radius_value,
        "removed_dead_nodes": stats.removed_dead_nodes,
        "removed_unused_initializers": stats.removed_unused_initializers,
        "removed_unused_value_info": stats.removed_unused_value_info,
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS)
    args = parser.parse_args()
    prune_task101_template_radius(
        source_model=args.source,
        output_model=args.output,
        radius=args.radius,
    )


if __name__ == "__main__":
    main()
