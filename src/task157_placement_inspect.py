"""Inspect task157 placement-table graph structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper


DEFAULT_MODEL = "outputs/onnx/task157.onnx"


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape) if shape else "scalar"


def inspect_task157_placement(model_path: str = DEFAULT_MODEL) -> dict[str, Any]:
    model = onnx.load(model_path)
    arrays = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    consumers: dict[str, list[dict[str, Any]]] = {}
    node_by_output: dict[str, dict[str, Any]] = {}
    for index, node in enumerate(model.graph.node):
        node_info = {
            "node_index": index,
            "node_name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": {
                attr.name: onnx.helper.get_attribute_value(attr).tolist()
                if hasattr(onnx.helper.get_attribute_value(attr), "tolist")
                else onnx.helper.get_attribute_value(attr)
                for attr in node.attribute
            },
        }
        for output in node.output:
            node_by_output[output] = node_info
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(node_info)

    values = []
    for name in ("plac_idx_963", "expand_idx_983"):
        array = arrays.get(name)
        values.append(
            {
                "name": name,
                "present": array is not None,
                "shape": _shape_text(tuple(array.shape)) if array is not None else "",
                "dtype": str(array.dtype) if array is not None else "",
                "size": int(array.size) if array is not None else 0,
                "nbytes": int(array.nbytes) if array is not None else 0,
                "preview": array.reshape(-1)[:20].astype(int).tolist()
                if array is not None and np.issubdtype(array.dtype, np.integer)
                else [],
            }
        )

    interesting_outputs = set()
    for name in ("plac_idx_963", "expand_idx_983"):
        for consumer in consumers.get(name, []):
            interesting_outputs.update(consumer["outputs"])

    frontier = set(interesting_outputs)
    downstream = []
    for _ in range(40):
        next_frontier = set()
        for index, node in enumerate(model.graph.node):
            if any(input_name in frontier for input_name in node.input):
                info = {
                    "node_index": index,
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "attributes": {
                        attr.name: onnx.helper.get_attribute_value(attr).tolist()
                        if hasattr(onnx.helper.get_attribute_value(attr), "tolist")
                        else onnx.helper.get_attribute_value(attr)
                        for attr in node.attribute
                    },
                }
                downstream.append(info)
                next_frontier.update(node.output)
        if not next_frontier - frontier:
            break
        frontier.update(next_frontier)

    return {
        "model_path": model_path,
        "values": values,
        "consumers": {
            name: consumers.get(name, [])
            for name in ("plac_idx_963", "expand_idx_983")
        },
        "downstream": downstream[:120],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    print(json.dumps(inspect_task157_placement(args.model), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
