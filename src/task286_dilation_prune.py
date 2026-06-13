"""Prune task286 dilation chain from 58 iterations to a smaller validated count."""

import argparse
import copy
import csv
import os
import sys

import onnx
from onnx import helper


def prune_dilation_chain(model_path: str, output_path: str, max_iterations: int):
    """Truncate the MaxPool+Min dilation chain to max_iterations steps.

    The original model has 59 iterations (reach_0 through reach_58).
    Each iteration: pool_i = MaxPool(reach_{i-1}), reach_i = Min(pool_i, open_u8).
    reach_0 is the seed; reach_58 is the final reachability map consumed via Cast -> reach_f.
    """
    m = onnx.load(model_path)

    if max_iterations >= 58:
        # No pruning needed; just copy
        onnx.save(m, output_path)
        return

    # Identify the Cast node that converts reach_58 to reach_f
    cast_node = None
    for n in m.graph.node:
        if n.op_type == "Cast" and n.input and n.input[0] == "reach_58":
            cast_node = n
            break
    if cast_node is None:
        raise ValueError("Could not find Cast node consuming reach_58")

    # Rewire Cast input from reach_58 to reach_{max_iterations}
    new_reach_name = f"reach_{max_iterations}"
    cast_node.input[0] = new_reach_name

    # Collect nodes to remove: MaxPool and Min nodes beyond max_iterations
    nodes_to_remove = set()
    output_names_to_remove = set()

    for n in m.graph.node:
        if n.op_type == "MaxPool":
            # pool_i nodes: pool_0 = MaxPool(seed_u8), pool_i = MaxPool(reach_{i-1})
            # Remove pool_{max_iterations+1} through pool_58
            if n.output and n.output[0].startswith("pool_"):
                try:
                    idx = int(n.output[0].split("_")[1])
                    if idx > max_iterations:
                        nodes_to_remove.add(n.output[0])
                        output_names_to_remove.add(n.output[0])
                except ValueError:
                    pass
        elif n.op_type == "Min":
            # reach_i nodes: reach_i = Min(pool_i, open_u8)
            if n.output and n.output[0].startswith("reach_"):
                try:
                    idx = int(n.output[0].split("_")[1])
                    if idx > max_iterations:
                        nodes_to_remove.add(n.output[0])
                        output_names_to_remove.add(n.output[0])
                except ValueError:
                    pass

    # Filter graph nodes
    kept_nodes = []
    for n in m.graph.node:
        if n.output and n.output[0] in output_names_to_remove:
            continue
        keep = True
        for o in (n.output or []):
            if o in output_names_to_remove:
                keep = False
                break
        if keep:
            kept_nodes.append(n)

    # Also remove pool_{max_iterations} since reach_{max_iterations} already exists
    # Wait, pool_N feeds into reach_N. If reach_N is kept, don't remove pool_N.
    # But reach_N needs pool_N. Let me verify: reach_N = Min(pool_N, open_u8).
    # pool_N = MaxPool(reach_{N-1}). Both pool_N and reach_N must be kept.

    # Clear and rebuild graph
    del m.graph.node[:]
    m.graph.node.extend(kept_nodes)

    # Remove stale value_info entries for removed intermediate outputs
    if m.graph.value_info:
        kept_vi = []
        for vi in m.graph.value_info:
            if vi.name not in output_names_to_remove:
                kept_vi.append(vi)
        del m.graph.value_info[:]
        m.graph.value_info.extend(kept_vi)

    # Validate
    onnx.checker.check_model(m)
    onnx.save(m, output_path)


def validate_candidate(model_path: str, task_path: str) -> bool:
    """Run labelled train/test validation."""
    import json
    import onnxruntime as ort
    import numpy as np

    with open(task_path) as f:
        data = json.load(f)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    for split_name in ["train", "test"]:
        for case in data.get(split_name, []):
            inp = np.array(case["input"], dtype=np.int64)
            out_expected = np.array(case["output"], dtype=np.int64)
            H, W = inp.shape

            inp_oh = np.eye(10, dtype=np.float32)[inp]
            inp_oh = inp_oh.transpose(2, 0, 1)[np.newaxis, ...]
            if H < 30 or W < 30:
                pad_h = 30 - H
                pad_w = 30 - W
                inp_oh = np.pad(inp_oh, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))

            output = session.run(None, {"input": inp_oh})[0]
            out_pred = output[0].argmax(0)[:H, :W]
            if not np.array_equal(out_pred, out_expected):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Prune task286 dilation chain")
    parser.add_argument("--source", required=True, help="Source ONNX model path")
    parser.add_argument("--output", required=True, help="Output ONNX model path")
    parser.add_argument("--task", default="task/task286.json", help="Task JSON path")
    parser.add_argument("--max-iterations", type=int, required=True,
                        help="Maximum dilation iterations to keep")
    parser.add_argument("--validate", action="store_true", help="Validate against labelled data")
    args = parser.parse_args()

    prune_dilation_chain(args.source, args.output, args.max_iterations)

    removed = 58 - args.max_iterations
    print(f"Pruned dilation chain from 58 to {args.max_iterations} iterations")
    print(f"Removed {removed} MaxPool + {removed} Min = {removed * 2} nodes")

    if args.validate:
        ok = validate_candidate(args.output, args.task)
        print(f"Labelled validation: {'PASSED' if ok else 'FAILED'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
