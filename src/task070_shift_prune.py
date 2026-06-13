"""Prune task070 shift branches.

The model has 20 conv/gt pairs for shift amounts: 6,9,12,15,18,21,24,27,30,33,36,39,42,45,48,51,54,57,61,64
(and 16 cast nodes for the same range minus some).

This script prunes shift amounts above a threshold, removing unused branches.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import onnx
from onnx import helper, checker


SHIFT_AMOUNTS = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57, 61, 64]
CAST_AMOUNTS = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 58, 61]  # slightly different


def prune_shifts(source_path: str, output_path: str, max_shift: int):
    """Remove shift branches with amount > max_shift.

    Each shift branch produces shift_max_bool_mask_N. These are AND'd together
    in a chain to compute the final inside_mask. When removing a branch, we
    replace its mask with a constant True (all 1s) so the AND chain still works.
    """
    m = onnx.load(source_path)
    graph = m.graph

    # Find which shift amounts to keep vs remove
    kept_amounts = {a for a in SHIFT_AMOUNTS if a <= max_shift}
    removed_amounts = {a for a in SHIFT_AMOUNTS if a > max_shift}

    # Build a constant True tensor for removed masks
    true_tensor = onnx.helper.make_tensor(
        name="const_true_mask",
        data_type=onnx.TensorProto.BOOL,
        dims=[1, 1, 30, 30],
        vals=[True] * 900
    )
    graph.initializer.append(true_tensor)

    # Find the mask-producing nodes for removed amounts
    removed_mask_outputs = set()
    mask_producer_map = {}  # mask_name -> producing node

    for n in graph.node:
        m_mask = re.search(r'shift_max_bool_mask_(\d+)$', n.name)
        if m_mask:
            amount = int(m_mask.group(1))
            if amount in removed_amounts and n.output:
                removed_mask_outputs.add(n.output[0])
        # Also find conv/count/gt nodes for removed amounts
        m_any = re.search(r'shift_max_bool_(?:conv|count|gt|cast)_(\d+)$', n.name)
        if m_any:
            amount = int(m_any.group(1))
            if amount in removed_amounts:
                for o in n.output:
                    removed_mask_outputs.add(o)

    # Remove nodes that produce removed outputs
    nodes_to_keep = []
    for n in graph.node:
        should_remove = False
        for o in n.output:
            if o in removed_mask_outputs:
                should_remove = True
                break
        if should_remove:
            continue
        nodes_to_keep.append(n)

    # Replace references to removed masks with const_true_mask
    for n in nodes_to_keep:
        new_inputs = []
        for inp in n.input:
            if inp in removed_mask_outputs:
                new_inputs.append("const_true_mask")
            else:
                new_inputs.append(inp)
        n.input[:] = new_inputs

    print(f"Removed {len(graph.node) - len(nodes_to_keep)} nodes (shift > {max_shift})")

    # Cleanup unused initializers
    all_inputs = set()
    for n in nodes_to_keep:
        for inp in n.input:
            all_inputs.add(inp)
    kept_inits = [i for i in graph.initializer if i.name in all_inputs]

    # Replace
    del graph.node[:]
    graph.node.extend(nodes_to_keep)
    del graph.initializer[:]
    graph.initializer.extend(kept_inits)

    # Clean value_info
    if graph.value_info:
        kept_vi = [vi for vi in graph.value_info if vi.name not in removed_mask_outputs]
        del graph.value_info[:]
        graph.value_info.extend(kept_vi)

    checker.check_model(m)
    onnx.save(m, output_path)


def validate_model(model_path: str, task_path: str) -> bool:
    import onnxruntime as ort

    with open(task_path) as f:
        data = json.load(f)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    all_cases = []
    for split_name in ["train", "test"]:
        for i, case in enumerate(data.get(split_name, [])):
            all_cases.append((split_name, i, case))

    # Add arc-gen
    for key in data:
        if key not in ("train", "test") and isinstance(data[key], list):
            for i, case in enumerate(data[key]):
                all_cases.append(("arc-gen", i, case))

    for split_name, i, case in all_cases:
        inp = np.array(case["input"], dtype=np.int64)
        out_expected = np.array(case["output"], dtype=np.int64)
        H, W = inp.shape
        if H > 30 or W > 30:
            continue

        inp_oh = np.eye(10, dtype=np.float32)[inp]
        inp_oh = inp_oh.transpose(2, 0, 1)[np.newaxis, ...]
        if H < 30 or W < 30:
            inp_oh = np.pad(inp_oh, ((0, 0), (0, 0), (0, 30 - H), (0, 30 - W)))

        output = session.run(None, {"input": inp_oh})[0]
        out_pred = output[0].argmax(0)[:H, :W]
        if not np.array_equal(out_pred, out_expected):
            print(f"  {split_name}[{i}]: FAIL")
            return False
    print(f"  All {len(all_cases)} cases passed")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/current_6349_78_stack/overrides/task070.onnx")
    parser.add_argument("--output-dir", default="outputs/candidates/task070_shift_prune")
    parser.add_argument("--task", default="task/task070.json")
    parser.add_argument("--max-shift", type=int, default=64)
    parser.add_argument("--sweep", action="store_true", help="Sweep max_shift values")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep:
        for ms in SHIFT_AMOUNTS:
            out_path = os.path.join(args.output_dir, f"task070_MaxShift{ms}.onnx")
            prune_shifts(args.source, out_path, ms)
            ok = validate_model(out_path, args.task)
            status = "PASS" if ok else "FAIL"
            print(f"  max_shift={ms}: {status}")
            print()
    else:
        out_path = os.path.join(args.output_dir, f"task070_MaxShift{args.max_shift}.onnx")
        prune_shifts(args.source, out_path, args.max_shift)
        ok = validate_model(out_path, args.task)
        status = "PASSED" if ok else "FAILED"
        print(f"Validation: {status}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
