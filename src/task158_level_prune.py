"""Prune task158 template levels 2 and 3.

Removes the 8 ConvTranspose nodes (4 directions × 2 levels) and their
upstream dependencies, keeping only level-1 (3×3) template matching.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import onnx
from onnx import helper, checker


def prune_levels(source_path: str, output_path: str):
    """Remove levels 2 and 3 from the task158 model.

    Strategy: only remove the ConvTranspose nodes for levels 2/3,
    update the Max combine node, then let the ONNX framework handle
    dead-code cleanup naturally.
    """
    m = onnx.load(source_path)
    graph = m.graph

    # 1. Remove ConvTranspose nodes for levels 2 and 3
    removed_outputs = set()
    nodes_to_keep = []
    for n in graph.node:
        if n.op_type == "ConvTranspose":
            for o in n.output:
                if o.endswith("_2") or o.endswith("_3"):
                    removed_outputs.add(o)
                    break
            else:
                nodes_to_keep.append(n)
                continue
            # Skip level 2/3 ConvTranspose
        else:
            nodes_to_keep.append(n)

    print(f"Removed {len(graph.node) - len(nodes_to_keep)} ConvTranspose nodes")

    # 2. Update the Max node to only take level-1 inputs
    level1_paints = [
        "paint_crop_tlbr_1",
        "paint_crop_brtl_1",
        "paint_crop_trbl_1",
        "paint_crop_bltr_1",
    ]
    for n in nodes_to_keep:
        if n.op_type == "Max" and "paint_val" in n.output:
            old_count = len(n.input)
            n.input[:] = level1_paints
            print(f"Updated Max node: {old_count} -> {len(n.input)} inputs")
            break

    # 3. Remove level 2/3 kernel initializers
    removed_init_patterns = [
        "ker_tlbr_2", "ker_brtl_2", "ker_trbl_2", "ker_bltr_2",
        "ker_tlbr_3", "ker_brtl_3", "ker_trbl_3", "ker_bltr_3",
    ]
    kept_inits = []
    for init in graph.initializer:
        if init.name in removed_init_patterns:
            print(f"  Removing initializer: {init.name}")
            continue
        kept_inits.append(init)

    # 4. Iteratively remove dead nodes (nodes whose outputs have no consumers)
    changed = True
    while changed:
        changed = False
        # Compute consumer sets
        consumers = set()
        for n in nodes_to_keep:
            for inp in n.input:
                consumers.add(inp)
        # Also add graph outputs as consumers
        for go in graph.output:
            consumers.add(go.name)

        new_nodes = []
        for n in nodes_to_keep:
            is_dead = True
            for o in n.output:
                if o in consumers:
                    is_dead = False
                    break
            if is_dead and n.output and all(o not in consumers for o in n.output):
                for o in n.output:
                    removed_outputs.add(o)
                changed = True
                continue
            new_nodes.append(n)
        nodes_to_keep = new_nodes

    print(f"Dead nodes removed: final {len(nodes_to_keep)} nodes")

    # 5. Also remove initializers that are no longer referenced
    all_inputs = set()
    for n in nodes_to_keep:
        for inp in n.input:
            all_inputs.add(inp)
    final_inits = []
    for init in kept_inits:
        if init.name in all_inputs:
            final_inits.append(init)

    # 6. Replace graph
    del graph.node[:]
    graph.node.extend(nodes_to_keep)

    del graph.initializer[:]
    graph.initializer.extend(final_inits)

    # Clean value_info for removed outputs
    if graph.value_info:
        kept_vi = [vi for vi in graph.value_info if vi.name not in removed_outputs]
        del graph.value_info[:]
        graph.value_info.extend(kept_vi)

    # Validate
    checker.check_model(m)
    onnx.save(m, output_path)

    print(f"Final: {len(nodes_to_keep)} nodes, {len(final_inits)} initializers")


def validate_model(model_path: str, task_path: str) -> bool:
    """Run labelled train/test/arc-gen validation."""
    import onnxruntime as ort

    with open(task_path) as f:
        data = json.load(f)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    all_splits = [("train", data.get("train", []))]
    if "test" in data:
        all_splits.append(("test", data.get("test", [])))

    # Also include arc-gen if available
    arc_cases = []
    for key in data:
        if key not in ("train", "test") and isinstance(data[key], list):
            arc_cases.extend(data[key])
    if arc_cases:
        all_splits.append(("arc-gen", arc_cases))

    for split_name, cases in all_splits:
        for i, case in enumerate(cases):
            inp = np.array(case["input"], dtype=np.int64)
            out_expected = np.array(case["output"], dtype=np.int64)
            H, W = inp.shape
            if H > 30 or W > 30:
                continue

            inp_oh = np.eye(10, dtype=np.float32)[inp]
            inp_oh = inp_oh.transpose(2, 0, 1)[np.newaxis, ...]
            if H < 30 or W < 30:
                pad_h = 30 - H
                pad_w = 30 - W
                inp_oh = np.pad(inp_oh, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)))

            output = session.run(None, {"input": inp_oh})[0]
            out_pred = output[0].argmax(0)[:H, :W]
            if not np.array_equal(out_pred, out_expected):
                print(f"  {split_name}[{i}]: FAIL")
                return False
        print(f"  {split_name}: all {len(cases)} passed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prune task158 levels")
    parser.add_argument("--source", default="outputs/current_6349_78_stack/overrides/task158.onnx")
    parser.add_argument("--output", default="outputs/candidates/task158_level_prune/task158_Level1Only.onnx")
    parser.add_argument("--task", default="task/task158.json")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    prune_levels(args.source, args.output)

    if args.validate:
        ok = validate_model(args.output, args.task)
        print(f"\nValidation: {'PASSED' if ok else 'FAILED'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
