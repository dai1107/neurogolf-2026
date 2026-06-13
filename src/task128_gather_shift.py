"""Optimize task128: replace (10,30,30) MatMul-based row shift with per-channel Gather.

The current model builds a large M matrix (10,30,30) float16 to permute rows
via MatMul. This accounts for ~36KB of the ~95KB tensor memory cost.

The optimized version uses per-channel Gather operations with small 1D indices,
eliminating the large intermediate permutation matrices.
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, checker


def build_gather_shift_model(source_path: str, output_path: str):
    """Replace MatMul shift with Gather-based shift."""
    m = onnx.load(source_path)

    # Build a new graph with the Gather-based approach
    # We'll rebuild the graph, keeping nodes before the M matrix and replacing
    # the shift computation

    graph = m.graph

    # Find key nodes by role
    nodes_by_output = {n.output[0]: n for n in graph.node if n.output}

    # Find existing nodes we need to keep
    cast_fp16 = None       # Cast(input) -> _fp16_input
    reducemax_4d = None    # ReduceMax -> row_any_4d
    reshape_c = None       # Reshape -> row_any_c
    argmax_y0 = None       # ArgMax -> y0_c
    gather_rev = None      # Gather(reverse_idx) -> row_any_c_rev
    argmax_y1e = None      # ArgMax -> y1_from_end_c
    sub_y1 = None          # Sub -> y1_c
    sub_h = None           # Sub -> h_m1_c
    add_h = None           # Add -> h_c_raw
    reducemax_ch = None    # ReduceMax -> ch_any_2d
    cast_ch = None         # Cast -> ch_any_i
    mul_valid_ch = None    # Mul -> valid_ch_i
    mul_h = None           # Mul -> h_c
    reshape_h_col = None   # Reshape -> h_c_col
    add_row_src = None     # Add -> row_src_2d
    less_valid = None      # Less -> valid_shift_b
    reshape_valid_3d = None  # Reshape -> valid_shift_f_pre_bool
    cast_valid = None      # Cast -> valid_shift_f
    reshape_row_src_3d = None  # Reshape -> row_src_3d
    equal_m = None         # Equal -> M_eq_b
    where_m = None         # Where -> M
    matmul = None          # MatMul -> shifted
    reducemax_grid = None  # ReduceMax -> in_grid
    slice_nonbg = None     # Slice -> shifted_nonbg
    reducesum = None       # ReduceSum -> sum_nonbg
    sub_bg = None          # Sub -> bg_raw
    mul_bg = None          # Mul -> bg_final
    concat_out = None      # Concat -> output

    # Also need constant/initializer nodes
    const_nodes = {}  # output_name -> node for Constant nodes

    for n in graph.node:
        if n.op_type == "Constant":
            for o in n.output:
                const_nodes[o] = n
        if n.op_type == "Cast" and n.output and n.output[0] == "_fp16_input":
            cast_fp16 = n
        elif n.op_type == "ReduceMax" and n.output and n.output[0] == "row_any_4d":
            reducemax_4d = n
        elif n.op_type == "ArgMax" and n.output and n.output[0] == "y0_c":
            argmax_y0 = n
        elif n.op_type == "Gather" and n.output and n.output[0] == "row_any_c_rev":
            gather_rev = n
        elif n.op_type == "ArgMax" and n.output and n.output[0] == "y1_from_end_c":
            argmax_y1e = n
        elif n.op_type == "Sub" and n.output and n.output[0] == "y1_c":
            sub_y1 = n
        elif n.op_type == "Sub" and n.output and n.output[0] == "h_m1_c":
            sub_h = n
        elif n.op_type == "Add" and n.output and n.output[0] == "h_c_raw":
            add_h = n
        elif n.op_type == "ReduceMax" and any("ch_any_2d" in o for o in (n.output or [])):
            reducemax_ch = n
        elif n.op_type == "Cast" and n.output and n.output[0] == "ch_any_i":
            cast_ch = n
        elif n.op_type == "Mul" and n.output and n.output[0] == "valid_ch_i":
            mul_valid_ch = n
        elif n.op_type == "Mul" and n.output and n.output[0] == "h_c":
            mul_h = n
        elif n.op_type == "Reshape" and n.output and n.output[0] == "h_c_col":
            reshape_h_col = n
        elif n.op_type == "Add" and n.output and n.output[0] == "row_src_2d":
            add_row_src = n
        elif n.op_type == "Less" and n.output and n.output[0] == "valid_shift_b":
            less_valid = n
        elif n.op_type == "Reshape" and any("valid_shift_f_pre_bool" in o for o in (n.output or [])):
            reshape_valid_3d = n
        elif n.op_type == "Cast" and n.output and n.output[0] == "valid_shift_f":
            cast_valid = n
        elif n.op_type == "Reshape" and n.output and n.output[0] == "row_src_3d":
            reshape_row_src_3d = n
        elif n.op_type == "Equal" and n.output and n.output[0] == "M_eq_b":
            equal_m = n
        elif n.op_type == "Where" and n.output and n.output[0] == "M":
            where_m = n
        elif n.op_type == "MatMul" and n.output and n.output[0] == "shifted":
            matmul = n
        elif n.op_type == "ReduceMax" and n.output and n.output[0] == "in_grid":
            reducemax_grid = n
        elif n.op_type == "Slice" and n.output and n.output[0] == "shifted_nonbg":
            slice_nonbg = n
        elif n.op_type == "ReduceSum" and n.output and n.output[0] == "sum_nonbg":
            reducesum = n
        elif n.op_type == "Sub" and n.output and n.output[0] == "bg_raw":
            sub_bg = n
        elif n.op_type == "Mul" and n.output and n.output[0] == "bg_final":
            mul_bg = n
        elif n.op_type == "Concat" and n.output and n.output[0] == "output":
            concat_out = n

    # Build new nodes
    new_nodes = []

    # Copy nodes up to and including add_row_src and cast_valid
    # (everything before M matrix construction)
    nodes_to_keep = set()
    nodes_to_remove = set()

    # Keep everything except M-related and MatMul nodes
    m_related = {"M_eq_b", "M", "boolmask_where_20"}
    for n in graph.node:
        if n.output and n.output[0] in m_related:
            nodes_to_remove.add(n.output[0])
        elif n.op_type == "MatMul" and n.output and n.output[0] == "shifted":
            nodes_to_remove.add(n.output[0])

    # Build the per-channel Gather shift
    # 1. Split _fp16_input into 10 channels
    # 2. For each channel, Gather rows using row_src_2d indices
    # 3. Mask invalid shifts
    # 4. Concat back

    # Create the channel split
    # We'll use Split instead of 10 Slice nodes for efficiency
    split_node = helper.make_node(
        "Split",
        inputs=["_fp16_input"],
        outputs=[f"ch_{c}" for c in range(10)],
        axis=1,
        name="split_channels"
    )
    new_nodes.append(split_node)

    # For each channel, create Gather + mask
    ch_shifted_names = []
    for c in range(10):
        ch_name = f"ch_{c}"

        # Extract the c-th row of row_src_2d: (10,30) -> (1,30)
        # Use Slice or Gather
        c_idx_init = helper.make_tensor(
            name=f"c_idx_{c}",
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=[c]
        )
        graph.initializer.append(c_idx_init)

        ch_src_gather = helper.make_node(
            "Gather",
            inputs=["row_src_2d", f"c_idx_{c}"],
            outputs=[f"ch_src_{c}"],
            axis=0,
            name=f"gather_ch_src_{c}"
        )
        new_nodes.append(ch_src_gather)
        # ch_src_{c}: (1, 30) float16

        # Cast float16 → float32 for Clip (opset 10 Clip doesn't support float16)
        cast_to_f32 = helper.make_node(
            "Cast",
            inputs=[f"ch_src_{c}"],
            outputs=[f"ch_src_f32_{c}"],
            to=onnx.TensorProto.FLOAT,
            name=f"cast_ch_src_f32_{c}"
        )
        new_nodes.append(cast_to_f32)

        # Clamp to [0, 29]
        clip_node = helper.make_node(
            "Clip",
            inputs=[f"ch_src_f32_{c}"],
            outputs=[f"ch_src_clamped_{c}"],
            min=0.0,
            max=29.0,
            name=f"clip_ch_src_{c}"
        )
        new_nodes.append(clip_node)

        # Cast to int64 for Gather indices
        cast_src = helper.make_node(
            "Cast",
            inputs=[f"ch_src_clamped_{c}"],
            outputs=[f"ch_src_i64_{c}"],
            to=onnx.TensorProto.INT64,
            name=f"cast_ch_src_{c}"
        )
        new_nodes.append(cast_src)

        # Reshape to (30,) for 1D indices
        reshape_src = helper.make_node(
            "Reshape",
            inputs=[f"ch_src_i64_{c}", "shape_30"],
            outputs=[f"ch_src_1d_{c}"],
            name=f"reshape_ch_src_1d_{c}"
        )
        new_nodes.append(reshape_src)

        # Gather rows from channel slice
        gather_rows = helper.make_node(
            "Gather",
            inputs=[ch_name, f"ch_src_1d_{c}"],
            outputs=[f"ch_shifted_raw_{c}"],
            axis=2,
            name=f"gather_rows_{c}"
        )
        new_nodes.append(gather_rows)
        # ch_shifted_raw_{c}: (1, 1, 30, 30)

        # Get validity mask for this channel
        # valid_shift_f is (10, 30, 1) — need to extract and reshape
        ch_valid_gather = helper.make_node(
            "Gather",
            inputs=["valid_shift_f", f"c_idx_{c}"],
            outputs=[f"ch_valid_{c}"],
            axis=0,
            name=f"gather_ch_valid_{c}"
        )
        new_nodes.append(ch_valid_gather)
        # ch_valid_{c}: (1, 30, 1)

        # Reshape for broadcasting: (1, 30, 1) -> (1, 1, 30, 1)
        reshape_valid = helper.make_node(
            "Reshape",
            inputs=[f"ch_valid_{c}", "shape_1_1_30_1"],
            outputs=[f"ch_valid_4d_{c}"],
            name=f"reshape_ch_valid_{c}"
        )
        new_nodes.append(reshape_valid)

        # Mask: shifted_raw * valid
        mul_mask = helper.make_node(
            "Mul",
            inputs=[f"ch_shifted_raw_{c}", f"ch_valid_4d_{c}"],
            outputs=[f"ch_shifted_{c}"],
            name=f"mul_valid_ch_{c}"
        )
        new_nodes.append(mul_mask)

        ch_shifted_names.append(f"ch_shifted_{c}")

    # Concat all channels back
    concat_shifted = helper.make_node(
        "Concat",
        inputs=ch_shifted_names,
        outputs=["shifted"],
        axis=1,
        name="concat_shifted"
    )
    new_nodes.append(concat_shifted)

    # Add shape initializers needed
    shape_30_init = helper.make_tensor(
        name="shape_30",
        data_type=onnx.TensorProto.INT64,
        dims=[1],
        vals=[30]
    )
    graph.initializer.append(shape_30_init)

    shape_1_1_30_1_init = helper.make_tensor(
        name="shape_1_1_30_1",
        data_type=onnx.TensorProto.INT64,
        dims=[4],
        vals=[1, 1, 30, 1]
    )
    graph.initializer.append(shape_1_1_30_1_init)


    # Now rebuild the full node list
    # Keep all original nodes except the removed ones
    kept_nodes = []
    for n in graph.node:
        should_remove = False
        for o in (n.output or []):
            if o in nodes_to_remove:
                should_remove = True
                break
        # Also remove the MatMul node and its Where/Equal predecessors
        if n.op_type == "MatMul":
            should_remove = True
        if not should_remove:
            kept_nodes.append(n)

    # Add new nodes after existing ones but before the downstream nodes
    # Find where to insert: after add_row_src and friends, before reducemax_grid
    insert_idx = len(kept_nodes)
    for i, n in enumerate(kept_nodes):
        if n.op_type == "ReduceMax" and n.output and n.output[0] == "in_grid":
            insert_idx = i
            break

    final_nodes = kept_nodes[:insert_idx] + new_nodes + kept_nodes[insert_idx:]

    # Replace graph nodes
    del graph.node[:]
    graph.node.extend(final_nodes)

    # Clean up stale value_info
    if graph.value_info:
        kept_vi = []
        remove_outputs = m_related | {"shifted"}  # old shifted will be replaced
        for vi in graph.value_info:
            if vi.name not in remove_outputs:
                kept_vi.append(vi)
        del graph.value_info[:]
        graph.value_info.extend(kept_vi)

    # Validate and save
    checker.check_model(m)
    onnx.save(m, output_path)
    print(f"Saved optimized model to {output_path}")


def validate_model(model_path: str, task_path: str) -> bool:
    """Run labelled train/test validation."""
    import onnxruntime as ort

    with open(task_path) as f:
        data = json.load(f)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    for split_name in ["train", "test"]:
        for i, case in enumerate(data.get(split_name, [])):
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
                print(f"  {split_name}[{i}]: FAIL")
                return False
            print(f"  {split_name}[{i}]: PASS")
    return True


def main():
    parser = argparse.ArgumentParser(description="Optimize task128 matrix shift")
    parser.add_argument("--source", default="outputs/current_6349_78_stack/overrides/task128.onnx")
    parser.add_argument("--output", default="outputs/candidates/task128_gather_shift/task128_GatherShift.onnx")
    parser.add_argument("--task", default="task/task128.json")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    build_gather_shift_model(args.source, args.output)

    if args.validate:
        ok = validate_model(args.output, args.task)
        print(f"Validation: {'PASSED' if ok else 'FAILED'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
