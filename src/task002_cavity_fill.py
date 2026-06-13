"""Build a minimal ONNX model for task002: 4-directional cavity fill.

Current model: 95 nodes, 24 Conv, cost 97,293.
New model: ~15 nodes, 4 directional Conv, estimated cost ~15K.

Rule: a 0-cell becomes color-4 if there are color-3 cells on all 4 sides.
"""

import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import helper, checker, numpy_helper


def build_model(output_path: str):
    """Build minimal cavity fill model."""
    # Input: (1, 10, 30, 30) one-hot float32
    input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 30, 30])
    output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10, 30, 30])

    nodes = []
    initializers = []

    # Constants
    def make_const_f16(name, value):
        t = numpy_helper.from_array(np.array([value], dtype=np.float16), name=name)
        initializers.append(t)
        return name

    def make_const_i64(name, value, dims=None):
        arr = np.array(value, dtype=np.int64)
        if dims:
            arr = arr.reshape(dims)
        t = numpy_helper.from_array(arr, name=name)
        initializers.append(t)
        return name

    def make_weights(name, shape, values):
        arr = np.array(values, dtype=np.float16).reshape(shape)
        t = numpy_helper.from_array(arr, name=name)
        initializers.append(t)
        return name

    # 1-hot weights for EXTRACTING color channels from (1,10,30,30)
    # (out_ch, in_ch, H, W) = (1, 10, 1, 1)
    ch0_w = make_weights("ch0_w", (1, 10, 1, 1), [1.0] + [0.0] * 9)
    ch3_w = make_weights("ch3_w", (1, 10, 1, 1), [0.0, 0.0, 0.0, 1.0] + [0.0] * 6)

    # Weights for SPREADING single channel to 10-channel (for output)
    # (out_ch, in_ch, H, W) = (10, 1, 1, 1)
    ch0_spread_w = make_weights("ch0_spread_w", (10, 1, 1, 1), [1.0] + [0.0] * 9)
    ch4_spread_w = make_weights("ch4_spread_w", (10, 1, 1, 1), [0.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 5)

    # Pad values
    half16 = make_const_f16("half16", 0.5)
    one16 = make_const_f16("one16", 1.0)
    zero16 = make_const_f16("zero16", 0.0)

    # Cast input to float16
    nodes.append(helper.make_node("Cast", ["input"], ["input_f16"], to=onnx.TensorProto.FLOAT16))

    # Extract color-0 and color-3 channels
    nodes.append(helper.make_node("Conv", ["input_f16", ch0_w], ["ch0"], kernel_shape=[1, 1]))
    nodes.append(helper.make_node("Conv", ["input_f16", ch3_w], ["ch3"], kernel_shape=[1, 1]))
    # ch0: (1,1,30,30), ch3: (1,1,30,30)

    # Mark color-3 cells (cast bool->float16 for Conv)
    nodes.append(helper.make_node("Greater", ["ch3", half16], ["is3_bool"]))
    nodes.append(helper.make_node("Cast", ["is3_bool"], ["is3"], to=onnx.TensorProto.FLOAT16))
    # is3: (1,1,30,30) float16 (0.0 or 1.0)

    # Compute directional reachability using MaxPool
    # MaxPool with asymmetric padding checks "is there any color-3 in this direction?"
    # Kernel size matches the number of cells to look at (29 out of max 29 in that direction)
    # has_left: kernel (1,29), pad [0,28,0,0] — looks at 28 cells left, excludes self
    nodes.append(helper.make_node("MaxPool", ["is3"], ["has_left_cnt"],
                                  kernel_shape=[1, 29], pads=[0, 28, 0, 0], strides=[1, 1]))
    # has_right: kernel (1,29), pad [0,0,0,28] — looks at 28 cells right
    nodes.append(helper.make_node("MaxPool", ["is3"], ["has_right_cnt"],
                                  kernel_shape=[1, 29], pads=[0, 0, 0, 28], strides=[1, 1]))
    # has_up: kernel (29,1), pad [28,0,0,0] — looks at 28 cells above
    nodes.append(helper.make_node("MaxPool", ["is3"], ["has_up_cnt"],
                                  kernel_shape=[29, 1], pads=[28, 0, 0, 0], strides=[1, 1]))
    # has_down: kernel (29,1), pad [0,0,28,0] — looks at 28 cells below
    nodes.append(helper.make_node("MaxPool", ["is3"], ["has_down_cnt"],
                                  kernel_shape=[29, 1], pads=[0, 0, 28, 0], strides=[1, 1]))

    # has_X_cnt: (1,1,30,30) float16 — count of color-3 cells in that direction
    # Enclosed if all 4 directions have count > 0
    nodes.append(helper.make_node("Greater", ["has_left_cnt", zero16], ["has_left_bool"]))
    nodes.append(helper.make_node("Greater", ["has_right_cnt", zero16], ["has_right_bool"]))
    nodes.append(helper.make_node("Greater", ["has_up_cnt", zero16], ["has_up_bool"]))
    nodes.append(helper.make_node("Greater", ["has_down_cnt", zero16], ["has_down_bool"]))

    # Cast bool to float16 for And (ONNX And only works on bool)
    # And is fine with bool inputs, but subsequent operations need float16
    nodes.append(helper.make_node("And", ["has_left_bool", "has_right_bool"], ["and_lr_bool"]))
    nodes.append(helper.make_node("And", ["has_up_bool", "has_down_bool"], ["and_ud_bool"]))
    nodes.append(helper.make_node("And", ["and_lr_bool", "and_ud_bool"], ["enclosed_bool"]))

    # Fill: enclosed AND (original cell is color 0)
    nodes.append(helper.make_node("Greater", ["ch0", half16], ["is0_bool"]))
    nodes.append(helper.make_node("And", ["enclosed_bool", "is0_bool"], ["fill_mask_bool"]))

    # Cast to float16 for arithmetic
    nodes.append(helper.make_node("Cast", ["fill_mask_bool"], ["fill_mask"], to=onnx.TensorProto.FLOAT16))

    # Create color-4 fill (fill_mask is 0.0 or 1.0 float16)
    nodes.append(helper.make_node("Mul", ["fill_mask", one16], ["fill_val"]))
    # fill_val: (1,1,30,30) = 0 or 1 (representing color 4)

    # Build output: for each channel, original OR filled (color 4 = channel 4)
    # We need to blend: output = input + fill_delta
    # fill_delta: channel 0 becomes 0, channel 4 becomes fill_val
    # Use a simple approach: Conv to spread fill_val into channel 4

    # Convert fill_val (1 chan) to channel 4 one-hot (10 chan)
    nodes.append(helper.make_node("Conv", ["fill_val", "ch4_spread_w"], ["fill_ch4"], kernel_shape=[1, 1]))
    # fill_ch4: (1,10,30,30)

    # Remove original color-0 from filled cells: spread fill_mask to channel 0
    nodes.append(helper.make_node("Conv", ["fill_mask", "ch0_spread_w"], ["remove_ch0"], kernel_shape=[1, 1]))
    # remove_ch0: (1,10,30,30) — channel 0 mask

    # Combine: input - remove_ch0 + fill_ch4
    nodes.append(helper.make_node("Sub", ["input_f16", "remove_ch0"], ["tmp"]))
    nodes.append(helper.make_node("Add", ["tmp", "fill_ch4"], ["output_f16"]))

    # Cast back to float32
    nodes.append(helper.make_node("Cast", ["output_f16"], ["output"], to=onnx.TensorProto.FLOAT))

    # Build graph
    graph = helper.make_graph(
        nodes=nodes,
        name="task002_cavity_fill",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 10)])
    checker.check_model(model)
    onnx.save(model, output_path)
    print(f"Built model: {len(nodes)} nodes, {len(initializers)} initializers")


def validate_model(model_path: str, task_path: str) -> bool:
    import onnxruntime as ort

    with open(task_path) as f:
        data = json.load(f)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    for split_name in ["train", "test"]:
        for i, case in enumerate(data.get(split_name, [])):
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
                # Show differences
                diff = out_pred != out_expected
                for r, c in np.argwhere(diff)[:5]:
                    print(f"    [{r},{c}]: pred={out_pred[r,c]}, expected={out_expected[r,c]}")
                return False
            print(f"  {split_name}[{i}]: PASS")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/candidates/task002_cavity_fill/task002_CavityFill.onnx")
    parser.add_argument("--task", default="task/task002.json")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    build_model(args.output)

    if args.validate:
        ok = validate_model(args.output, args.task)
        print(f"Validation: {'PASSED' if ok else 'FAILED'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
