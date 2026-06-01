"""Build the required single-layer Conv2D ONNX model for task000."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .cost_estimator import check_forbidden_ops, estimate_model_cost
from .encoding import DEFAULT_COLORS, DEFAULT_HEIGHT, DEFAULT_SHAPE, DEFAULT_WIDTH
from .validate_onnx_model import run_model


WeightFn = Callable[[int, int, tuple[int, int]], float]


def example_weight(
    channel_out: int,
    channel_in: int,
    kernel_coord: tuple[int, int],
) -> float:
    """Weight rule from the task000 example in the competition description."""
    if kernel_coord == (0, 0) and channel_in == channel_out:
        return 1.0
    if kernel_coord == (0, 0) and channel_in != 5 and channel_out == 0:
        return -1.0
    if kernel_coord == (-1, -1) and channel_in != 5 and channel_out == 0:
        return 1.0
    if kernel_coord == (-1, -1) and channel_in != 5 and channel_out == 5:
        return -1.0
    return 0.0


def make_conv_weights(
    weight_fn: WeightFn,
    num_channels: int = DEFAULT_COLORS,
    kernel_size: int = 3,
) -> np.ndarray:
    """Materialize a relative-coordinate weight function as ONNX Conv weights."""
    if num_channels <= 0:
        raise ValueError("num_channels must be positive")
    if kernel_size <= 0 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")

    radius = kernel_size // 2
    weights = np.zeros(
        (num_channels, num_channels, kernel_size, kernel_size),
        dtype=np.float32,
    )
    for channel_out in range(num_channels):
        for channel_in in range(num_channels):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    weights[channel_out, channel_in, dy + radius, dx + radius] = weight_fn(
                        channel_out,
                        channel_in,
                        (dy, dx),
                    )
    return weights


def build_single_layer_conv2d_model(
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    weight_fn: WeightFn = example_weight,
) -> None:
    """Build and save a static-shape single-layer 3x3 Conv2D ONNX model."""
    if num_channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("num_channels, height, and width must be positive")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    input_shape = [1, num_channels, height, width]
    output_shape = [1, num_channels, height, width]
    weights = make_conv_weights(weight_fn, num_channels=num_channels, kernel_size=3)

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)
    weight_initializer = numpy_helper.from_array(weights, name="W")
    conv = helper.make_node(
        "Conv",
        inputs=["input", "W"],
        outputs=["output"],
        name="output",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    graph = helper.make_graph(
        nodes=[conv],
        name="single_layer_conv2d",
        inputs=[input_info],
        outputs=[output_info],
        initializer=[weight_initializer],
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-2026",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )

    onnx.checker.check_model(model)
    onnx.save(model, str(output))
    onnx.checker.check_model(str(output))


def _print_cost_report(model_path: str) -> None:
    cost = estimate_model_cost(model_path)
    forbidden = check_forbidden_ops(model_path)
    print(f"num_parameters = {cost['num_parameters']}")
    print(f"initializer_memory_bytes = {cost['initializer_memory_bytes']}")
    print(f"file_size_bytes = {cost['file_size_bytes']}")
    print(f"estimated_cost = {cost['estimated_cost']}")
    print(f"estimated_score = {cost['estimated_score']:.6f}")
    print(f"file_size_ok = {cost['file_size_ok']}")
    print(f"forbidden_ops_check = {'passed' if forbidden['passed'] else 'failed'}")
    if not forbidden["passed"]:
        print(f"forbidden_ops_found = {forbidden['forbidden_ops_found']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/onnx/task000.onnx")
    parser.add_argument("--num-channels", type=int, default=DEFAULT_COLORS)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    args = parser.parse_args()

    build_single_layer_conv2d_model(
        output_path=args.output,
        num_channels=args.num_channels,
        height=args.height,
        width=args.width,
    )
    print(f"saved model to {args.output}")
    print("onnx checker passed")
    _print_cost_report(args.output)

    sample_input = np.zeros(DEFAULT_SHAPE, dtype=np.float32)
    sample_output = run_model(args.output, sample_input)
    print(f"onnxruntime output_shape = {list(sample_output.shape)}")


if __name__ == "__main__":
    main()
