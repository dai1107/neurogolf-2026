from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from src.cost_estimator import estimate_model_cost
from src.zero_initializer_compression import compress_zero_initializers


def _run(model_path, value: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session.run(None, {"input": value.astype(np.float32)})[0]


def test_compress_zero_initializers_uses_constant_of_shape(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])
    zero = np.zeros((1, 1, 4, 4), dtype=np.float32)
    graph = helper.make_graph(
        nodes=[helper.make_node("Add", ["input", "Zero"], ["output"], name="output")],
        name="large_zero_initializer",
        inputs=[x],
        outputs=[y],
        initializer=[numpy_helper.from_array(zero, name="Zero")],
    )
    model = helper.make_model(
        graph,
        producer_name="test",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.save(model, str(input_path))

    report = compress_zero_initializers(str(input_path), str(output_path), min_elements=4)
    output = onnx.load(str(output_path))

    assert report["zero_initializers_replaced"] == 1
    assert report["zero_initializer_elements_replaced"] == 16
    assert report["output_cost"] < report["source_cost"]
    assert any(node.op_type == "ConstantOfShape" and node.output == ["Zero"] for node in output.graph.node)
    assert [initializer.name for initializer in output.graph.initializer] == ["Zero_shape"]
    onnx.checker.check_model(output)

    value = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    np.testing.assert_array_equal(_run(input_path, value), _run(output_path, value))


def test_compress_zero_initializers_keeps_small_zero_constants(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    zero = np.zeros((1, 1), dtype=np.float32)
    graph = helper.make_graph(
        nodes=[helper.make_node("Add", ["input", "Zero"], ["output"], name="output")],
        name="small_zero_initializer",
        inputs=[x],
        outputs=[y],
        initializer=[numpy_helper.from_array(zero, name="Zero")],
    )
    model = helper.make_model(
        graph,
        producer_name="test",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.save(model, str(input_path))

    report = compress_zero_initializers(str(input_path), str(output_path), min_elements=4)
    output = onnx.load(str(output_path))

    assert report["zero_initializers_replaced"] == 0
    assert report["output_cost"] == estimate_model_cost(str(input_path))["estimated_cost"]
    assert all(node.op_type != "ConstantOfShape" for node in output.graph.node)
    assert [initializer.name for initializer in output.graph.initializer] == ["Zero"]
    onnx.checker.check_model(output)
