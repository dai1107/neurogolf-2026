from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from src.initializer_dtype_compression import compress_initializer_dtypes


def _run(model_path, value: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session.run(None, {"input": value.astype(np.float32)})[0]


def test_compress_initializer_dtypes_stores_gather_indices_compactly(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 6])
    indices = np.asarray([0, 1, 2, 3, 2, 1], dtype=np.int64)
    graph = helper.make_graph(
        nodes=[helper.make_node("Gather", ["input", "Indices"], ["output"], axis=2, name="output")],
        name="gather_indices",
        inputs=[x],
        outputs=[y],
        initializer=[numpy_helper.from_array(indices, name="Indices")],
    )
    model = helper.make_model(
        graph,
        producer_name="test",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.save(model, str(input_path))

    report = compress_initializer_dtypes(str(input_path), str(output_path), min_elements=4)
    output = onnx.load(str(output_path))

    assert report["compressed_initializers"] == 1
    assert report["compressed_initializer_elements"] == 6
    assert report["output_cost"] < report["source_cost"]
    assert any(node.op_type == "Cast" and node.output == ["Indices"] for node in output.graph.node)
    assert any(initializer.name == "Indices_compact" for initializer in output.graph.initializer)
    assert all(initializer.name != "Indices" for initializer in output.graph.initializer)
    onnx.checker.check_model(output)

    value = np.arange(4, dtype=np.float32).reshape(1, 1, 4)
    np.testing.assert_array_equal(_run(input_path, value), _run(output_path, value))


def test_compress_initializer_dtypes_stores_float_binary_mask_as_bool(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 3])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 3])
    mask = np.asarray([[[[1, 0, 1], [0, 1, 0]]]], dtype=np.float32)
    graph = helper.make_graph(
        nodes=[helper.make_node("Mul", ["input", "Mask"], ["output"], name="output")],
        name="float_mask",
        inputs=[x],
        outputs=[y],
        initializer=[numpy_helper.from_array(mask, name="Mask")],
    )
    model = helper.make_model(
        graph,
        producer_name="test",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.save(model, str(input_path))

    report = compress_initializer_dtypes(str(input_path), str(output_path), min_elements=4)
    output = onnx.load(str(output_path))

    assert report["compressed_initializers"] == 1
    assert report["output_cost"] < report["source_cost"]
    compact = next(initializer for initializer in output.graph.initializer if initializer.name == "Mask_compact")
    assert compact.data_type == TensorProto.BOOL
    onnx.checker.check_model(output)

    value = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3)
    np.testing.assert_array_equal(_run(input_path, value), _run(output_path, value))
