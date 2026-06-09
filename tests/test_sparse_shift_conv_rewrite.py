from __future__ import annotations

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from src.sparse_shift_conv_rewrite import rewrite_sparse_shift_convs


def _run(model_path, value: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    return session.run(None, {"input": value.astype(np.float32)})[0]


def test_rewrite_sparse_shift_convs_preserves_conv_output(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 3, 3])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4, 3, 3])
    weight = np.zeros((4, 1, 5, 5), dtype=np.float32)
    weight[0, 0, 2, 2] = 1.0
    weight[1, 0, 1, 2] = 1.0
    weight[2, 0, 2, 1] = 1.0
    weight[3, 0, 0, 0] = 1.0
    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Conv",
                ["input", "wk"],
                ["output"],
                name="shift_conv",
                pads=[2, 2, 2, 2],
            )
        ],
        name="sparse_shift_conv",
        inputs=[x],
        outputs=[y],
        initializer=[numpy_helper.from_array(weight, name="wk")],
    )
    model = helper.make_model(
        graph,
        producer_name="test",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, str(input_path))

    report = rewrite_sparse_shift_convs(str(input_path), str(output_path), weight_name="wk")
    rewritten = onnx.load(str(output_path))

    assert report["rewritten_conv_nodes"] == 1
    assert report["removed_initializer_elements"] == 100
    assert report["output_cost"] < report["source_cost"]
    assert not any(initializer.name == "wk" for initializer in rewritten.graph.initializer)
    assert not any(node.op_type == "Conv" for node in rewritten.graph.node)
    assert any(node.op_type == "Pad" for node in rewritten.graph.node)
    assert any(node.op_type == "Slice" for node in rewritten.graph.node)
    assert any(node.op_type == "Concat" for node in rewritten.graph.node)
    onnx.checker.check_model(rewritten)

    value = np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3)
    np.testing.assert_array_equal(_run(input_path, value), _run(output_path, value))
