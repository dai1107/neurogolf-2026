from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.remove_zero_adds import optimize_model


def test_remove_zero_adds_keeps_graph_output_identity_node(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    zero = np.asarray([[0.0]], dtype=np.float32)
    graph = helper.make_graph(
        nodes=[helper.make_node("Add", ["input", "Zero"], ["output"], name="output_add")],
        name="graph_output_identity",
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

    report = optimize_model(str(input_path), str(output_path))
    output = onnx.load(str(output_path))

    assert report["removed_nodes"] == 0
    assert [node.name for node in output.graph.node] == ["output_add"]
    assert output.graph.output[0].name == "output"
    onnx.checker.check_model(output)


def test_remove_zero_adds_handles_unnamed_nodes_independently(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    zero = np.asarray([[0.0]], dtype=np.float32)
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Add", ["input", "Zero"], ["middle"], name=""),
            helper.make_node("Relu", ["middle"], ["output"], name=""),
        ],
        name="unnamed_nodes",
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

    report = optimize_model(str(input_path), str(output_path))
    output = onnx.load(str(output_path))

    assert report["removed_nodes"] == 1
    assert [node.op_type for node in output.graph.node] == ["Relu"]
    assert output.graph.node[0].input[0] == "input"
    assert output.graph.node[0].output[0] == "output"
    onnx.checker.check_model(output)


def test_remove_zero_adds_keeps_identity_when_it_broadcasts_shape(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 3])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 3, 3])
    zero = np.zeros((1, 1, 3, 3), dtype=np.float32)
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Add", ["input", "Zero"], ["expanded"], name="broadcast_add"),
            helper.make_node("Relu", ["expanded"], ["output"], name="relu"),
        ],
        name="broadcast_identity",
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

    report = optimize_model(str(input_path), str(output_path))
    output = onnx.load(str(output_path))

    assert report["removed_nodes"] == 0
    assert [node.name for node in output.graph.node] == ["broadcast_add", "relu"]
    onnx.checker.check_model(output)
