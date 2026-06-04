from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.deduplicate_initializers import deduplicate_initializers


def test_deduplicate_initializers_merges_identical_constants(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    value = np.asarray([[1.0]], dtype=np.float32)
    graph = helper.make_graph(
        nodes=[
            helper.make_node("Add", ["input", "A"], ["a"], name="a"),
            helper.make_node("Add", ["a", "B"], ["output"], name="output"),
        ],
        name="duplicate_initializers",
        inputs=[x],
        outputs=[y],
        initializer=[
            numpy_helper.from_array(value, name="A"),
            numpy_helper.from_array(value, name="B"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="test",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.save(model, str(input_path))

    report = deduplicate_initializers(str(input_path), str(output_path))
    output = onnx.load(str(output_path))

    assert report["deduplicated_initializers"] == 1
    assert report["output_initializer_count"] == 1
    assert [initializer.name for initializer in output.graph.initializer] == ["A"]
    assert output.graph.node[1].input[1] == "A"
    onnx.checker.check_model(output)


def test_deduplicate_initializers_removes_unreferenced_non_input_constants(tmp_path) -> None:
    input_path = tmp_path / "input.onnx"
    output_path = tmp_path / "output.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1])
    value = np.asarray([[1.0]], dtype=np.float32)
    unused = np.asarray([[2.0]], dtype=np.float32)
    graph = helper.make_graph(
        nodes=[helper.make_node("Add", ["input", "A"], ["output"], name="output")],
        name="unused_initializer",
        inputs=[x],
        outputs=[y],
        initializer=[
            numpy_helper.from_array(value, name="A"),
            numpy_helper.from_array(unused, name="Unused"),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="test",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.save(model, str(input_path))

    report = deduplicate_initializers(str(input_path), str(output_path))
    output = onnx.load(str(output_path))

    assert report["removed_unused_initializers"] == 1
    assert [initializer.name for initializer in output.graph.initializer] == ["A"]
    onnx.checker.check_model(output)
