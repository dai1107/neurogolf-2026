from __future__ import annotations

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

from src.official_cost_estimator import estimate_official_static_cost


def test_official_static_cost_counts_params_and_intermediate_memory(tmp_path) -> None:
    model_path = tmp_path / "model.onnx"
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 2, 2])
    mid = helper.make_tensor_value_info("mid", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 2, 2])
    weight = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="W")
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "W"], ["mid"]), helper.make_node("Relu", ["mid"], ["output"])],
        "g",
        [x],
        [y],
        [weight],
        value_info=[mid],
    )
    model = helper.make_model(
        graph,
        producer_name="test",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, str(model_path))

    cost = estimate_official_static_cost(str(model_path))

    assert cost["params"] == 1
    assert cost["initializer_params"] == 1
    assert cost["constant_params"] == 0
    assert cost["tensor_memory_bytes"] == 16
    assert cost["official_static_cost"] == 17


def test_official_static_cost_counts_constant_node_params(tmp_path) -> None:
    model_path = tmp_path / "model.onnx"
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["output"],
                value=numpy_helper.from_array(np.asarray([1.0, 2.0], dtype=np.float32)),
            )
        ],
        "g",
        [],
        [y],
    )
    model = helper.make_model(
        graph,
        producer_name="test",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, str(model_path))

    cost = estimate_official_static_cost(str(model_path))

    assert cost["params"] == 2
    assert cost["constant_params"] == 2
    assert cost["tensor_memory_bytes"] == 0
    assert cost["official_static_cost"] == 2
