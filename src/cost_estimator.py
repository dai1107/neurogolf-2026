"""Cost and constraint checks for generated ONNX models."""

from __future__ import annotations

import math
import os
from pathlib import Path

import onnx


FILE_SIZE_LIMIT_BYTES = int(1.44 * 1024 * 1024)
FORBIDDEN_OPS = {
    "LOOP",
    "SCAN",
    "NONZERO",
    "UNIQUE",
    "SCRIPT",
    "FUNCTION",
    "COMPRESS",
}


def _num_elements(dims: list[int]) -> int:
    if not dims:
        return 1
    if any(dim <= 0 for dim in dims):
        raise ValueError(f"initializer has non-positive dims: {dims}")
    return math.prod(dims)


def _dtype_size_bytes(data_type: int) -> int:
    try:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(data_type)
    except Exception as exc:  # pragma: no cover - defensive for unknown dtypes.
        raise ValueError(f"unsupported ONNX tensor data type: {data_type}") from exc
    return int(np_dtype.itemsize)


def estimate_model_cost(model_path: str) -> dict[str, float | int | bool]:
    """Estimate NeuroGolf cost from ONNX initializers and file size."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"model file does not exist: {model_path}")

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    num_parameters = 0
    initializer_memory_bytes = 0
    for initializer in model.graph.initializer:
        count = _num_elements(list(initializer.dims))
        num_parameters += count
        initializer_memory_bytes += count * _dtype_size_bytes(initializer.data_type)

    file_size_bytes = os.path.getsize(path)
    estimated_cost = num_parameters + initializer_memory_bytes
    estimated_score = max(1.0, 25.0 - math.log(max(1, estimated_cost)))

    return {
        "num_parameters": num_parameters,
        "initializer_memory_bytes": initializer_memory_bytes,
        "file_size_bytes": file_size_bytes,
        "estimated_cost": estimated_cost,
        "estimated_score": estimated_score,
        "file_size_ok": file_size_bytes <= FILE_SIZE_LIMIT_BYTES,
    }


def check_static_shapes(model_path: str) -> dict[str, bool | list[str]]:
    """Check that graph inputs, outputs, and value_info tensors use static shapes."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"model file does not exist: {model_path}")

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    graph = onnx.shape_inference.infer_shapes(model).graph
    failures: list[str] = []

    for value_info in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor_type = value_info.type.tensor_type
        if not value_info.type.HasField("tensor_type"):
            failures.append(f"{value_info.name}: not a tensor")
            continue
        if not tensor_type.HasField("shape"):
            failures.append(f"{value_info.name}: missing shape")
            continue
        for dim_index, dim in enumerate(tensor_type.shape.dim):
            if dim.HasField("dim_param"):
                failures.append(f"{value_info.name}[{dim_index}]: dynamic dim_param {dim.dim_param}")
            elif not dim.HasField("dim_value"):
                failures.append(f"{value_info.name}[{dim_index}]: missing dim_value")
            elif dim.dim_value <= 0:
                failures.append(f"{value_info.name}[{dim_index}]: non-positive dim {dim.dim_value}")

    return {
        "passed": not failures,
        "failures": failures,
    }


def check_forbidden_ops(model_path: str) -> dict[str, bool | list[str]]:
    """Check whether an ONNX model uses operators disallowed by the contest."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"model file does not exist: {model_path}")

    model = onnx.load(str(path))
    found = sorted({node.op_type for node in model.graph.node if node.op_type.upper() in FORBIDDEN_OPS})
    return {
        "passed": not found,
        "forbidden_ops_found": found,
    }
