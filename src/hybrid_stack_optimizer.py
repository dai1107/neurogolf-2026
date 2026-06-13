"""Analyze and conservatively optimize hybrid-stack ONNX submissions.

The active 6348.56 reference uses two lanes:
``base_submission/taskNNN.onnx`` and ``overrides/taskNNN.onnx``.  This module
implements low-risk graph-equivalent passes and lane-level one-task ablation
builders.  It intentionally does not overwrite ``outputs/submission.zip``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import AttributeProto, TensorProto, numpy_helper

from .arc_io import load_task
from .cost_estimator import check_forbidden_ops, check_static_shapes, estimate_model_cost
from .encoding import DEFAULT_SHAPE, grid_to_onehot
from .inspect_submission import HYBRID_STACK_DIRS, inspect_submission


ort.set_default_logger_severity(3)


ANALYSIS_FIELDS = [
    "task_id",
    "lane",
    "model_path",
    "file_size_bytes",
    "initializer_bytes",
    "initializer_count",
    "node_count",
    "op_count",
    "gather_count",
    "conv_count",
    "constant_count",
    "op_sequence",
    "initializer_shape_signature",
    "large_initializer_topk",
    "model_sha256",
    "peer_sha256",
    "same_as_peer",
    "template_id",
]

OPTIMIZE_FIELDS = [
    "task_id",
    "lane",
    "source_model_path",
    "output_model_path",
    "source_estimated_cost",
    "output_estimated_cost",
    "estimated_cost_delta",
    "source_file_size_bytes",
    "output_file_size_bytes",
    "file_size_delta",
    "source_initializer_bytes",
    "output_initializer_bytes",
    "initializer_bytes_delta",
    "source_initializer_count",
    "output_initializer_count",
    "changed",
    "removed_dead_nodes",
    "removed_unused_initializers",
    "removed_unused_value_info",
    "deduplicated_initializers",
    "constant_gather_tables_pruned",
    "constant_gather_rows_removed",
    "constant_gather_bytes_removed",
    "checker_passed",
    "forbidden_ops_passed",
    "static_shapes_passed",
    "equivalence_passed",
    "equivalence_inputs_checked",
    "equivalence_max_abs_diff",
    "candidate_valid",
    "failure_reason",
]

ABLATION_FIELDS = [
    "task_id",
    "lane",
    "candidate_model_path",
    "candidate_zip_path",
    "upload_submission_path",
    "base_entry_replaced",
    "source_entry_sha256",
    "candidate_sha256",
    "candidate_valid",
    "failure_reason",
]

MERGE_FIELDS = [
    "task_id",
    "lane",
    "candidate_model_path",
    "base_entry_replaced",
    "source_entry_sha256",
    "candidate_sha256",
    "candidate_valid",
    "failure_reason",
]


@dataclass(frozen=True)
class GraphRewriteStats:
    removed_dead_nodes: int = 0
    removed_unused_initializers: int = 0
    removed_unused_value_info: int = 0
    deduplicated_initializers: int = 0
    constant_gather_tables_pruned: int = 0
    constant_gather_rows_removed: int = 0
    constant_gather_bytes_removed: int = 0

    def changed(self) -> bool:
        return any(
            (
                self.removed_dead_nodes,
                self.removed_unused_initializers,
                self.removed_unused_value_info,
                self.deduplicated_initializers,
                self.constant_gather_tables_pruned,
                self.constant_gather_rows_removed,
                self.constant_gather_bytes_removed,
            )
        )

    def plus(self, other: "GraphRewriteStats") -> "GraphRewriteStats":
        return GraphRewriteStats(
            removed_dead_nodes=self.removed_dead_nodes + other.removed_dead_nodes,
            removed_unused_initializers=(
                self.removed_unused_initializers + other.removed_unused_initializers
            ),
            removed_unused_value_info=(
                self.removed_unused_value_info + other.removed_unused_value_info
            ),
            deduplicated_initializers=(
                self.deduplicated_initializers + other.deduplicated_initializers
            ),
            constant_gather_tables_pruned=(
                self.constant_gather_tables_pruned + other.constant_gather_tables_pruned
            ),
            constant_gather_rows_removed=(
                self.constant_gather_rows_removed + other.constant_gather_rows_removed
            ),
            constant_gather_bytes_removed=(
                self.constant_gather_bytes_removed + other.constant_gather_bytes_removed
            ),
        )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _tensor_numel(dims: Iterable[int]) -> int:
    result = 1
    for dim in dims:
        result *= int(dim)
    return result


def _dtype_size(data_type: int) -> int:
    try:
        return int(onnx.helper.tensor_dtype_to_np_dtype(data_type).itemsize)
    except Exception:
        return 0


def _initializer_bytes(initializer: onnx.TensorProto) -> int:
    return _tensor_numel(initializer.dims) * _dtype_size(initializer.data_type)


def _initializer_key(initializer: onnx.TensorProto) -> bytes:
    clone = onnx.TensorProto()
    clone.CopyFrom(initializer)
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


def _tensor_hash_prefix(initializer: onnx.TensorProto) -> str:
    return hashlib.sha256(_initializer_key(initializer)).hexdigest()[:12]


def _sanitize_name(value: str) -> str:
    safe = value.replace("\\", "_").replace("/", "_").replace(":", "_")
    for char in '*?"<>|':
        safe = safe.replace(char, "_")
    return safe


def _attr_int(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name and attr.type == AttributeProto.INT:
            return int(attr.i)
    return default


def _graph_boundary_names(graph: onnx.GraphProto) -> set[str]:
    return {value.name for value in graph.input} | {value.name for value in graph.output}


def _consumer_map(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            if name:
                consumers.setdefault(name, []).append(node)
    return consumers


def _producer_map(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    producers: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for name in node.output:
            if name:
                producers[name] = node
    return producers


def _constant_tensor_attribute(node: onnx.NodeProto) -> onnx.AttributeProto | None:
    if node.op_type != "Constant":
        return None
    for attr in node.attribute:
        if attr.name == "value" and attr.type == AttributeProto.TENSOR:
            return attr
    return None


def _constant_array_source(
    model: onnx.ModelProto,
    name: str,
) -> tuple[str, np.ndarray, onnx.TensorProto | onnx.NodeProto] | None:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return ("initializer", numpy_helper.to_array(initializer), initializer)

    producer = _producer_map(model.graph).get(name)
    if producer is None:
        return None
    attr = _constant_tensor_attribute(producer)
    if attr is None:
        return None
    return ("constant", numpy_helper.to_array(attr.t), producer)


def _replace_initializer_array(
    model: onnx.ModelProto,
    name: str,
    array: np.ndarray,
) -> None:
    for initializer in model.graph.initializer:
        if initializer.name == name:
            initializer.CopyFrom(numpy_helper.from_array(array, name=name))
            return
    raise ValueError(f"initializer not found: {name}")


def _replace_constant_array(node: onnx.NodeProto, array: np.ndarray) -> None:
    attr = _constant_tensor_attribute(node)
    if attr is None:
        raise ValueError(f"node is not a tensor Constant: {node.name}")
    old_name = attr.t.name
    replacement = numpy_helper.from_array(array)
    replacement.name = old_name
    attr.t.CopyFrom(replacement)


def _prune_dead_graph(model: onnx.ModelProto) -> GraphRewriteStats:
    """Remove nodes and metadata not reachable from graph outputs."""
    graph = model.graph
    required = {output.name for output in graph.output}
    keep_node_indices: set[int] = set()

    for index in range(len(graph.node) - 1, -1, -1):
        node = graph.node[index]
        if any(output_name in required for output_name in node.output if output_name):
            keep_node_indices.add(index)
            required.update(input_name for input_name in node.input if input_name)

    if len(keep_node_indices) != len(graph.node):
        kept_nodes = [
            node
            for index, node in enumerate(graph.node)
            if index in keep_node_indices
        ]
        removed_nodes = len(graph.node) - len(kept_nodes)
        del graph.node[:]
        graph.node.extend(kept_nodes)
    else:
        removed_nodes = 0

    protected_initializers = {value.name for value in graph.input}
    kept_initializers = [
        initializer
        for initializer in graph.initializer
        if initializer.name in required or initializer.name in protected_initializers
    ]
    removed_initializers = len(graph.initializer) - len(kept_initializers)
    if removed_initializers:
        del graph.initializer[:]
        graph.initializer.extend(kept_initializers)

    protected_value_info = required | {value.name for value in graph.input} | {
        value.name for value in graph.output
    }
    kept_value_info = [
        value_info
        for value_info in graph.value_info
        if value_info.name in protected_value_info
    ]
    removed_value_info = len(graph.value_info) - len(kept_value_info)
    if removed_value_info:
        del graph.value_info[:]
        graph.value_info.extend(kept_value_info)

    return GraphRewriteStats(
        removed_dead_nodes=removed_nodes,
        removed_unused_initializers=removed_initializers,
        removed_unused_value_info=removed_value_info,
    )


def _deduplicate_initializers(model: onnx.ModelProto) -> GraphRewriteStats:
    """Merge byte-identical initializers inside one model."""
    graph = model.graph
    protected = _graph_boundary_names(graph)
    seen: dict[bytes, str] = {}
    rename: dict[str, str] = {}
    kept: list[onnx.TensorProto] = []

    for initializer in graph.initializer:
        if initializer.name in protected:
            kept.append(initializer)
            continue
        key = _initializer_key(initializer)
        canonical_name = seen.get(key)
        if canonical_name is None:
            seen[key] = initializer.name
            kept.append(initializer)
        else:
            rename[initializer.name] = canonical_name

    if not rename:
        return GraphRewriteStats()

    for node in graph.node:
        for index, input_name in enumerate(node.input):
            if input_name in rename:
                node.input[index] = rename[input_name]

    del graph.initializer[:]
    graph.initializer.extend(kept)
    return GraphRewriteStats(deduplicated_initializers=len(rename))


def _remap_indices(indices: np.ndarray, used: list[int], axis_dim: int) -> np.ndarray:
    normalized = indices.astype(np.int64, copy=False)
    normalized = np.where(normalized < 0, normalized + axis_dim, normalized)
    mapping = {old: new for new, old in enumerate(used)}
    flattened = normalized.reshape(-1)
    remapped = np.asarray([mapping[int(value)] for value in flattened], dtype=np.int64)
    return remapped.reshape(indices.shape).astype(indices.dtype, copy=False)


def _prune_constant_gather_tables(model: onnx.ModelProto) -> GraphRewriteStats:
    """Prune Gather data tables when both data and indices are constant.

    This is the P3-A rewrite from the strategy: no op-type or dtype changes,
    only remove unreachable rows from a constant table and remap the constant
    indices used by the same Gather.
    """
    graph = model.graph
    consumers = _consumer_map(graph)
    protected = _graph_boundary_names(graph)
    initializer_names = {initializer.name for initializer in graph.initializer}

    tables_pruned = 0
    rows_removed = 0
    bytes_removed = 0
    changed_initializer_names: set[str] = set()

    for node in list(graph.node):
        if node.op_type != "Gather" or len(node.input) < 2:
            continue
        table_name, indices_name = node.input[0], node.input[1]
        if table_name not in initializer_names or table_name in protected:
            continue
        if len(consumers.get(table_name, [])) != 1:
            continue
        if indices_name in protected:
            continue
        if len(consumers.get(indices_name, [])) != 1:
            continue

        table_source = _constant_array_source(model, table_name)
        indices_source = _constant_array_source(model, indices_name)
        if table_source is None or indices_source is None:
            continue
        _table_kind, table_array, _table_ref = table_source
        indices_kind, indices_array, indices_ref = indices_source
        if not np.issubdtype(indices_array.dtype, np.integer):
            continue
        if table_array.ndim == 0:
            continue

        axis = _attr_int(node, "axis", 0)
        if axis < 0:
            axis += table_array.ndim
        if axis < 0 or axis >= table_array.ndim:
            continue
        axis_dim = int(table_array.shape[axis])
        if axis_dim <= 0:
            continue

        normalized = indices_array.astype(np.int64, copy=False)
        normalized = np.where(normalized < 0, normalized + axis_dim, normalized)
        if normalized.size == 0:
            continue
        if int(normalized.min()) < 0 or int(normalized.max()) >= axis_dim:
            continue
        used = sorted(int(value) for value in np.unique(normalized))
        if len(used) >= axis_dim:
            continue

        pruned_table = np.take(table_array, np.asarray(used, dtype=np.int64), axis=axis)
        remapped_indices = _remap_indices(indices_array, used, axis_dim)

        old_bytes = int(table_array.nbytes)
        new_bytes = int(pruned_table.nbytes)
        if new_bytes >= old_bytes:
            continue

        _replace_initializer_array(model, table_name, pruned_table)
        if indices_kind == "initializer":
            _replace_initializer_array(model, indices_name, remapped_indices)
        else:
            assert isinstance(indices_ref, onnx.NodeProto)
            _replace_constant_array(indices_ref, remapped_indices)

        changed_initializer_names.add(table_name)
        tables_pruned += 1
        rows_removed += axis_dim - len(used)
        bytes_removed += old_bytes - new_bytes

    if changed_initializer_names:
        kept_value_info = [
            value_info
            for value_info in graph.value_info
            if value_info.name not in changed_initializer_names
        ]
        if len(kept_value_info) != len(graph.value_info):
            del graph.value_info[:]
            graph.value_info.extend(kept_value_info)

    return GraphRewriteStats(
        constant_gather_tables_pruned=tables_pruned,
        constant_gather_rows_removed=rows_removed,
        constant_gather_bytes_removed=bytes_removed,
    )


def optimize_model_equivalent(
    input_model: str,
    output_model: str,
    passes: tuple[str, ...] = ("dead", "const-gather", "dedup"),
) -> dict[str, Any]:
    """Apply conservative graph-equivalent optimizations to one ONNX model."""
    input_path = Path(input_model)
    output_path = Path(output_model)
    if not input_path.is_file():
        raise FileNotFoundError(f"model does not exist: {input_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    onnx.checker.check_model(model)
    source_cost = estimate_model_cost(str(input_path))
    source_initializer_count = len(model.graph.initializer)

    stats = GraphRewriteStats()
    for pass_name in passes:
        if pass_name == "dead":
            stats = stats.plus(_prune_dead_graph(model))
        elif pass_name == "dedup":
            stats = stats.plus(_deduplicate_initializers(model))
        elif pass_name == "const-gather":
            stats = stats.plus(_prune_constant_gather_tables(model))
        else:
            raise ValueError(f"unsupported pass: {pass_name}")

    if "dead" in passes:
        stats = stats.plus(_prune_dead_graph(model))

    if not stats.changed():
        shutil.copyfile(input_path, output_path)
        output_cost = source_cost
        output_initializer_count = source_initializer_count
    else:
        onnx.checker.check_model(model)
        onnx.save(model, str(output_path))
        onnx.checker.check_model(str(output_path))
        output_cost = estimate_model_cost(str(output_path))
        output_initializer_count = len(model.graph.initializer)

    return {
        "source_model_path": str(input_path),
        "output_model_path": str(output_path),
        "source_estimated_cost": int(source_cost["estimated_cost"]),
        "output_estimated_cost": int(output_cost["estimated_cost"]),
        "estimated_cost_delta": int(output_cost["estimated_cost"])
        - int(source_cost["estimated_cost"]),
        "source_file_size_bytes": int(source_cost["file_size_bytes"]),
        "output_file_size_bytes": int(output_cost["file_size_bytes"]),
        "file_size_delta": int(output_cost["file_size_bytes"])
        - int(source_cost["file_size_bytes"]),
        "source_initializer_bytes": int(source_cost["initializer_memory_bytes"]),
        "output_initializer_bytes": int(output_cost["initializer_memory_bytes"]),
        "initializer_bytes_delta": int(output_cost["initializer_memory_bytes"])
        - int(source_cost["initializer_memory_bytes"]),
        "source_initializer_count": source_initializer_count,
        "output_initializer_count": output_initializer_count,
        "removed_dead_nodes": stats.removed_dead_nodes,
        "removed_unused_initializers": stats.removed_unused_initializers,
        "removed_unused_value_info": stats.removed_unused_value_info,
        "deduplicated_initializers": stats.deduplicated_initializers,
        "constant_gather_tables_pruned": stats.constant_gather_tables_pruned,
        "constant_gather_rows_removed": stats.constant_gather_rows_removed,
        "constant_gather_bytes_removed": stats.constant_gather_bytes_removed,
        "changed": stats.changed(),
    }


def _run_session(
    session: ort.InferenceSession,
    input_tensor: np.ndarray,
) -> list[np.ndarray]:
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise ValueError(f"model must have exactly one input, got {len(inputs)}")
    return session.run(None, {inputs[0].name: input_tensor.astype(np.float32, copy=False)})


def _random_onehot(rng: np.random.Generator) -> np.ndarray:
    colors = rng.integers(0, 10, size=(30, 30), dtype=np.int64)
    return grid_to_onehot(colors.astype(int).tolist())


def _nonzero_padding_onehot(rng: np.random.Generator) -> np.ndarray:
    tensor = grid_to_onehot([[2 for _ in range(10)] for _ in range(10)])
    pad_colors = rng.integers(0, 10, size=(30, 30), dtype=np.int64)
    tensor[:, :, 10:, :] = 0.0
    tensor[:, :, :, 10:] = 0.0
    for row in range(30):
        for col in range(30):
            if row < 10 and col < 10:
                continue
            tensor[0, int(pad_colors[row, col]), row, col] = 1.0
    return tensor


def _equivalence_inputs(task_path: str, fuzz_count: int, seed: int) -> list[np.ndarray]:
    tensors: list[np.ndarray] = []
    task = load_task(task_path)
    for case in task.get("train", []):
        tensors.append(grid_to_onehot(case["input"]))

    tensors.append(grid_to_onehot([[0 for _ in range(30)] for _ in range(30)]))
    tensors.append(grid_to_onehot([[1 for _ in range(30)] for _ in range(30)]))
    single = [[0 for _ in range(30)] for _ in range(30)]
    single[15][15] = 7
    tensors.append(grid_to_onehot(single))

    rng = np.random.default_rng(seed)
    tensors.append(_random_onehot(rng))
    tensors.append(_nonzero_padding_onehot(rng))
    for _ in range(fuzz_count):
        tensors.append(_random_onehot(rng))
    return tensors


def validate_model_equivalence(
    source_model: str,
    output_model: str,
    task_path: str,
    fuzz_count: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Require old and rewritten models to produce exactly equal tensors."""
    old_session = ort.InferenceSession(source_model, providers=["CPUExecutionProvider"])
    new_session = ort.InferenceSession(output_model, providers=["CPUExecutionProvider"])
    max_abs_diff = 0.0
    checked = 0

    for tensor in _equivalence_inputs(task_path, fuzz_count=fuzz_count, seed=seed):
        old_outputs = _run_session(old_session, tensor)
        new_outputs = _run_session(new_session, tensor)
        if len(old_outputs) != len(new_outputs):
            return {
                "passed": False,
                "inputs_checked": checked,
                "max_abs_diff": max_abs_diff,
                "failure_reason": "output_count_mismatch",
            }
        for old, new in zip(old_outputs, new_outputs):
            if old.shape != new.shape:
                return {
                    "passed": False,
                    "inputs_checked": checked,
                    "max_abs_diff": max_abs_diff,
                    "failure_reason": f"output_shape_mismatch: {old.shape} != {new.shape}",
                }
            if np.array_equal(old, new):
                diff = 0.0
            elif old.dtype == np.bool_ or new.dtype == np.bool_:
                diff = float(np.any(np.logical_xor(old, new)))
            else:
                diff = float(np.max(np.abs(old - new))) if old.size else 0.0
            max_abs_diff = max(max_abs_diff, diff)
            if not np.array_equal(old, new):
                return {
                    "passed": False,
                    "inputs_checked": checked,
                    "max_abs_diff": max_abs_diff,
                    "failure_reason": "tensor_values_changed",
                }
        checked += 1

    return {
        "passed": True,
        "inputs_checked": checked,
        "max_abs_diff": max_abs_diff,
        "failure_reason": "",
    }


def validate_one_candidate(
    source_model: str,
    candidate_model: str,
    task_path: str,
    fuzz_count: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Validate one already-built candidate against its source model."""
    failure_reasons: list[str] = []
    checker_passed = False
    forbidden_passed = False
    static_passed = False
    equivalence_passed = False
    equivalence_inputs = 0
    equivalence_max_abs_diff: str | float = ""

    try:
        onnx.checker.check_model(candidate_model)
        checker_passed = True
    except Exception as exc:
        failure_reasons.append(f"checker: {exc}")

    try:
        forbidden = check_forbidden_ops(candidate_model)
        forbidden_passed = bool(forbidden["passed"])
        if not forbidden_passed:
            failure_reasons.append(f"forbidden_ops={forbidden['forbidden_ops_found']}")
    except Exception as exc:
        failure_reasons.append(f"forbidden check: {exc}")

    try:
        static = check_static_shapes(candidate_model)
        static_passed = bool(static["passed"])
        if not static_passed:
            failure_reasons.append(f"static_shapes={static['failures'][:3]}")
    except Exception as exc:
        failure_reasons.append(f"static shape check: {exc}")

    if checker_passed and forbidden_passed and static_passed:
        try:
            equivalence = validate_model_equivalence(
                source_model,
                candidate_model,
                task_path,
                fuzz_count=fuzz_count,
                seed=seed,
            )
            equivalence_passed = bool(equivalence["passed"])
            equivalence_inputs = int(equivalence["inputs_checked"])
            equivalence_max_abs_diff = equivalence["max_abs_diff"]
            if not equivalence_passed:
                failure_reasons.append(str(equivalence["failure_reason"]))
        except Exception as exc:
            failure_reasons.append(f"equivalence exception: {exc}")

    return {
        "checker_passed": checker_passed,
        "forbidden_ops_passed": forbidden_passed,
        "static_shapes_passed": static_passed,
        "equivalence_passed": equivalence_passed,
        "equivalence_inputs_checked": equivalence_inputs,
        "equivalence_max_abs_diff": equivalence_max_abs_diff,
        "candidate_valid": checker_passed
        and forbidden_passed
        and static_passed
        and equivalence_passed,
        "failure_reason": "; ".join(failure_reasons),
    }


def _model_analysis_row(model_path: Path, task_id: str, lane: str, peer_sha: str) -> dict[str, Any]:
    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)
    op_types = [node.op_type for node in model.graph.node]
    init_bytes = sum(_initializer_bytes(initializer) for initializer in model.graph.initializer)
    init_signature_parts = []
    large_initializers = []
    for initializer in model.graph.initializer:
        dtype_name = TensorProto.DataType.Name(initializer.data_type)
        shape = "x".join(str(dim) for dim in initializer.dims) or "scalar"
        numel = _tensor_numel(initializer.dims)
        init_signature_parts.append(f"{dtype_name}:{shape}:{numel}")
        large_initializers.append(
            (
                _initializer_bytes(initializer),
                f"{initializer.name}:{dtype_name}:{shape}:{numel}:{_tensor_hash_prefix(initializer)}",
            )
        )

    model_sha = _sha256_path(model_path)
    return {
        "task_id": task_id,
        "lane": lane,
        "model_path": str(model_path),
        "file_size_bytes": model_path.stat().st_size,
        "initializer_bytes": init_bytes,
        "initializer_count": len(model.graph.initializer),
        "node_count": len(model.graph.node),
        "op_count": len(op_types),
        "gather_count": op_types.count("Gather"),
        "conv_count": op_types.count("Conv"),
        "constant_count": op_types.count("Constant"),
        "op_sequence": ">".join(op_types),
        "initializer_shape_signature": "|".join(init_signature_parts),
        "large_initializer_topk": ";".join(
            item for _nbytes, item in sorted(large_initializers, reverse=True)[:5]
        ),
        "model_sha256": model_sha,
        "peer_sha256": peer_sha,
        "same_as_peer": bool(peer_sha and model_sha == peer_sha),
        "template_id": "",
    }


def analyze_stack(
    stack_dir: str = "outputs/reference_6348_56_stack",
    report_path: str = "outputs/reports/ref6348_graph_fingerprints_20260612.csv",
) -> dict[str, Any]:
    """Write P0 graph fingerprints for every model in a hybrid stack."""
    root = Path(stack_dir)
    rows: list[dict[str, Any]] = []
    task_ids = sorted(path.stem for path in (root / "base_submission").glob("task*.onnx"))
    peer_hash: dict[tuple[str, str], str] = {}
    for task_id in task_ids:
        for lane, peer_lane in (("base_submission", "overrides"), ("overrides", "base_submission")):
            peer = root / peer_lane / f"{task_id}.onnx"
            peer_hash[(task_id, lane)] = _sha256_path(peer) if peer.is_file() else ""

    for task_id in task_ids:
        for lane in HYBRID_STACK_DIRS:
            model_path = root / lane / f"{task_id}.onnx"
            if model_path.is_file():
                rows.append(
                    _model_analysis_row(
                        model_path,
                        task_id=task_id,
                        lane=lane,
                        peer_sha=peer_hash.get((task_id, lane), ""),
                    )
                )

    template_ids: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["op_sequence"], row["initializer_shape_signature"])
        if key not in template_ids:
            template_ids[key] = f"tpl{len(template_ids) + 1:04d}"
        row["template_id"] = template_ids[key]

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANALYSIS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    by_template: dict[str, dict[str, int]] = {}
    for row in rows:
        item = by_template.setdefault(row["template_id"], {"count": 0, "initializer_bytes": 0})
        item["count"] += 1
        item["initializer_bytes"] += int(row["initializer_bytes"])

    top_templates = sorted(
        (
            {"template_id": template_id, **values}
            for template_id, values in by_template.items()
        ),
        key=lambda item: (-item["initializer_bytes"], item["template_id"]),
    )[:20]

    summary = {
        "stack_dir": str(root),
        "report_path": str(report),
        "models_analyzed": len(rows),
        "task_ids": len(task_ids),
        "template_count": len(template_ids),
        "byte_identical_lane_tasks": len(
            {
                row["task_id"]
                for row in rows
                if row["lane"] == "base_submission" and row["same_as_peer"]
            }
        ),
        "top_templates": top_templates,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_csv_set(raw: str) -> set[str] | None:
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _iter_stack_models(
    stack_dir: Path,
    task_ids: set[str] | None,
    lanes: set[str] | None,
) -> list[tuple[str, str, Path]]:
    selected_lanes = sorted(lanes or set(HYBRID_STACK_DIRS))
    models: list[tuple[str, str, Path]] = []
    for lane in selected_lanes:
        if lane not in HYBRID_STACK_DIRS:
            raise ValueError(f"unsupported lane: {lane}")
        for path in sorted((stack_dir / lane).glob("task*.onnx")):
            if task_ids is not None and path.stem not in task_ids:
                continue
            models.append((path.stem, lane, path))
    return models


def optimize_stack(
    stack_dir: str = "outputs/reference_6348_56_stack",
    task_dir: str = "task",
    output_dir: str = "outputs/candidates/ref6348_equiv_optimized_stack",
    report_path: str = "outputs/reports/ref6348_equiv_optimized_stack.csv",
    task_ids: set[str] | None = None,
    lanes: set[str] | None = None,
    passes: tuple[str, ...] = ("dead", "const-gather", "dedup"),
    fuzz_count: int = 20,
    max_models: int = 0,
    validate_equivalence: bool = True,
) -> dict[str, Any]:
    """Optimize selected hybrid-stack models and write a lane/task report."""
    stack_root = Path(stack_dir)
    output_root = Path(output_dir)
    report = Path(report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    selected_models = _iter_stack_models(stack_root, task_ids=task_ids, lanes=lanes)
    if max_models > 0:
        selected_models = selected_models[:max_models]

    rows: list[dict[str, Any]] = []
    for task_id, lane, source_path in selected_models:
        output_path = output_root / lane / f"{task_id}.onnx"
        row: dict[str, Any] = {
            "task_id": task_id,
            "lane": lane,
            "source_model_path": str(source_path),
            "output_model_path": "",
            "checker_passed": False,
            "forbidden_ops_passed": False,
            "static_shapes_passed": False,
            "equivalence_passed": False,
            "equivalence_inputs_checked": 0,
            "equivalence_max_abs_diff": "",
            "candidate_valid": False,
            "failure_reason": "",
        }
        try:
            result = optimize_model_equivalent(
                str(source_path),
                str(output_path),
                passes=passes,
            )
            row.update(result)
            checker_passed = False
            forbidden_passed = False
            static_passed = False
            equivalence_passed = not validate_equivalence
            equivalence_inputs = 0
            equivalence_max_abs_diff: str | float = ""
            failure_reasons: list[str] = []

            try:
                onnx.checker.check_model(str(output_path))
                checker_passed = True
            except Exception as exc:
                failure_reasons.append(f"checker: {exc}")
            try:
                forbidden = check_forbidden_ops(str(output_path))
                forbidden_passed = bool(forbidden["passed"])
                if not forbidden_passed:
                    failure_reasons.append(f"forbidden_ops={forbidden['forbidden_ops_found']}")
            except Exception as exc:
                failure_reasons.append(f"forbidden check: {exc}")
            try:
                static = check_static_shapes(str(output_path))
                static_passed = bool(static["passed"])
                if not static_passed:
                    failure_reasons.append(f"static_shapes={static['failures'][:3]}")
            except Exception as exc:
                failure_reasons.append(f"static shape check: {exc}")

            if validate_equivalence and bool(result["changed"]):
                task_path = Path(task_dir) / f"{task_id}.json"
                seed = int(task_id[-3:])
                equivalence = validate_model_equivalence(
                    str(source_path),
                    str(output_path),
                    str(task_path),
                    fuzz_count=fuzz_count,
                    seed=seed,
                )
                equivalence_passed = bool(equivalence["passed"])
                equivalence_inputs = int(equivalence["inputs_checked"])
                equivalence_max_abs_diff = equivalence["max_abs_diff"]
                if not equivalence_passed:
                    failure_reasons.append(str(equivalence["failure_reason"]))
            elif not bool(result["changed"]):
                equivalence_passed = True

            candidate_valid = (
                bool(result["changed"])
                and checker_passed
                and forbidden_passed
                and static_passed
                and equivalence_passed
            )
            row.update(
                {
                    "output_model_path": str(output_path) if candidate_valid else "",
                    "checker_passed": checker_passed,
                    "forbidden_ops_passed": forbidden_passed,
                    "static_shapes_passed": static_passed,
                    "equivalence_passed": equivalence_passed,
                    "equivalence_inputs_checked": equivalence_inputs,
                    "equivalence_max_abs_diff": equivalence_max_abs_diff,
                    "candidate_valid": candidate_valid,
                    "failure_reason": "; ".join(failure_reasons),
                }
            )
        except Exception as exc:
            row["failure_reason"] = str(exc)
        rows.append(row)

    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPTIMIZE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row.get("candidate_valid")]
    summary = {
        "stack_dir": str(stack_root),
        "output_dir": str(output_root),
        "report_path": str(report),
        "models_scanned": len(rows),
        "valid_candidates": len(valid_rows),
        "total_estimated_cost_delta": sum(int(row["estimated_cost_delta"]) for row in valid_rows),
        "total_file_size_delta": sum(int(row["file_size_delta"]) for row in valid_rows),
        "total_initializer_bytes_delta": sum(int(row["initializer_bytes_delta"]) for row in valid_rows),
        "total_dead_nodes_removed": sum(int(row["removed_dead_nodes"]) for row in valid_rows),
        "total_deduplicated_initializers": sum(
            int(row["deduplicated_initializers"]) for row in valid_rows
        ),
        "total_constant_gather_tables_pruned": sum(
            int(row["constant_gather_tables_pruned"]) for row in valid_rows
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _subprocess_validate_one(
    source_model: str,
    candidate_model: str,
    task_path: str,
    fuzz_count: int,
    seed: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "src.hybrid_stack_optimizer",
        "validate-one",
        "--source-model",
        source_model,
        "--candidate-model",
        candidate_model,
        "--task",
        task_path,
        "--fuzz-count",
        str(fuzz_count),
        "--seed",
        str(seed),
    ]
    completed = subprocess.run(
        command,
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "checker_passed": False,
            "forbidden_ops_passed": False,
            "static_shapes_passed": False,
            "equivalence_passed": False,
            "equivalence_inputs_checked": 0,
            "equivalence_max_abs_diff": "",
            "candidate_valid": False,
            "failure_reason": (
                f"validation subprocess exited {completed.returncode}: "
                f"{(completed.stderr or completed.stdout).strip()[:500]}"
            ),
        }
    try:
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            "checker_passed": False,
            "forbidden_ops_passed": False,
            "static_shapes_passed": False,
            "equivalence_passed": False,
            "equivalence_inputs_checked": 0,
            "equivalence_max_abs_diff": "",
            "candidate_valid": False,
            "failure_reason": f"could not parse validation subprocess output: {exc}",
        }


def validate_candidate_report(
    input_report: str,
    task_dir: str = "task",
    output_report: str = "outputs/reports/ref6348_equiv_optimized_stack_strict.csv",
    task_ids: set[str] | None = None,
    lanes: set[str] | None = None,
    fuzz_count: int = 20,
    max_candidates: int = 0,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    """Strictly validate a graph-only candidate report using one subprocess per row."""
    report_in = Path(input_report)
    if not report_in.is_file():
        raise FileNotFoundError(f"candidate report does not exist: {input_report}")

    with report_in.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    candidate_indexes = [
        index
        for index, row in enumerate(rows)
        if row.get("candidate_valid") == "True"
        and row.get("output_model_path")
        and (task_ids is None or row.get("task_id") in task_ids)
        and (lanes is None or row.get("lane") in lanes)
    ]
    candidate_indexes.sort(
        key=lambda index: (
            int(rows[index].get("file_size_delta") or 0),
            int(rows[index].get("initializer_bytes_delta") or 0),
            rows[index].get("task_id", ""),
            rows[index].get("lane", ""),
        )
    )
    if max_candidates > 0:
        candidate_indexes = candidate_indexes[:max_candidates]

    for index in candidate_indexes:
        row = rows[index]
        task_id = row["task_id"]
        validation = _subprocess_validate_one(
            source_model=row["source_model_path"],
            candidate_model=row["output_model_path"],
            task_path=str(Path(task_dir) / f"{task_id}.json"),
            fuzz_count=fuzz_count,
            seed=int(task_id[-3:]),
            timeout_seconds=timeout_seconds,
        )
        row.update(validation)
        if not validation["candidate_valid"]:
            row["output_model_path"] = ""

    selected = set(candidate_indexes)
    for index, row in enumerate(rows):
        if index not in selected:
            row["candidate_valid"] = False
            row["output_model_path"] = ""
            if row.get("changed") == "True" and not row.get("failure_reason"):
                row["failure_reason"] = "not selected for strict validation"

    output = Path(output_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPTIMIZE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [
        row
        for row in rows
        if row.get("candidate_valid") is True or row.get("candidate_valid") == "True"
    ]
    summary = {
        "input_report": str(report_in),
        "output_report": str(output),
        "selected_candidates": len(candidate_indexes),
        "valid_candidates": len(valid_rows),
        "failed_candidates": len(candidate_indexes) - len(valid_rows),
        "total_estimated_cost_delta": sum(int(row["estimated_cost_delta"]) for row in valid_rows),
        "total_file_size_delta": sum(int(row["file_size_delta"]) for row in valid_rows),
        "total_initializer_bytes_delta": sum(int(row["initializer_bytes_delta"]) for row in valid_rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _load_valid_candidate_rows(report_path: Path) -> list[dict[str, str]]:
    with report_path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("candidate_valid") == "True"]
    rows.sort(
        key=lambda row: (
            int(row.get("file_size_delta") or 0),
            int(row.get("initializer_bytes_delta") or 0),
            row["task_id"],
            row["lane"],
        )
    )
    return rows


def build_lane_ablations(
    base_zip: str = "outputs/submissions/6348_56_hybrid_stack_submission.zip",
    candidate_report: str = "outputs/reports/ref6348_equiv_optimized_stack.csv",
    output_dir: str = "outputs/ablation_submissions/ref6348_equiv_optimized_lane_20260612",
    report_path: str = "outputs/reports/ref6348_equiv_optimized_lane_ablations.csv",
    task_ids: set[str] | None = None,
    lanes: set[str] | None = None,
    max_candidates: int = 20,
    upload_friendly_folders: bool = True,
    inspect_first: bool = True,
) -> dict[str, Any]:
    """Build one-task, one-lane hybrid ablations for valid optimized models."""
    base_path = Path(base_zip)
    inspect_submission(str(base_path), layout="hybrid_stack")
    output_root = Path(output_dir)
    report = Path(report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(base_path, "r") as archive:
        base_entries = {
            name: archive.read(name)
            for name in sorted(archive.namelist())
            if not name.endswith("/")
        }

    candidate_rows = _load_valid_candidate_rows(Path(candidate_report))
    if task_ids is not None:
        candidate_rows = [row for row in candidate_rows if row["task_id"] in task_ids]
    if lanes is not None:
        candidate_rows = [row for row in candidate_rows if row["lane"] in lanes]
    if max_candidates > 0:
        candidate_rows = candidate_rows[:max_candidates]

    rows: list[dict[str, Any]] = []
    first_valid_zip = ""
    for row in candidate_rows:
        task_id = row["task_id"]
        lane = row["lane"]
        candidate_model = Path(row["output_model_path"])
        entry_name = f"{lane}/{task_id}.onnx"
        output_zip = output_root / f"{task_id}_{lane}_EquivOptimized.zip"
        folder_lane = lane.replace("_submission", "").replace("/", "_")
        upload_path = output_root / f"{task_id}_{folder_lane}_EquivOptimized" / "submission.zip"
        try:
            if entry_name not in base_entries:
                raise ValueError(f"base zip missing {entry_name}")
            candidate_data = candidate_model.read_bytes()
            candidate_sha = _sha256_bytes(candidate_data)
            source_sha = _sha256_bytes(base_entries[entry_name])
            if candidate_sha == source_sha:
                raise ValueError("candidate identical to base entry")
            output_zip.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for name, data in base_entries.items():
                    if name == entry_name:
                        archive.writestr(name, candidate_data)
                    else:
                        archive.writestr(name, data)
            upload_submission_path = ""
            if upload_friendly_folders:
                upload_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(output_zip, upload_path)
                upload_submission_path = str(upload_path)
            if not first_valid_zip:
                first_valid_zip = str(output_zip)
            rows.append(
                {
                    "task_id": task_id,
                    "lane": lane,
                    "candidate_model_path": str(candidate_model),
                    "candidate_zip_path": str(output_zip),
                    "upload_submission_path": upload_submission_path,
                    "base_entry_replaced": entry_name,
                    "source_entry_sha256": source_sha,
                    "candidate_sha256": candidate_sha,
                    "candidate_valid": True,
                    "failure_reason": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "task_id": task_id,
                    "lane": lane,
                    "candidate_model_path": str(candidate_model),
                    "candidate_zip_path": str(output_zip),
                    "upload_submission_path": "",
                    "base_entry_replaced": entry_name,
                    "source_entry_sha256": "",
                    "candidate_sha256": "",
                    "candidate_valid": False,
                    "failure_reason": str(exc),
                }
            )

    if inspect_first and first_valid_zip:
        inspect_submission(first_valid_zip, layout="hybrid_stack")

    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ABLATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "base_zip": str(base_path),
        "candidate_report": str(candidate_report),
        "output_dir": str(output_root),
        "report_path": str(report),
        "candidate_rows": len(candidate_rows),
        "valid_zip_count": sum(1 for row in rows if row["candidate_valid"]),
        "failed_count": sum(1 for row in rows if not row["candidate_valid"]),
        "first_valid_zip_inspected": first_valid_zip if inspect_first else "",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def build_merged_submission(
    base_zip: str = "outputs/submissions/6348_56_hybrid_stack_submission.zip",
    candidate_report: str = "outputs/reports/ref6348_equiv_optimized_stack_strict.csv",
    output_zip: str = "outputs/ablation_submissions/ref6348_equiv_optimized_merged_20260613/submission.zip",
    report_path: str = "outputs/reports/ref6348_equiv_optimized_merged_20260613.csv",
    task_ids: set[str] | None = None,
    lanes: set[str] | None = None,
    max_candidates: int = 0,
    inspect_output: bool = True,
) -> dict[str, Any]:
    """Build one hybrid-stack submission with all selected valid replacements."""
    base_path = Path(base_zip)
    inspect_submission(str(base_path), layout="hybrid_stack")
    output_path = Path(output_zip)
    report = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(base_path, "r") as archive:
        base_entries = {
            name: archive.read(name)
            for name in sorted(archive.namelist())
            if not name.endswith("/")
        }

    candidate_rows = _load_valid_candidate_rows(Path(candidate_report))
    if task_ids is not None:
        candidate_rows = [row for row in candidate_rows if row["task_id"] in task_ids]
    if lanes is not None:
        candidate_rows = [row for row in candidate_rows if row["lane"] in lanes]
    if max_candidates > 0:
        candidate_rows = candidate_rows[:max_candidates]
    if not candidate_rows:
        raise ValueError("no valid candidate rows selected for merge")

    replacements: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        task_id = row["task_id"]
        lane = row["lane"]
        candidate_model = Path(row["output_model_path"])
        entry_name = f"{lane}/{task_id}.onnx"
        try:
            if entry_name in replacements:
                raise ValueError(f"duplicate replacement entry: {entry_name}")
            if entry_name not in base_entries:
                raise ValueError(f"base zip missing {entry_name}")
            candidate_data = candidate_model.read_bytes()
            source_sha = _sha256_bytes(base_entries[entry_name])
            candidate_sha = _sha256_bytes(candidate_data)
            if candidate_sha == source_sha:
                raise ValueError("candidate identical to base entry")
            replacements[entry_name] = candidate_data
            rows.append(
                {
                    "task_id": task_id,
                    "lane": lane,
                    "candidate_model_path": str(candidate_model),
                    "base_entry_replaced": entry_name,
                    "source_entry_sha256": source_sha,
                    "candidate_sha256": candidate_sha,
                    "candidate_valid": True,
                    "failure_reason": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "task_id": task_id,
                    "lane": lane,
                    "candidate_model_path": str(candidate_model),
                    "base_entry_replaced": entry_name,
                    "source_entry_sha256": "",
                    "candidate_sha256": "",
                    "candidate_valid": False,
                    "failure_reason": str(exc),
                }
            )

    failed_rows = [row for row in rows if not row["candidate_valid"]]
    if failed_rows:
        with report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MERGE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        raise ValueError(f"merge has invalid candidate rows: {len(failed_rows)}")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in base_entries.items():
            archive.writestr(name, replacements.get(name, data))

    inspection: dict[str, Any] = {}
    if inspect_output:
        inspection = dict(inspect_submission(str(output_path), layout="hybrid_stack"))

    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MERGE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "base_zip": str(base_path),
        "candidate_report": str(candidate_report),
        "output_zip": str(output_path),
        "report_path": str(report),
        "candidate_rows": len(candidate_rows),
        "merged_replacements": len(replacements),
        "failed_count": 0,
        "total_estimated_cost_delta": sum(
            int(row.get("estimated_cost_delta") or 0) for row in candidate_rows
        ),
        "total_file_size_delta": sum(
            int(row.get("file_size_delta") or 0) for row in candidate_rows
        ),
        "total_initializer_bytes_delta": sum(
            int(row.get("initializer_bytes_delta") or 0) for row in candidate_rows
        ),
        "inspection": inspection,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _passes(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or ("dead", "const-gather", "dedup")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Write graph fingerprint report")
    analyze.add_argument("--stack-dir", default="outputs/reference_6348_56_stack")
    analyze.add_argument(
        "--report",
        default="outputs/reports/ref6348_graph_fingerprints_20260612.csv",
    )

    optimize = subparsers.add_parser("optimize", help="Build graph-equivalent candidates")
    optimize.add_argument("--stack-dir", default="outputs/reference_6348_56_stack")
    optimize.add_argument("--task-dir", default="task")
    optimize.add_argument(
        "--output-dir",
        default="outputs/candidates/ref6348_equiv_optimized_stack",
    )
    optimize.add_argument(
        "--report",
        default="outputs/reports/ref6348_equiv_optimized_stack.csv",
    )
    optimize.add_argument("--task-ids", default="")
    optimize.add_argument("--lanes", default="")
    optimize.add_argument("--passes", default="dead,const-gather,dedup")
    optimize.add_argument("--fuzz-count", type=int, default=20)
    optimize.add_argument("--max-models", type=int, default=0)
    optimize.add_argument("--no-equivalence-validation", action="store_true")

    ablate = subparsers.add_parser("ablate", help="Build one-task one-lane ablation zips")
    ablate.add_argument("--base-zip", default="outputs/submissions/6348_56_hybrid_stack_submission.zip")
    ablate.add_argument(
        "--candidate-report",
        default="outputs/reports/ref6348_equiv_optimized_stack.csv",
    )
    ablate.add_argument(
        "--output-dir",
        default="outputs/ablation_submissions/ref6348_equiv_optimized_lane_20260612",
    )
    ablate.add_argument(
        "--report",
        default="outputs/reports/ref6348_equiv_optimized_lane_ablations.csv",
    )
    ablate.add_argument("--task-ids", default="")
    ablate.add_argument("--lanes", default="")
    ablate.add_argument("--max-candidates", type=int, default=20)
    ablate.add_argument("--no-upload-friendly-folders", action="store_true")
    ablate.add_argument("--no-inspect-first", action="store_true")

    merge = subparsers.add_parser("merge", help="Build one submission with selected valid replacements")
    merge.add_argument("--base-zip", default="outputs/submissions/6348_56_hybrid_stack_submission.zip")
    merge.add_argument(
        "--candidate-report",
        default="outputs/reports/ref6348_equiv_optimized_stack_strict.csv",
    )
    merge.add_argument(
        "--output-zip",
        default="outputs/ablation_submissions/ref6348_equiv_optimized_merged_20260613/submission.zip",
    )
    merge.add_argument(
        "--report",
        default="outputs/reports/ref6348_equiv_optimized_merged_20260613.csv",
    )
    merge.add_argument("--task-ids", default="")
    merge.add_argument("--lanes", default="")
    merge.add_argument("--max-candidates", type=int, default=0)
    merge.add_argument("--no-inspect-output", action="store_true")

    validate = subparsers.add_parser(
        "validate-report",
        help="Strictly validate graph-only candidate report rows",
    )
    validate.add_argument(
        "--input-report",
        default="outputs/reports/ref6348_equiv_optimized_stack_graphonly.csv",
    )
    validate.add_argument("--task-dir", default="task")
    validate.add_argument(
        "--output-report",
        default="outputs/reports/ref6348_equiv_optimized_stack_strict.csv",
    )
    validate.add_argument("--task-ids", default="")
    validate.add_argument("--lanes", default="")
    validate.add_argument("--fuzz-count", type=int, default=20)
    validate.add_argument("--max-candidates", type=int, default=0)
    validate.add_argument("--timeout-seconds", type=int, default=180)

    validate_one = subparsers.add_parser("validate-one", help=argparse.SUPPRESS)
    validate_one.add_argument("--source-model", required=True)
    validate_one.add_argument("--candidate-model", required=True)
    validate_one.add_argument("--task", required=True)
    validate_one.add_argument("--fuzz-count", type=int, default=20)
    validate_one.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    if args.command == "analyze":
        analyze_stack(stack_dir=args.stack_dir, report_path=args.report)
    elif args.command == "optimize":
        optimize_stack(
            stack_dir=args.stack_dir,
            task_dir=args.task_dir,
            output_dir=args.output_dir,
            report_path=args.report,
            task_ids=_parse_csv_set(args.task_ids),
            lanes=_parse_csv_set(args.lanes),
            passes=_passes(args.passes),
            fuzz_count=args.fuzz_count,
            max_models=args.max_models,
            validate_equivalence=not args.no_equivalence_validation,
        )
    elif args.command == "ablate":
        build_lane_ablations(
            base_zip=args.base_zip,
            candidate_report=args.candidate_report,
            output_dir=args.output_dir,
            report_path=args.report,
            task_ids=_parse_csv_set(args.task_ids),
            lanes=_parse_csv_set(args.lanes),
            max_candidates=args.max_candidates,
            upload_friendly_folders=not args.no_upload_friendly_folders,
            inspect_first=not args.no_inspect_first,
        )
    elif args.command == "merge":
        build_merged_submission(
            base_zip=args.base_zip,
            candidate_report=args.candidate_report,
            output_zip=args.output_zip,
            report_path=args.report,
            task_ids=_parse_csv_set(args.task_ids),
            lanes=_parse_csv_set(args.lanes),
            max_candidates=args.max_candidates,
            inspect_output=not args.no_inspect_output,
        )
    elif args.command == "validate-report":
        validate_candidate_report(
            input_report=args.input_report,
            task_dir=args.task_dir,
            output_report=args.output_report,
            task_ids=_parse_csv_set(args.task_ids),
            lanes=_parse_csv_set(args.lanes),
            fuzz_count=args.fuzz_count,
            max_candidates=args.max_candidates,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "validate-one":
        print(
            json.dumps(
                validate_one_candidate(
                    source_model=args.source_model,
                    candidate_model=args.candidate_model,
                    task_path=args.task,
                    fuzz_count=args.fuzz_count,
                    seed=args.seed,
                ),
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
