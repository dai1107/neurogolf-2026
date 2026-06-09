"""Inspect and prune task076 current-model dihedral permutation rows."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .cost_estimator import estimate_model_cost


DEFAULT_SOURCE = "outputs/onnx/task076.onnx"
DEFAULT_OUTPUT_DIR = "outputs/candidates/task076_perm_prune"
DEFAULT_REPORT = "outputs/reports/task076_perm_prune_report.csv"
DEFAULT_GATHER_OUTPUT = "outputs/candidates/task076_perm_gather/task076_Task076PermGatherExact.onnx"
DEFAULT_GATHER_REPORT = "outputs/reports/task076_perm_gather_report.csv"
MODE_CHOICES = (
    "keep_0_1_2_3",
    "keep_0_1_2_3_4_5",
    "keep_0_1_2_3_6_7",
    "keep_0_1_4_5_6_7",
    "drop_4",
    "drop_5",
    "drop_6",
    "drop_7",
)
MODE_INDICES = {
    "keep_0_1_2_3": (0, 1, 2, 3),
    "keep_0_1_2_3_4_5": (0, 1, 2, 3, 4, 5),
    "keep_0_1_2_3_6_7": (0, 1, 2, 3, 6, 7),
    "keep_0_1_4_5_6_7": (0, 1, 4, 5, 6, 7),
    "drop_4": (0, 1, 2, 3, 5, 6, 7),
    "drop_5": (0, 1, 2, 3, 4, 6, 7),
    "drop_6": (0, 1, 2, 3, 4, 5, 7),
    "drop_7": (0, 1, 2, 3, 4, 5, 6),
}

FIELDS = [
    "mode",
    "kept_indices",
    "source_model_path",
    "output_model_path",
    "source_cost",
    "output_cost",
    "cost_delta",
    "source_file_size_bytes",
    "output_file_size_bytes",
    "file_size_delta",
    "updated_initializers",
    "failure_reason",
]
GATHER_FIELDS = [
    "source_model_path",
    "output_model_path",
    "source_cost",
    "output_cost",
    "cost_delta",
    "source_file_size_bytes",
    "output_file_size_bytes",
    "file_size_delta",
    "removed_nodes",
    "removed_initializers",
    "added_initializers",
    "failure_reason",
]


def _shape_text(shape: tuple[int, ...]) -> str:
    return "x".join(str(dim) for dim in shape) if shape else "scalar"


def inspect_task076_perm_tables(source_model: str = DEFAULT_SOURCE) -> dict[str, Any]:
    """Return a small structural summary for current task076 perm tables."""
    model = onnx.load(source_model)
    arrays = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    values = []
    for name in ("perm_flat", "onnx::Mul_651", "/Reshape_2_output_0"):
        array = arrays.get(name)
        values.append(
            {
                "name": name,
                "present": array is not None,
                "shape": _shape_text(tuple(array.shape)) if array is not None else "",
                "dtype": str(array.dtype) if array is not None else "",
                "size": int(array.size) if array is not None else 0,
                "nbytes": int(array.nbytes) if array is not None else 0,
                "preview": array.reshape(-1)[:12].astype(float).tolist()
                if array is not None and np.issubdtype(array.dtype, np.floating)
                else (array.reshape(-1)[:12].astype(int).tolist() if array is not None else []),
            }
        )
    consumers: dict[str, list[dict[str, Any]]] = {name: [] for name in arrays}
    node_window = []
    direction_constants = []
    for index, node in enumerate(model.graph.node):
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name != "value":
                    continue
                array = numpy_helper.to_array(onnx.helper.get_attribute_value(attr))
                flat = array.reshape(-1) if array.size else array
                if array.size <= 16 and any(int(value) == 8 for value in flat.tolist()):
                    direction_constants.append(
                        {
                            "node_index": index,
                            "node_name": node.name,
                            "output": list(node.output),
                            "shape": _shape_text(tuple(array.shape)),
                            "dtype": str(array.dtype),
                            "values": flat.astype(int).tolist()
                            if np.issubdtype(array.dtype, np.integer)
                            else flat.astype(float).tolist(),
                        }
                    )
        if 120 <= index <= 220:
            node_window.append(
                {
                    "node_index": index,
                    "node_name": node.name,
                    "op_type": node.op_type,
                    "inputs": list(node.input),
                    "outputs": list(node.output),
                    "attributes": {
                        attr.name: onnx.helper.get_attribute_value(attr).tolist()
                        if hasattr(onnx.helper.get_attribute_value(attr), "tolist")
                        else onnx.helper.get_attribute_value(attr)
                        for attr in node.attribute
                    },
                }
            )
        for input_index, input_name in enumerate(node.input):
            if input_name in consumers:
                consumers[input_name].append(
                    {
                        "node_index": index,
                        "node_name": node.name,
                        "op_type": node.op_type,
                        "input_index": input_index,
                        "outputs": list(node.output),
                    }
                )
    return {
        "source_model": source_model,
        "values": values,
        "consumers": {
            name: consumers.get(name, [])
            for name in ("perm_flat", "onnx::Mul_651", "/Reshape_2_output_0")
        },
        "node_window_120_165": node_window,
        "small_constants_containing_8": direction_constants,
    }


def prune_task076_perm_rows(
    source_model: str,
    output_model: str,
    mode: str,
) -> dict[str, Any]:
    """Slice the current task076 8-row permutation table in lockstep."""
    if mode not in MODE_INDICES:
        raise ValueError(f"mode must be one of {', '.join(MODE_CHOICES)}, got {mode}")
    keep = np.asarray(MODE_INDICES[mode], dtype=np.int64)
    source_path = Path(source_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")

    model = onnx.load(str(source_path))
    onnx.checker.check_model(model)
    source_cost = estimate_model_cost(str(source_path))

    updated = 0
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        if initializer.name == "perm_flat":
            if array.shape != (8, 169, 169):
                raise ValueError(f"unexpected perm_flat shape: {array.shape}")
            initializer.CopyFrom(numpy_helper.from_array(array[keep], name=initializer.name))
            updated += 1
        elif initializer.name == "onnx::Mul_651":
            if array.shape != (8,):
                raise ValueError(f"unexpected onnx::Mul_651 shape: {array.shape}")
            initializer.CopyFrom(numpy_helper.from_array(array[keep], name=initializer.name))
            updated += 1

    if updated != 2:
        raise ValueError(f"expected to update 2 initializers, updated {updated}")

    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path))
    output_cost = estimate_model_cost(str(output_path))
    return {
        "mode": mode,
        "kept_indices": ",".join(str(int(value)) for value in keep.tolist()),
        "source_model_path": str(source_path),
        "output_model_path": str(output_path),
        "source_cost": int(source_cost["estimated_cost"]),
        "output_cost": int(output_cost["estimated_cost"]),
        "cost_delta": int(output_cost["estimated_cost"]) - int(source_cost["estimated_cost"]),
        "source_file_size_bytes": int(source_cost["file_size_bytes"]),
        "output_file_size_bytes": int(output_cost["file_size_bytes"]),
        "file_size_delta": int(output_cost["file_size_bytes"]) - int(source_cost["file_size_bytes"]),
        "updated_initializers": updated,
        "failure_reason": "",
    }


def _perm_gather_indices(perm_flat: np.ndarray) -> np.ndarray:
    if perm_flat.shape != (8, 169, 169):
        raise ValueError(f"unexpected perm_flat shape: {perm_flat.shape}")
    if not np.isfinite(perm_flat).all():
        raise ValueError("perm_flat contains non-finite values")
    nonzero_per_row = np.count_nonzero(np.abs(perm_flat) > 0.5, axis=2)
    if not np.all(nonzero_per_row == 1):
        raise ValueError("perm_flat rows are not one-hot permutation rows")
    indices = np.argmax(perm_flat, axis=2).astype(np.int64)
    reconstructed = np.zeros_like(perm_flat)
    for transform in range(indices.shape[0]):
        reconstructed[transform, np.arange(indices.shape[1]), indices[transform]] = 1.0
    if not np.array_equal(reconstructed, perm_flat):
        raise ValueError("perm_flat contains values other than exact one-hot permutation entries")
    return indices


def build_task076_perm_gather_exact(
    source_model: str = DEFAULT_SOURCE,
    output_model: str = DEFAULT_GATHER_OUTPUT,
    report_path: str = DEFAULT_GATHER_REPORT,
) -> dict[str, Any]:
    """Replace task076's dense permutation matrices with exact Gather indices."""
    source_path = Path(source_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")

    model = onnx.load(str(source_path))
    onnx.checker.check_model(model)
    source_cost = estimate_model_cost(str(source_path))

    arrays = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    if "perm_flat" not in arrays:
        raise ValueError("source model is missing perm_flat")
    gather_indices = _perm_gather_indices(arrays["perm_flat"])

    remove_names = {
        "/Unsqueeze_8",
        "/MatMul",
        "/Squeeze_7",
        "/Unsqueeze_9",
        "/MatMul_1",
        "/Squeeze_8",
        "/Unsqueeze_10",
        "/MatMul_2",
        "/Squeeze_9",
    }
    new_nodes = [
        helper.make_node(
            "Gather",
            ["/Reshape_9_output_0", "PermGatherIdx"],
            ["/Squeeze_7_output_0"],
            name="/Task076PermGather0",
            axis=0,
        ),
        helper.make_node(
            "Gather",
            ["/Reshape_10_output_0", "PermGatherIdx"],
            ["/Squeeze_8_output_0"],
            name="/Task076PermGather1",
            axis=0,
        ),
        helper.make_node(
            "Gather",
            ["/Reshape_11_output_0", "PermGatherIdx"],
            ["/Squeeze_9_output_0"],
            name="/Task076PermGather2",
            axis=0,
        ),
    ]

    rewritten_nodes = []
    inserted = False
    removed_nodes = 0
    for node in model.graph.node:
        if node.name in remove_names:
            removed_nodes += 1
            if not inserted:
                rewritten_nodes.extend(new_nodes)
                inserted = True
            continue
        rewritten_nodes.append(node)
    if removed_nodes != len(remove_names):
        raise ValueError(f"expected to remove {len(remove_names)} nodes, removed {removed_nodes}")
    if not inserted:
        raise ValueError("failed to insert gather rewrite nodes")

    del model.graph.node[:]
    model.graph.node.extend(rewritten_nodes)

    kept_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.name != "perm_flat"
    ]
    removed_initializers = len(model.graph.initializer) - len(kept_initializers)
    if removed_initializers != 1:
        raise ValueError(f"expected to remove perm_flat only, removed {removed_initializers}")
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    model.graph.initializer.append(numpy_helper.from_array(gather_indices, name="PermGatherIdx"))

    del model.graph.value_info[:]
    onnx.checker.check_model(model)
    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path))
    output_cost = estimate_model_cost(str(output_path))
    row = {
        "source_model_path": str(source_path),
        "output_model_path": str(output_path),
        "source_cost": int(source_cost["estimated_cost"]),
        "output_cost": int(output_cost["estimated_cost"]),
        "cost_delta": int(output_cost["estimated_cost"]) - int(source_cost["estimated_cost"]),
        "source_file_size_bytes": int(source_cost["file_size_bytes"]),
        "output_file_size_bytes": int(output_cost["file_size_bytes"]),
        "file_size_delta": int(output_cost["file_size_bytes"]) - int(source_cost["file_size_bytes"]),
        "removed_nodes": removed_nodes,
        "removed_initializers": removed_initializers,
        "added_initializers": 1,
        "failure_reason": "",
    }
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=GATHER_FIELDS)
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, ensure_ascii=False, indent=2))
    return row


def build_prune_candidates(
    source_model: str = DEFAULT_SOURCE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    report_path: str = DEFAULT_REPORT,
    modes: list[str] | None = None,
) -> dict[str, Any]:
    """Build task076 permutation-row prune candidates and write a CSV report."""
    selected_modes = modes or list(MODE_CHOICES)
    rows: list[dict[str, Any]] = []
    for mode in selected_modes:
        output_model = Path(output_dir) / f"task076_Task076PermPrune_{mode}.onnx"
        try:
            rows.append(prune_task076_perm_rows(source_model, str(output_model), mode))
        except Exception as exc:
            rows.append(
                {
                    "mode": mode,
                    "kept_indices": ",".join(str(value) for value in MODE_INDICES.get(mode, ())),
                    "source_model_path": source_model,
                    "output_model_path": str(output_model),
                    "source_cost": "",
                    "output_cost": "",
                    "cost_delta": "",
                    "source_file_size_bytes": "",
                    "output_file_size_bytes": "",
                    "file_size_delta": "",
                    "updated_initializers": "",
                    "failure_reason": str(exc),
                }
            )

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "source_model": source_model,
        "output_dir": output_dir,
        "report_path": str(report),
        "candidate_count": sum(1 for row in rows if not row["failure_reason"]),
        "rows": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_modes(raw: str) -> list[str] | None:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    return modes or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--modes", default="")
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--gather-exact", action="store_true")
    parser.add_argument("--output", default=DEFAULT_GATHER_OUTPUT)
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(inspect_task076_perm_tables(args.source), ensure_ascii=False, indent=2))
        return
    if args.gather_exact:
        build_task076_perm_gather_exact(
            source_model=args.source,
            output_model=args.output,
            report_path=args.report if args.report != DEFAULT_REPORT else DEFAULT_GATHER_REPORT,
        )
        return
    build_prune_candidates(
        source_model=args.source,
        output_dir=args.output_dir,
        report_path=args.report,
        modes=_parse_modes(args.modes),
    )


if __name__ == "__main__":
    main()
