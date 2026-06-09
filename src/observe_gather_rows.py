"""Observe which rows of Gather-data tables are used during evaluation.

For tables consumed as data by Gather nodes (input[0]), run the model on all
labelled cases and record which row indices are accessed.  Used to build safe
row-prune candidates without changing dtype or adding Cast nodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from .encoding import grid_to_onehot

ort.set_default_logger_severity(3)

FIELDS = [
    "task_id",
    "table_name",
    "shape",
    "dtype",
    "total_rows",
    "observed_rows",
    "observed_indices",
    "index_source",
    "consumer_node",
    "notes",
]


def _add_intermediate_outputs(
    model: onnx.ModelProto,
    table_names: set[str],
) -> onnx.ModelProto:
    """Clone the model with extra outputs for the indices that select rows."""
    import copy
    new_model = copy.deepcopy(model)

    # Find Gather nodes that consume these tables as data
    for node in new_model.graph.node:
        if node.op_type != "Gather":
            continue
        if node.input[0] not in table_names:
            continue

        # The index input (input[1]) tells us which rows are selected
        index_name = node.input[1] if len(node.input) > 1 else None
        if index_name is None:
            continue

        # Check if this output is already in graph outputs
        already_output = any(o.name == index_name for o in new_model.graph.output)
        if already_output:
            continue

        # Add as intermediate output
        intermediate = onnx.helper.make_tensor_value_info(
            index_name,
            onnx.TensorProto.INT64,  # assume int64, will be handled by ORT
            [],
        )
        # Don't set shape - ORT will infer
        new_model.graph.output.append(intermediate)

    return new_model


def observe_table_rows(
    model_path: str,
    task_path: str,
    table_names: list[str],
) -> dict[str, Any]:
    """Return observed row indices for each table across all labelled cases."""
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))

    model = onnx.load(model_path)
    name_set = set(table_names)

    # Collect all cases
    cases = []
    for split_name in ("train", "test", "arc-gen"):
        for i, case in enumerate(task.get(split_name, [])):
            cases.append((f"{split_name}[{i}]", case))

    if not cases:
        return {"error": "no cases found"}

    # Find which Gather nodes consume these tables
    table_consumers = {}
    for node in model.graph.node:
        if node.op_type == "Gather" and node.input[0] in name_set:
            table_consumers[node.input[0]] = {
                "node_name": node.name,
                "index_input": node.input[1] if len(node.input) > 1 else None,
                "node_output": node.output[0] if node.output else None,
            }

    # Add intermediate outputs
    instrumented = _add_intermediate_outputs(model, name_set)

    # Save temp model
    temp_path = Path(model_path).parent / "_temp_observe.onnx"
    onnx.save(instrumented, str(temp_path))

    try:
        session = ort.InferenceSession(str(temp_path), providers=["CPUExecutionProvider"])
    except Exception as e:
        temp_path.unlink(missing_ok=True)
        return {"error": f"session creation failed: {e}"}

    # Run all cases
    observed = {name: set() for name in table_names}
    output_names = [o.name for o in session.get_outputs()]

    for case_id, case in cases:
        try:
            inp_grid = np.array(case["input"], dtype=np.int64)
            inp = grid_to_onehot(inp_grid).astype(np.float32)
            outputs = session.run(None, {"input": inp})
        except Exception:
            continue

        for i, name in enumerate(output_names):
            if name in table_consumers:
                table_name = None
                for tn, info in table_consumers.items():
                    if info["index_input"] == name:
                        table_name = tn
                        break
                if table_name is None:
                    continue

                indices = outputs[i]
                if isinstance(indices, np.ndarray):
                    observed[table_name].update(indices.flatten().tolist())

    temp_path.unlink(missing_ok=True)

    result = {}
    for name in table_names:
        rows_set = observed.get(name, set())
        result[name] = {
            "total_rows": "unknown",
            "observed_count": len(rows_set),
            "observed_min": min(rows_set) if rows_set else None,
            "observed_max": max(rows_set) if rows_set else None,
            "observed_set": sorted(rows_set) if len(rows_set) <= 50 else f"{len(rows_set)} values",
        }

    # Get actual row counts
    for init in model.graph.initializer:
        if init.name in table_names:
            arr = onnx.numpy_helper.to_array(init)
            result[init.name]["total_rows"] = arr.shape[0] if arr.ndim >= 1 else 1
            result[init.name]["shape"] = list(arr.shape)
            result[init.name]["dtype"] = str(arr.dtype)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--table-names", required=True, help="comma-separated initializer names")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    table_names = [n.strip() for n in args.table_names.split(",")]
    result = observe_table_rows(args.model, args.task, table_names)

    if args.report:
        rows = []
        for name, info in result.items():
            if "error" in info:
                continue
            rows.append({
                "task_id": Path(args.task).stem,
                "table_name": name,
                "shape": "x".join(str(d) for d in info.get("shape", [])),
                "dtype": info.get("dtype", "?"),
                "total_rows": info.get("total_rows", "?"),
                "observed_rows": info.get("observed_count", "?"),
                "observed_indices": str(info.get("observed_set", "")),
                "index_source": "dynamic",
                "consumer_node": "",
                "notes": f"range=[{info.get('observed_min')},{info.get('observed_max')}]",
            })
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
