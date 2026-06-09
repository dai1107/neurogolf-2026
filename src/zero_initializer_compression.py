"""Replace large all-zero ONNX initializers with ConstantOfShape nodes."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .cost_estimator import estimate_model_cost


FIELDS = [
    "task_id",
    "source_model_path",
    "output_model_path",
    "source_cost",
    "output_cost",
    "cost_delta",
    "source_file_size_bytes",
    "output_file_size_bytes",
    "file_size_delta",
    "source_initializer_count",
    "output_initializer_count",
    "zero_initializers_replaced",
    "zero_initializer_elements_replaced",
]


def _referenced_value_names(graph: onnx.GraphProto) -> set[str]:
    names = {value.name for value in graph.input}
    names.update(value.name for value in graph.output)
    for node in graph.node:
        names.update(name for name in node.input if name)
        names.update(name for name in node.output if name)
    for initializer in graph.initializer:
        names.add(initializer.name)
    return names


def _num_elements(dims: list[int]) -> int:
    result = 1
    for dim in dims:
        result *= int(dim)
    return result


def _is_all_zero_initializer(initializer: onnx.TensorProto) -> bool:
    if not initializer.dims:
        return False
    if initializer.data_type == TensorProto.STRING:
        return False
    array = numpy_helper.to_array(initializer)
    return bool(array.size and np.all(array == 0))


def _unique_name(base: str, used: set[str]) -> str:
    name = base
    index = 0
    while name in used:
        index += 1
        name = f"{base}_{index}"
    used.add(name)
    return name


def compress_zero_initializers(
    input_model: str,
    output_model: str,
    min_elements: int = 16,
) -> dict[str, Any]:
    """Write a graph-equivalent model with large all-zero constants materialized by shape."""
    input_path = Path(input_model)
    output_path = Path(output_model)
    if not input_path.is_file():
        raise FileNotFoundError(f"input model does not exist: {input_model}")
    if min_elements <= 0:
        raise ValueError("min_elements must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    onnx.checker.check_model(model)
    source_cost = estimate_model_cost(str(input_path))

    protected = {value.name for value in model.graph.input}
    protected.update(value.name for value in model.graph.output)
    used_names = _referenced_value_names(model.graph)

    kept_initializers: list[onnx.TensorProto] = []
    shape_initializers: list[onnx.TensorProto] = []
    constant_nodes: list[onnx.NodeProto] = []
    replaced_count = 0
    replaced_elements = 0

    for initializer in model.graph.initializer:
        element_count = _num_elements(list(initializer.dims))
        should_replace = (
            initializer.name not in protected
            and element_count >= min_elements
            and _is_all_zero_initializer(initializer)
        )
        if not should_replace:
            kept_initializers.append(initializer)
            continue

        shape_name = _unique_name(f"{initializer.name}_shape", used_names)
        shape_value = np.asarray(initializer.dims, dtype=np.int64)
        shape_initializers.append(numpy_helper.from_array(shape_value, name=shape_name))
        zero_value = helper.make_tensor(
            name=f"{initializer.name}_zero",
            data_type=initializer.data_type,
            dims=[1],
            vals=[0],
        )
        constant_nodes.append(
            helper.make_node(
                "ConstantOfShape",
                [shape_name],
                [initializer.name],
                name=_unique_name(f"{initializer.name}_constant_of_shape", used_names),
                value=zero_value,
            )
        )
        replaced_count += 1
        replaced_elements += element_count

    source_initializer_count = len(model.graph.initializer)
    if replaced_count:
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept_initializers)
        model.graph.initializer.extend(shape_initializers)
        original_nodes = list(model.graph.node)
        del model.graph.node[:]
        model.graph.node.extend(constant_nodes)
        model.graph.node.extend(original_nodes)
        onnx.checker.check_model(model)
        onnx.save(model, str(output_path))
        onnx.checker.check_model(str(output_path))
    else:
        shutil.copyfile(input_path, output_path)

    output_cost = estimate_model_cost(str(output_path))
    return {
        "source_model_path": str(input_path),
        "output_model_path": str(output_path),
        "source_cost": int(source_cost["estimated_cost"]),
        "output_cost": int(output_cost["estimated_cost"]),
        "cost_delta": int(output_cost["estimated_cost"]) - int(source_cost["estimated_cost"]),
        "source_file_size_bytes": int(source_cost["file_size_bytes"]),
        "output_file_size_bytes": int(output_cost["file_size_bytes"]),
        "file_size_delta": int(output_cost["file_size_bytes"]) - int(source_cost["file_size_bytes"]),
        "source_initializer_count": source_initializer_count,
        "output_initializer_count": len(model.graph.initializer),
        "zero_initializers_replaced": replaced_count,
        "zero_initializer_elements_replaced": replaced_elements,
    }


def compress_task_models(
    model_dir: str,
    output_dir: str,
    report_path: str,
    task_ids: list[str],
    min_elements: int = 16,
) -> dict[str, Any]:
    """Compress selected task models and write a CSV report."""
    model_root = Path(model_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        source = model_root / f"{task_id}.onnx"
        destination = output_root / f"{task_id}_ZeroInitializerCompression.onnx"
        row = compress_zero_initializers(str(source), str(destination), min_elements=min_elements)
        rows.append({"task_id": task_id, **row})

    with Path(report_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    improvements = [row for row in rows if int(row["cost_delta"]) < 0]
    summary = {
        "task_ids": task_ids,
        "report_path": report_path,
        "output_dir": output_dir,
        "improvement_count": len(improvements),
        "total_cost_delta": sum(int(row["cost_delta"]) for row in rows),
        "total_file_size_delta": sum(int(row["file_size_delta"]) for row in rows),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_task_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _discover_task_ids(model_dir: str) -> list[str]:
    return sorted(path.stem for path in Path(model_dir).glob("task*.onnx"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/onnx")
    parser.add_argument("--output-dir", default="outputs/candidates/zero_initializer_compressed")
    parser.add_argument("--report", default="outputs/reports/zero_initializer_compression_report.csv")
    parser.add_argument("--task-ids", default="", help="comma-separated task ids; defaults to all task*.onnx")
    parser.add_argument("--min-elements", type=int, default=16)
    args = parser.parse_args()
    task_ids = _parse_task_ids(args.task_ids) if args.task_ids else _discover_task_ids(args.model_dir)
    if not task_ids:
        raise ValueError("--task-ids must contain at least one task id")
    compress_task_models(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        task_ids=task_ids,
        min_elements=args.min_elements,
    )


if __name__ == "__main__":
    main()
