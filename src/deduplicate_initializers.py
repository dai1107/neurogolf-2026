"""Deduplicate identical ONNX initializers without changing graph semantics."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import onnx

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
    "removed_unused_initializers",
    "deduplicated_initializers",
]


def _initializer_key(initializer: onnx.TensorProto) -> bytes:
    clone = onnx.TensorProto()
    clone.CopyFrom(initializer)
    clone.name = ""
    return clone.SerializeToString(deterministic=True)


def _referenced_initializer_names(graph: onnx.GraphProto) -> set[str]:
    """Return top-level initializer names that must be kept."""
    referenced = {value.name for value in graph.input}
    referenced.update(value.name for value in graph.output)
    for node in graph.node:
        referenced.update(input_name for input_name in node.input if input_name)
    return referenced


def deduplicate_initializers(input_model: str, output_model: str) -> dict[str, Any]:
    """Write a graph-equivalent model with unused and duplicate initializers removed."""
    input_path = Path(input_model)
    output_path = Path(output_model)
    if not input_path.is_file():
        raise FileNotFoundError(f"input model does not exist: {input_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    onnx.checker.check_model(model)
    source_cost = estimate_model_cost(str(input_path))

    referenced = _referenced_initializer_names(model.graph)
    active_initializers = [
        initializer for initializer in model.graph.initializer if initializer.name in referenced
    ]
    removed_unused = len(model.graph.initializer) - len(active_initializers)

    seen: dict[bytes, str] = {}
    rename: dict[str, str] = {}
    kept: list[onnx.TensorProto] = []
    source_initializer_count = len(model.graph.initializer)
    for initializer in active_initializers:
        key = _initializer_key(initializer)
        canonical_name = seen.get(key)
        if canonical_name is None:
            seen[key] = initializer.name
            kept.append(initializer)
        else:
            rename[initializer.name] = canonical_name

    if rename or removed_unused:
        for node in model.graph.node:
            for index, input_name in enumerate(node.input):
                if input_name in rename:
                    node.input[index] = rename[input_name]
        del model.graph.initializer[:]
        model.graph.initializer.extend(kept)
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
        "output_initializer_count": len(kept),
        "removed_unused_initializers": removed_unused,
        "deduplicated_initializers": len(rename),
    }


def deduplicate_task_models(
    model_dir: str,
    output_dir: str,
    report_path: str,
    task_ids: list[str],
) -> dict[str, Any]:
    """Deduplicate selected task models and write a CSV report."""
    model_root = Path(model_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        source = model_root / f"{task_id}.onnx"
        destination = output_root / f"{task_id}_DeduplicateInitializers.onnx"
        row = deduplicate_initializers(str(source), str(destination))
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
    parser.add_argument("--output-dir", default="outputs/candidates/deduplicated")
    parser.add_argument("--report", default="outputs/reports/deduplicate_initializers_report.csv")
    parser.add_argument("--task-ids", default="", help="comma-separated task ids; defaults to all task*.onnx")
    args = parser.parse_args()
    task_ids = _parse_task_ids(args.task_ids) if args.task_ids else _discover_task_ids(args.model_dir)
    if not task_ids:
        raise ValueError("--task-ids must contain at least one task id")
    deduplicate_task_models(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        task_ids=task_ids,
    )


if __name__ == "__main__":
    main()
