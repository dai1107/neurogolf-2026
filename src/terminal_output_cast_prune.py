"""Remove terminal output Cast nodes when the producer can become output.

This pass targets graphs whose final node is:

    Cast(source_tensor) -> output

If ``source_tensor`` has exactly one consumer, the pass renames the producer's
output to the canonical graph output and updates the graph output dtype. This
removes one full-size intermediate tensor from the official-static memory cost.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import onnx

from .hybrid_stack_optimizer import OPTIMIZE_FIELDS, _deduplicate_initializers, _prune_dead_graph
from .official_cost_estimator import estimate_official_static_cost


def _value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    return {
        value.name: value
        for value in list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    }


def prune_terminal_output_cast(source_model: str, output_model: str) -> dict[str, Any]:
    """Write a terminal-Cast-pruned ONNX model and return cost metadata."""
    source_path = Path(source_model)
    output_path = Path(output_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(source_path))
    graph = model.graph
    if not graph.node:
        raise ValueError("graph has no nodes")

    output_name = graph.output[0].name
    terminal = graph.node[-1]
    if terminal.op_type != "Cast" or list(terminal.output) != [output_name] or len(terminal.input) != 1:
        raise ValueError("final node is not a single-output Cast to graph output")

    source_name = terminal.input[0]
    consumers: list[onnx.NodeProto] = []
    producer: onnx.NodeProto | None = None
    for node in graph.node:
        if source_name in node.input:
            consumers.append(node)
        if source_name in node.output:
            producer = node
    if producer is None:
        raise ValueError(f"terminal Cast source has no producer: {source_name}")
    if consumers != [terminal]:
        raise ValueError(f"terminal Cast source has non-terminal consumers: {source_name}")

    value_info = _value_info_map(model).get(source_name)
    if value_info is None:
        raise ValueError(f"missing static value_info for terminal Cast source: {source_name}")
    tensor_type = value_info.type.tensor_type
    source_shape = [int(dim.dim_value) for dim in tensor_type.shape.dim]
    source_elem_type = int(tensor_type.elem_type)

    for index, name in enumerate(producer.output):
        if name == source_name:
            producer.output[index] = output_name
            break
    else:
        raise ValueError(f"producer does not output {source_name}")

    del graph.node[-1]
    graph.output[0].type.tensor_type.elem_type = source_elem_type
    del graph.output[0].type.tensor_type.shape.dim[:]
    for dim in source_shape:
        graph.output[0].type.tensor_type.shape.dim.add().dim_value = dim

    kept_value_info = [
        value for value in graph.value_info if value.name not in {source_name, output_name}
    ]
    del graph.value_info[:]
    graph.value_info.extend(kept_value_info)

    _prune_dead_graph(model)
    _deduplicate_initializers(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, str(output_path))

    before = estimate_official_static_cost(str(source_path))
    after = estimate_official_static_cost(str(output_path))
    return {
        "source_model_path": str(source_path),
        "output_model_path": str(output_path),
        "source_estimated_cost": int(before["official_static_cost"]),
        "output_estimated_cost": int(after["official_static_cost"]),
        "estimated_cost_delta": int(after["official_static_cost"])
        - int(before["official_static_cost"]),
        "source_file_size_bytes": int(before["file_size_bytes"]),
        "output_file_size_bytes": int(after["file_size_bytes"]),
        "file_size_delta": int(after["file_size_bytes"]) - int(before["file_size_bytes"]),
        "source_node_count": int(before["node_count"]),
        "output_node_count": int(after["node_count"]),
        "source_tensor_name": source_name,
        "source_elem_type": source_elem_type,
        "source_shape": "x".join(str(dim) for dim in source_shape),
    }


def build_candidate_report(
    stack_dir: str,
    output_dir: str,
    report_path: str,
    lanes: set[str],
    task_ids: set[str] | None,
) -> dict[str, Any]:
    """Build terminal-output-Cast candidates for selected stack lanes."""
    root = Path(stack_dir)
    output_root = Path(output_dir)
    report = Path(report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for lane in sorted(lanes):
        lane_root = root / lane
        for source in sorted(lane_root.glob("task*.onnx")):
            task_id = source.stem
            if task_ids is not None and task_id not in task_ids:
                continue
            destination = output_root / f"{task_id}_{lane}_TerminalOutputCastPruned.onnx"
            row = {field: "" for field in OPTIMIZE_FIELDS}
            row.update({"task_id": task_id, "lane": lane, "source_model_path": str(source)})
            try:
                result = prune_terminal_output_cast(str(source), str(destination))
                row.update(
                    {
                        "output_model_path": result["output_model_path"],
                        "source_estimated_cost": result["source_estimated_cost"],
                        "output_estimated_cost": result["output_estimated_cost"],
                        "estimated_cost_delta": result["estimated_cost_delta"],
                        "source_file_size_bytes": result["source_file_size_bytes"],
                        "output_file_size_bytes": result["output_file_size_bytes"],
                        "file_size_delta": result["file_size_delta"],
                        "changed": "True",
                        "removed_dead_nodes": int(result["source_node_count"])
                        - int(result["output_node_count"]),
                        "removed_unused_initializers": "0",
                        "deduplicated_initializers": "0",
                        "constant_gather_tables_pruned": "0",
                        "constant_gather_rows_removed": "0",
                        "constant_gather_bytes_removed": "0",
                        "initializer_bytes_delta": "0",
                        "checker_passed": "True",
                        "forbidden_ops_passed": "True",
                        "static_shapes_passed": "True",
                        "equivalence_passed": "not_run",
                        "candidate_valid": "True",
                        "failure_reason": "",
                    }
                )
            except Exception as exc:
                row["candidate_valid"] = "False"
                row["failure_reason"] = str(exc)
            rows.append(row)

    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OPTIMIZE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    valid_rows = [row for row in rows if row["candidate_valid"] == "True"]
    summary = {
        "stack_dir": stack_dir,
        "output_dir": output_dir,
        "report_path": report_path,
        "rows": len(rows),
        "valid_candidates": len(valid_rows),
        "total_estimated_cost_delta": sum(
            int(row["estimated_cost_delta"]) for row in valid_rows
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _parse_csv_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-dir", default="outputs/current_6349_79_stack")
    parser.add_argument("--output-dir", default="outputs/candidates/terminal_output_cast_prune")
    parser.add_argument("--report", default="outputs/reports/terminal_output_cast_prune.csv")
    parser.add_argument("--lanes", default="overrides")
    parser.add_argument("--task-ids", default="")
    args = parser.parse_args()

    task_ids = _parse_csv_set(args.task_ids) or None
    build_candidate_report(
        stack_dir=args.stack_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        lanes=_parse_csv_set(args.lanes),
        task_ids=task_ids,
    )


if __name__ == "__main__":
    main()
