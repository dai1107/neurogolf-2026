"""Direct int64->int32 shrink for Gather indices — no Cast nodes needed.

ONNX Gather accepts int32 or int64 indices natively. This pass changes oversized
int64 Gather-indices initializers to int32 directly, subtracting bytes without
adding any nodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

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
    "shrunk_initializers",
    "bytes_saved",
    "failure_reason",
]


def _is_gather_only_consumer(model: onnx.ModelProto, init_name: str) -> bool:
    for node in model.graph.node:
        if init_name in node.input:
            if node.op_type != "Gather":
                return False
    return True


def _shrink_int64_initializers(
    model: onnx.ModelProto,
    min_elements: int = 256,
    min_savings: int = 1000,
) -> tuple[int, int]:
    """Shrink int64 initializers to int32 where consumers are only Gather.

    Returns (count_shrunk, bytes_saved).
    """
    shrunk = 0
    saved = 0
    for idx, init in enumerate(model.graph.initializer):
        if init.data_type != TensorProto.INT64:
            continue
        dims = [int(d) for d in init.dims]
        elements = 1
        for d in dims:
            elements *= d
        if elements < min_elements:
            continue
        if not _is_gather_only_consumer(model, init.name):
            continue

        arr = numpy_helper.to_array(init)
        if arr.min() < -2147483648 or arr.max() > 2147483647:
            continue  # values don't fit in int32

        savings = arr.nbytes - (elements * 4)
        if savings < min_savings:
            continue

        # Replace with int32 version
        new_init = numpy_helper.from_array(
            arr.astype(np.int32, copy=False),
            name=init.name,
        )
        model.graph.initializer[idx].CopyFrom(new_init)
        shrunk += 1
        saved += savings

    return shrunk, saved


def shrink_model(
    input_model: str,
    output_model: str,
    min_elements: int = 256,
    min_savings: int = 1000,
) -> dict[str, Any]:
    input_path = Path(input_model)
    output_path = Path(output_model)
    if not input_path.is_file():
        raise FileNotFoundError(f"input model does not exist: {input_model}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = onnx.load(str(input_path))
    onnx.checker.check_model(model)
    source_cost = estimate_model_cost(str(input_path))

    shrunk, saved = _shrink_int64_initializers(model, min_elements, min_savings)

    if shrunk == 0:
        shutil.copyfile(input_path, output_path)
        return {
            "source_model_path": str(input_path),
            "output_model_path": str(output_path),
            "source_cost": int(source_cost["estimated_cost"]),
            "output_cost": int(source_cost["estimated_cost"]),
            "cost_delta": 0,
            "source_file_size_bytes": int(source_cost["file_size_bytes"]),
            "output_file_size_bytes": int(source_cost["file_size_bytes"]),
            "file_size_delta": 0,
            "shrunk_initializers": 0,
            "bytes_saved": 0,
            "failure_reason": "no shrinkable initializers found",
        }

    # Clear stale value_info
    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop()

    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))

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
        "shrunk_initializers": shrunk,
        "bytes_saved": saved,
        "failure_reason": "",
    }


def shrink_task_models(
    model_dir: str,
    output_dir: str,
    report_path: str,
    task_ids: list[str],
    min_elements: int = 256,
    min_savings: int = 1000,
) -> dict[str, Any]:
    model_root = Path(model_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for task_id in task_ids:
        source = model_root / f"{task_id}.onnx"
        dest = output_root / f"{task_id}_Int64ToInt32Shrink.onnx"
        row = shrink_model(str(source), str(dest), min_elements, min_savings)
        rows.append({"task_id": task_id, **row})

    with Path(report_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    improvements = [r for r in rows if int(r["cost_delta"]) < 0]
    summary = {
        "report_path": report_path,
        "improvement_count": len(improvements),
        "total_cost_delta": sum(int(r["cost_delta"]) for r in rows),
        "total_bytes_saved": sum(int(r.get("bytes_saved", 0)) for r in rows),
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
    parser.add_argument("--output-dir", default="outputs/candidates/int64_to_int32_shrink")
    parser.add_argument("--report", default="outputs/reports/int64_to_int32_shrink.csv")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--min-elements", type=int, default=256)
    parser.add_argument("--min-savings", type=int, default=1000)
    args = parser.parse_args()
    task_ids = _parse_task_ids(args.task_ids) if args.task_ids else _discover_task_ids(args.model_dir)
    if not task_ids:
        raise ValueError("--task-ids must contain at least one task id")
    shrink_task_models(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        report_path=args.report,
        task_ids=task_ids,
        min_elements=args.min_elements,
        min_savings=args.min_savings,
    )


if __name__ == "__main__":
    main()
