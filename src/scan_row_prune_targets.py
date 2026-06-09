"""Scan all models for safe row-pruning opportunities.

Observes which rows of large Gather-data tables are actually accessed during
evaluation. Reports candidates where significant row reduction is possible
without changing op types or adding nodes (only Sub for index adjustment).
"""

from __future__ import annotations

import argparse, copy, csv, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from .encoding import grid_to_onehot

ort.set_default_logger_severity(3)

FIELDS = [
    "task_id", "table_name", "table_shape", "table_dtype",
    "total_rows", "observed_min", "observed_max", "observed_unique",
    "table_nbytes", "index_consumer", "reduction_pct",
]


def observe_gather_rows(model_path: str, task_path: str) -> list[dict]:
    """Observe which rows of Gather-data tables are used."""
    model = onnx.load(model_path)
    task = json.loads(Path(task_path).read_text(encoding="utf-8"))

    # Find large tables consumed by Gather as data
    targets = {}
    for node in model.graph.node:
        if node.op_type != "Gather":
            continue
        data_name = node.input[0]
        idx_name = node.input[1] if len(node.input) > 1 else None
        if idx_name is None:
            continue

        for init in model.graph.initializer:
            if init.name == data_name:
                arr = numpy_helper.to_array(init)
                if arr.size >= 100 and arr.nbytes >= 4000:
                    if arr.ndim >= 1 and arr.shape[0] >= 10:
                        key = f"{data_name}@{idx_name}"
                        targets[key] = {
                            "table_name": data_name,
                            "idx_input": idx_name,
                            "shape": list(arr.shape), "dtype": str(arr.dtype),
                            "nbytes": arr.nbytes, "rows": arr.shape[0],
                        }
                break

    if not targets:
        return []

    # Add intermediate outputs (dedup by idx_input name)
    new_model = copy.deepcopy(model)
    added_outputs = set()
    for info in targets.values():
        idx_name = info["idx_input"]
        if idx_name in added_outputs:
            continue
        if any(o.name == idx_name for o in new_model.graph.output):
            added_outputs.add(idx_name)
            continue
        new_model.graph.output.append(
            onnx.helper.make_tensor_value_info(idx_name, onnx.TensorProto.INT64, [])
        )
        added_outputs.add(idx_name)

    temp_path = model_path + ".observe.tmp.onnx"
    onnx.save(new_model, temp_path)

    try:
        session = ort.InferenceSession(temp_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        Path(temp_path).unlink(missing_ok=True)
        return []

    output_names = [o.name for o in session.get_outputs()]

    # Run all labelled cases
    all_cases = []
    for split in ("train", "test", "arc-gen"):
        for i, case in enumerate(task.get(split, [])):
            all_cases.append((f"{split}[{i}]", case))

    observed = defaultdict(set)

    for case_id, case in all_cases:
        try:
            inp = grid_to_onehot(case["input"]).astype(np.float32)
            outputs = session.run(None, {"input": inp})
        except Exception:
            continue

        for j, name in enumerate(output_names):
            if name in added_outputs:
                val = outputs[j]
                if isinstance(val, np.ndarray):
                    observed[name].update(int(x) for x in val.flatten())

    Path(temp_path).unlink(missing_ok=True)

    # Build idx_name -> observed mapping
    results = []
    for key, info in targets.items():
        idx_name = info["idx_input"]
        obs = observed.get(idx_name, set())
        if not obs:
            continue

        obs_list = sorted(obs)
        obs_min, obs_max = obs_list[0], obs_list[-1]
        total_rows = info["rows"]
        used_range = obs_max - obs_min + 1
        reduction_pct = (1 - used_range / total_rows) * 100

        if reduction_pct < 10:
            continue

        results.append({
            "task_id": Path(task_path).stem,
            "table_name": info["table_name"],
            "table_shape": "x".join(str(d) for d in info["shape"]),
            "table_dtype": info["dtype"],
            "total_rows": total_rows,
            "observed_min": obs_min,
            "observed_max": obs_max,
            "observed_unique": len(obs),
            "table_nbytes": info["nbytes"],
            "index_consumer": idx_name,
            "reduction_pct": f"{reduction_pct:.1f}",
        })

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="outputs/onnx")
    parser.add_argument("--data-dir", default="task")
    parser.add_argument("--report", default="outputs/reports/row_prune_targets.csv")
    parser.add_argument("--task-ids", default="")
    parser.add_argument("--min-reduction", type=float, default=10.0)
    args = parser.parse_args()

    model_root = Path(args.model_dir)
    data_root = Path(args.data_dir)

    if args.task_ids:
        task_ids = [t.strip() for t in args.task_ids.split(",")]
    else:
        task_ids = sorted(p.stem for p in model_root.glob("task*.onnx"))

    all_results = []
    for tid in task_ids:
        model_path = model_root / f"{tid}.onnx"
        task_path = data_root / f"{tid}.json"
        if not model_path.is_file() or not task_path.is_file():
            continue
        try:
            results = observe_gather_rows(str(model_path), str(task_path))
            all_results.extend(results)
        except Exception as e:
            print(f"{tid}: ERROR {e}", file=sys.stderr)

    all_results.sort(key=lambda r: int(r["table_nbytes"]), reverse=True)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Found {len(all_results)} row-prune candidates across {len(set(r['task_id'] for r in all_results))} tasks")
    for r in all_results:
        print(f"  {r['task_id']}/{r['table_name']}: {r['total_rows']} rows -> "
              f"[{r['observed_min']},{r['observed_max']}] ({r['reduction_pct']}% reduction), "
              f"{r['table_nbytes']} bytes")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
