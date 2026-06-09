"""Observe and prefix-prune row_bank/col_bank enumeration tables.

Some generated ONNX graphs enumerate candidate top-left coordinates as
``row_bank_i`` and ``col_bank_i`` tables, then choose one coordinate with
``best_pos_i``.  Arbitrary row deletion would change index semantics, so this
module only supports prefix pruning: keep rows ``[0, prefix)`` for each bank.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from .arc_io import load_task
from .cost_estimator import estimate_model_cost
from .encoding import grid_to_onehot


ort.set_default_logger_severity(3)

BANK_RE = re.compile(r"^row_bank_(\d+)$")
OBSERVED_FIELDS = [
    "split",
    "case_index",
    "bank_id",
    "selected_index",
    "selected_row",
    "selected_col",
]
PRUNE_FIELDS = [
    "bank_id",
    "value_name",
    "old_shape",
    "new_shape",
    "old_num_elements",
    "new_num_elements",
    "prefix",
    "action",
]


def _shape_text(shape: Iterable[int]) -> str:
    values = tuple(int(dim) for dim in shape)
    return "x".join(str(dim) for dim in values) if values else "scalar"


def _initializer_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }


def _bank_ids(arrays: dict[str, np.ndarray]) -> list[int]:
    bank_ids = []
    for name in arrays:
        match = BANK_RE.match(name)
        if match and f"col_bank_{match.group(1)}" in arrays:
            bank_ids.append(int(match.group(1)))
    return sorted(bank_ids)


def _labelled_cases(task: dict[str, Any]) -> Iterable[tuple[str, int, dict[str, Any]]]:
    for split in ("train", "test", "arc-gen"):
        cases = task.get(split, [])
        if not isinstance(cases, list):
            continue
        for case_index, case in enumerate(cases):
            if isinstance(case, dict) and "input" in case and "output" in case:
                yield split, case_index, case


def _value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    inferred = onnx.shape_inference.infer_shapes(model)
    values: dict[str, onnx.ValueInfoProto] = {}
    for value_info in (
        list(inferred.graph.input)
        + list(inferred.graph.value_info)
        + list(inferred.graph.output)
    ):
        values[value_info.name] = value_info
    return values


def _add_intermediate_outputs(
    source_model: str,
    output_model: str,
    output_names: Iterable[str],
) -> None:
    model = onnx.load(source_model)
    existing_outputs = {output.name for output in model.graph.output}
    infos = _value_info_map(model)
    produced = {name for node in model.graph.node for name in node.output}
    for output_name in output_names:
        if output_name in existing_outputs:
            continue
        if output_name not in produced:
            raise ValueError(f"model does not produce requested output: {output_name}")
        info = infos.get(output_name)
        if info is None:
            info = onnx.helper.make_tensor_value_info(
                output_name,
                onnx.TensorProto.INT64,
                None,
            )
        model.graph.output.append(info)
    onnx.save(model, output_model)


def observe_row_bank_indices(
    model_path: str,
    task_path: str,
    report_path: str,
    summary_path: str,
) -> dict[str, Any]:
    """Record selected ``best_pos_i`` values on all labelled cases."""
    model = onnx.load(model_path)
    arrays = _initializer_arrays(model)
    bank_ids = _bank_ids(arrays)
    if not bank_ids:
        raise ValueError("model has no row_bank_i/col_bank_i pairs")

    produced = {name for node in model.graph.node for name in node.output}
    output_names = [f"best_pos_{bank_id}" for bank_id in bank_ids]
    missing = [name for name in output_names if name not in produced]
    if missing:
        raise ValueError(f"model is missing selected row outputs: {missing[:5]}")

    task = load_task(task_path)
    rows: list[dict[str, Any]] = []
    observed_by_bank: dict[int, set[int]] = {bank_id: set() for bank_id in bank_ids}

    with TemporaryDirectory() as tmpdir:
        instrumented = str(Path(tmpdir) / "row_bank_observe.onnx")
        _add_intermediate_outputs(model_path, instrumented, output_names)
        session = ort.InferenceSession(instrumented, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        session_outputs = [output.name for output in session.get_outputs()]

        for split, case_index, case in _labelled_cases(task):
            result = session.run(session_outputs, {input_name: grid_to_onehot(case["input"])})
            outputs = dict(zip(session_outputs, result))
            for bank_id in bank_ids:
                selected = int(np.asarray(outputs[f"best_pos_{bank_id}"]).reshape(-1)[0])
                row_bank = arrays[f"row_bank_{bank_id}"]
                col_bank = arrays[f"col_bank_{bank_id}"]
                if selected < 0 or selected >= int(row_bank.shape[0]):
                    raise ValueError(
                        f"selected index out of range for bank {bank_id}: {selected}"
                    )
                observed_by_bank[bank_id].add(selected)
                rows.append(
                    {
                        "split": split,
                        "case_index": case_index,
                        "bank_id": bank_id,
                        "selected_index": selected,
                        "selected_row": int(row_bank[selected]),
                        "selected_col": int(col_bank[selected]),
                    }
                )

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_path": model_path,
        "task_path": task_path,
        "report_path": str(report),
        "summary_path": summary_path,
        "labelled_cases": len({(row["split"], row["case_index"]) for row in rows}),
        "bank_count": len(bank_ids),
        "banks": {
            str(bank_id): {
                "row_count": int(arrays[f"row_bank_{bank_id}"].shape[0]),
                "observed_count": len(observed),
                "min_observed": min(observed) if observed else None,
                "max_observed": max(observed) if observed else None,
                "observed_prefix": (max(observed) + 1) if observed else None,
            }
            for bank_id, observed in observed_by_bank.items()
        },
    }
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _read_observed(path: str) -> dict[int, set[int]]:
    observed: dict[int, set[int]] = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bank_id = int(row["bank_id"])
            observed.setdefault(bank_id, set()).add(int(row["selected_index"]))
    if not observed:
        raise ValueError(f"observed row-bank report is empty: {path}")
    return observed


def _prefix_for_mode(row_count: int, observed: set[int], mode: str) -> int:
    if not observed:
        return row_count
    observed_prefix = max(observed) + 1
    if mode == "observed":
        return observed_prefix
    if mode == "medium":
        return min(row_count, max(observed_prefix + 4, int(np.ceil(row_count * 0.75))))
    if mode == "conservative":
        return min(row_count, max(observed_prefix + 8, int(np.ceil(row_count * 0.90))))
    raise ValueError(f"unknown row-bank prefix mode: {mode}")


def _replace_initializer(
    initializer: onnx.TensorProto,
    new_array: np.ndarray,
    bank_id: int,
    prefix: int,
    rows: list[dict[str, Any]],
) -> None:
    old_array = numpy_helper.to_array(initializer)
    initializer.CopyFrom(numpy_helper.from_array(new_array, name=initializer.name))
    rows.append(
        {
            "bank_id": bank_id,
            "value_name": initializer.name,
            "old_shape": _shape_text(old_array.shape),
            "new_shape": _shape_text(new_array.shape),
            "old_num_elements": int(old_array.size),
            "new_num_elements": int(new_array.size),
            "prefix": prefix,
            "action": "prefix_sliced",
        }
    )


def prune_row_bank_prefixes(
    source_model: str,
    output_model: str,
    report_path: str,
    observed_report: str,
    mode: str = "conservative",
) -> dict[str, Any]:
    """Prefix-slice row/col bank initializers without remapping indices."""
    if mode not in {"conservative", "medium", "observed"}:
        raise ValueError(f"unknown row-bank prefix prune mode: {mode}")
    model = onnx.load(source_model)
    arrays = _initializer_arrays(model)
    bank_ids = _bank_ids(arrays)
    observed_by_bank = _read_observed(observed_report)

    prefixes: dict[int, int] = {}
    for bank_id in bank_ids:
        row_count = int(arrays[f"row_bank_{bank_id}"].shape[0])
        observed = observed_by_bank.get(bank_id, set())
        prefix = _prefix_for_mode(row_count, observed, mode)
        if observed and max(observed) >= prefix:
            raise ValueError(f"{mode} prefix drops observed index in bank {bank_id}")
        prefixes[bank_id] = prefix

    rows: list[dict[str, Any]] = []
    for initializer in model.graph.initializer:
        for prefix_name in ("row_bank_", "col_bank_"):
            if not initializer.name.startswith(prefix_name):
                continue
            bank_id = int(initializer.name.removeprefix(prefix_name))
            if bank_id not in prefixes:
                continue
            array = numpy_helper.to_array(initializer)
            prefix = prefixes[bank_id]
            if prefix >= int(array.shape[0]):
                continue
            _replace_initializer(
                initializer,
                array[:prefix],
                bank_id,
                prefix,
                rows,
            )

    if not rows:
        raise ValueError(f"{mode} row-bank prefix pruning made no changes")

    del model.graph.value_info[:]
    onnx.checker.check_model(model)

    output = Path(output_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    onnx.checker.check_model(str(output))

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRUNE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    source_cost = estimate_model_cost(source_model)
    output_cost = estimate_model_cost(str(output))
    summary = {
        "source_model": source_model,
        "output_model": str(output),
        "report_path": str(report),
        "observed_report": observed_report,
        "mode": mode,
        "bank_count": len(bank_ids),
        "changed_initializers": len(rows),
        "prefixes": {str(bank_id): prefixes[bank_id] for bank_id in bank_ids},
        "source_estimated_cost": source_cost["estimated_cost"],
        "output_estimated_cost": output_cost["estimated_cost"],
        "source_file_size_bytes": source_cost["file_size_bytes"],
        "output_file_size_bytes": output_cost["file_size_bytes"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def generate_row_bank_prefix_candidates(
    task_id: str,
    source_model: str,
    task_path: str,
    candidate_dir: str,
    report_dir: str,
) -> dict[str, Any]:
    """Observe and generate Conservative/Medium/Observed prefix candidates."""
    report_root = Path(report_dir)
    observed_report = str(report_root / f"{task_id}_row_bank_selected_observed.csv")
    observed_summary = str(report_root / f"{task_id}_row_bank_selected_summary.json")
    observe_summary = observe_row_bank_indices(
        model_path=source_model,
        task_path=task_path,
        report_path=observed_report,
        summary_path=observed_summary,
    )

    mode_names = {
        "conservative": "RowBankPrefixConservative",
        "medium": "RowBankPrefixMedium",
        "observed": "RowBankPrefixObserved",
    }
    candidates: list[dict[str, Any]] = []
    for mode, name in mode_names.items():
        candidates.append(
            prune_row_bank_prefixes(
                source_model=source_model,
                output_model=str(Path(candidate_dir) / f"{task_id}_{name}.onnx"),
                report_path=str(report_root / f"{task_id}_row_bank_prefix_{mode}.csv"),
                observed_report=observed_report,
                mode=mode,
            )
        )
    summary = {
        "task_id": task_id,
        "source_model": source_model,
        "task_path": task_path,
        "candidate_dir": candidate_dir,
        "report_dir": report_dir,
        "observe_summary": observe_summary,
        "candidates": candidates,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    observe = subparsers.add_parser("observe")
    observe.add_argument("--model", required=True)
    observe.add_argument("--task", required=True)
    observe.add_argument("--report", required=True)
    observe.add_argument("--summary", required=True)

    prune = subparsers.add_parser("prune")
    prune.add_argument("--source", required=True)
    prune.add_argument("--output", required=True)
    prune.add_argument("--report", required=True)
    prune.add_argument("--observed-report", required=True)
    prune.add_argument("--mode", choices=("conservative", "medium", "observed"), required=True)

    generate = subparsers.add_parser("generate-candidates")
    generate.add_argument("--task-id", required=True)
    generate.add_argument("--source", required=True)
    generate.add_argument("--task", required=True)
    generate.add_argument("--candidate-dir", required=True)
    generate.add_argument("--report-dir", required=True)

    args = parser.parse_args()
    if args.command == "observe":
        observe_row_bank_indices(
            model_path=args.model,
            task_path=args.task,
            report_path=args.report,
            summary_path=args.summary,
        )
    elif args.command == "prune":
        prune_row_bank_prefixes(
            source_model=args.source,
            output_model=args.output,
            report_path=args.report,
            observed_report=args.observed_report,
            mode=args.mode,
        )
    elif args.command == "generate-candidates":
        generate_row_bank_prefix_candidates(
            task_id=args.task_id,
            source_model=args.source,
            task_path=args.task,
            candidate_dir=args.candidate_dir,
            report_dir=args.report_dir,
        )


if __name__ == "__main__":
    main()
