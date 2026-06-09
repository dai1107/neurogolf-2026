"""Observe and prune task157 placement-table rows.

The current task157 model uses a large placement enumeration table:
``plac_idx_963`` with shape ``1305 x 150`` and companion ``expand_idx_983``.
This module keeps the graph logic intact while slicing those rows and updating
row-count constants.  It is intentionally conservative: generated candidates
must still pass strict local validation before they are packaged for online
ablation.
"""

from __future__ import annotations

import argparse
import csv
import json
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

DEFAULT_SOURCE = "outputs/onnx/task157.onnx"
DEFAULT_TASK = "task/task157.json"
DEFAULT_OBSERVED_REPORT = "outputs/reports/task157_selected_placement_observed.csv"
DEFAULT_SUMMARY = "outputs/reports/task157_placement_prune_summary.json"
DEFAULT_CANDIDATE_DIR = "outputs/candidates/task157_placement_prune"
DEFAULT_REPORT = "outputs/reports/task157_placement_prune.csv"

ROW_COUNT = 1305
PLACEMENTS_PER_COMPONENT = 261
PLACEMENT_TABLES = ("plac_idx_963", "expand_idx_983")
ARGMAX_OUTPUTS = (
    "argmax_1039",
    "argmax_1098",
    "argmax_1157",
    "argmax_1216",
    "argmax_1275",
)
SELECTED_COMPONENT_OUTPUTS = (
    "gather_1051",
    "gather_1110",
    "gather_1169",
    "gather_1228",
    "gather_1287",
)
OBSERVE_OUTPUTS = ARGMAX_OUTPUTS + SELECTED_COMPONENT_OUTPUTS

OBSERVED_FIELDS = [
    "split",
    "case_index",
    "step",
    "prefix_size",
    "target_slot",
    "argmax_output",
    "selected_rows",
    "selected_components",
    "source_components",
    "source_component_sizes",
]

PRUNE_FIELDS = [
    "value_name",
    "value_kind",
    "old_shape",
    "new_shape",
    "old_num_elements",
    "new_num_elements",
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


def _component_sizes(expand_idx: np.ndarray) -> dict[int, int]:
    values, counts = np.unique(expand_idx.astype(np.int64), return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def _component_local_offsets(expand_idx: np.ndarray) -> np.ndarray:
    """Return each placement row's ordinal inside its source-component block."""
    offsets = np.zeros((int(expand_idx.shape[0]),), dtype=np.int64)
    seen: dict[int, int] = {}
    for row_index, component in enumerate(expand_idx.astype(np.int64).tolist()):
        component_id = int(component)
        offsets[row_index] = seen.get(component_id, 0)
        seen[component_id] = int(offsets[row_index]) + 1
    return offsets


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
            # This is only for a temporary instrumentation model.  ORT accepts
            # typed outputs without static shapes here, and final candidates are
            # never built from this temporary graph.
            info = onnx.helper.make_tensor_value_info(
                output_name,
                onnx.TensorProto.INT64,
                None,
            )
        model.graph.output.append(info)
    onnx.save(model, output_model)


def observe_task157_selected_rows(
    model_path: str = DEFAULT_SOURCE,
    task_path: str = DEFAULT_TASK,
    report_path: str = DEFAULT_OBSERVED_REPORT,
    summary_path: str = DEFAULT_SUMMARY,
) -> dict[str, Any]:
    """Record selected placement rows for every labelled task157 case."""
    task = load_task(task_path)
    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    observed_by_step: dict[str, set[int]] = {name: set() for name in ARGMAX_OUTPUTS}
    component_by_step: dict[str, set[int]] = {name: set() for name in SELECTED_COMPONENT_OUTPUTS}
    source_model = onnx.load(model_path)
    arrays = _initializer_arrays(source_model)
    if "expand_idx_983" not in arrays:
        raise ValueError("model is missing expand_idx_983")
    expand_idx = arrays["expand_idx_983"].astype(np.int64)
    component_sizes = _component_sizes(expand_idx)

    with TemporaryDirectory() as tmpdir:
        instrumented = str(Path(tmpdir) / "task157_observe.onnx")
        _add_intermediate_outputs(model_path, instrumented, OBSERVE_OUTPUTS)
        session = ort.InferenceSession(instrumented, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]

        for split, case_index, case in _labelled_cases(task):
            result = session.run(output_names, {input_name: grid_to_onehot(case["input"])})
            outputs = dict(zip(output_names, result))
            for step, (argmax_name, component_name) in enumerate(
                zip(ARGMAX_OUTPUTS, SELECTED_COMPONENT_OUTPUTS)
            ):
                selected = sorted(
                    {int(value) for value in np.asarray(outputs[argmax_name]).reshape(-1)}
                )
                components = sorted(
                    {int(value) for value in np.asarray(outputs[component_name]).reshape(-1)}
                )
                source_components = sorted(
                    {int(expand_idx[value]) for value in selected}
                )
                source_component_sizes = sorted(
                    {int(component_sizes[component]) for component in source_components}
                )
                observed_by_step[argmax_name].update(selected)
                component_by_step[component_name].update(components)
                rows.append(
                    {
                        "split": split,
                        "case_index": case_index,
                        "step": step,
                        "prefix_size": step,
                        "target_slot": step,
                        "argmax_output": argmax_name,
                        "selected_rows": json.dumps(selected, sort_keys=True),
                        "selected_components": json.dumps(components, sort_keys=True),
                        "source_components": json.dumps(source_components, sort_keys=True),
                        "source_component_sizes": json.dumps(
                            source_component_sizes,
                            sort_keys=True,
                        ),
                    }
                )

    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    all_rows = set().union(*observed_by_step.values()) if observed_by_step else set()
    all_components = (
        set().union(*component_by_step.values()) if component_by_step else set()
    )
    summary = {
        "model_path": model_path,
        "task_path": task_path,
        "report_path": str(report),
        "summary_path": str(summary_file),
        "labelled_cases": len({(row["split"], row["case_index"]) for row in rows}),
        "observed_rows": len(all_rows),
        "observed_components": len(all_components),
        "min_observed_row": min(all_rows) if all_rows else None,
        "max_observed_row": max(all_rows) if all_rows else None,
        "observed_rows_by_step": {
            name: sorted(values) for name, values in observed_by_step.items()
        },
        "observed_components_by_step": {
            name: sorted(values) for name, values in component_by_step.items()
        },
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _read_observed_rows(path: str) -> set[int]:
    observed_path = Path(path)
    if not observed_path.is_file():
        raise FileNotFoundError(f"observed placement report does not exist: {path}")
    observed: set[int] = set()
    with observed_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            selected = json.loads(row["selected_rows"])
            observed.update(int(value) for value in selected)
    if not observed:
        raise ValueError(f"observed placement report has no selected rows: {path}")
    return observed


def _read_int_list(path: str) -> set[int]:
    values_path = Path(path)
    if not values_path.is_file():
        raise FileNotFoundError(f"row list does not exist: {path}")
    raw = values_path.read_text(encoding="utf-8").replace("\n", ",")
    values = {int(item.strip()) for item in raw.split(",") if item.strip()}
    if not values:
        raise ValueError(f"row list is empty: {path}")
    return values


def _keep_indices_for_mode(
    mode: str,
    arrays: dict[str, np.ndarray],
    observed_rows: set[int],
    row_list: str | None,
) -> np.ndarray:
    expand_idx = arrays["expand_idx_983"]
    if mode == "observed":
        keep = set(observed_rows)
    elif mode in {"conservative", "component"}:
        observed_components = {int(expand_idx[row]) for row in observed_rows}
        keep = {
            row
            for row, component in enumerate(expand_idx.tolist())
            if int(component) in observed_components
        }
    elif mode == "medium":
        observed_components = {int(expand_idx[row]) for row in observed_rows}
        local_offsets = _component_local_offsets(expand_idx)
        observed_offsets = {int(local_offsets[row]) for row in observed_rows}
        keep = {
            row
            for row, component in enumerate(expand_idx.tolist())
            if int(component) in observed_components
            and int(local_offsets[row]) in observed_offsets
        }
    elif mode == "drop-list":
        if row_list is None:
            raise ValueError("drop-list mode requires --row-list")
        keep = set(range(int(expand_idx.shape[0]))) - _read_int_list(row_list)
    elif mode == "keep-list":
        if row_list is None:
            raise ValueError("keep-list mode requires --row-list")
        keep = _read_int_list(row_list)
    else:
        raise ValueError(f"unknown task157 placement prune mode: {mode}")

    missing_observed = observed_rows - keep
    if missing_observed:
        preview = sorted(missing_observed)[:10]
        raise ValueError(f"{mode} pruning drops observed placement rows: {preview}")
    if not keep:
        raise ValueError(f"{mode} pruning would keep zero rows")
    bad = [row for row in keep if row < 0 or row >= int(expand_idx.shape[0])]
    if bad:
        raise ValueError(f"{mode} pruning has out-of-range rows: {bad[:10]}")
    return np.asarray(sorted(keep), dtype=np.int64)


def _replace_initializer(
    initializer: onnx.TensorProto,
    new_array: np.ndarray,
    action: str,
    rows: list[dict[str, Any]],
) -> None:
    old_array = numpy_helper.to_array(initializer)
    initializer.CopyFrom(numpy_helper.from_array(new_array, name=initializer.name))
    rows.append(
        {
            "value_name": initializer.name,
            "value_kind": "initializer",
            "old_shape": _shape_text(old_array.shape),
            "new_shape": _shape_text(new_array.shape),
            "old_num_elements": int(old_array.size),
            "new_num_elements": int(new_array.size),
            "action": action,
        }
    )


def _replace_row_count_values(
    array: np.ndarray,
    old_row_count: int,
    new_row_count: int,
) -> np.ndarray | None:
    if not np.issubdtype(array.dtype, np.integer):
        return None
    mask = array == old_row_count
    if not np.any(mask):
        return None
    new_array = np.array(array, copy=True)
    new_array[mask] = np.asarray(new_row_count, dtype=array.dtype)
    return new_array


def _slice_row_count_axes(
    array: np.ndarray,
    keep_indices: np.ndarray,
    row_count: int = ROW_COUNT,
) -> np.ndarray | None:
    """Slice every axis whose length is the placement row count."""
    axes = [axis for axis, dim in enumerate(array.shape) if int(dim) == row_count]
    if not axes:
        return None
    new_array = array
    for axis in axes:
        new_array = np.take(new_array, keep_indices, axis=axis)
    return new_array


def _slice_row_count_initializer_axes(
    model: onnx.ModelProto,
    keep_indices: np.ndarray,
    row_count: int,
    rows: list[dict[str, Any]],
) -> int:
    updated = 0
    for initializer in model.graph.initializer:
        if initializer.name in PLACEMENT_TABLES:
            continue
        array = numpy_helper.to_array(initializer)
        new_array = _slice_row_count_axes(array, keep_indices, row_count)
        if new_array is None:
            continue
        _replace_initializer(initializer, new_array, "row_count_axes_sliced", rows)
        updated += 1
    return updated


def _replace_row_count_initializer_values(
    model: onnx.ModelProto,
    old_row_count: int,
    new_row_count: int,
    rows: list[dict[str, Any]],
) -> int:
    updated = 0
    for initializer in model.graph.initializer:
        if initializer.name in PLACEMENT_TABLES:
            continue
        array = numpy_helper.to_array(initializer)
        new_array = _replace_row_count_values(array, old_row_count, new_row_count)
        if new_array is None:
            continue
        _replace_initializer(initializer, new_array, "row_count_values_updated", rows)
        updated += 1
    return updated


def _replace_constant_tensor(
    node: onnx.NodeProto,
    attr: onnx.AttributeProto,
    old_array: np.ndarray,
    new_array: np.ndarray,
    action: str,
    rows: list[dict[str, Any]],
) -> None:
    attr.CopyFrom(
        onnx.helper.make_attribute(
            "value",
            numpy_helper.from_array(new_array),
        )
    )
    rows.append(
        {
            "value_name": node.output[0] if node.output else node.name,
            "value_kind": "constant_node",
            "old_shape": _shape_text(old_array.shape),
            "new_shape": _shape_text(new_array.shape),
            "old_num_elements": int(old_array.size),
            "new_num_elements": int(new_array.size),
            "action": action,
        }
    )


def _slice_row_count_constant_axes(
    model: onnx.ModelProto,
    keep_indices: np.ndarray,
    row_count: int,
    rows: list[dict[str, Any]],
) -> int:
    updated = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            old_array = numpy_helper.to_array(onnx.helper.get_attribute_value(attr))
            new_array = _slice_row_count_axes(old_array, keep_indices, row_count)
            if new_array is None:
                continue
            _replace_constant_tensor(
                node,
                attr,
                old_array,
                new_array,
                "row_count_axes_sliced",
                rows,
            )
            updated += 1
    return updated


def _replace_row_count_constant_values(
    model: onnx.ModelProto,
    old_row_count: int,
    new_row_count: int,
    rows: list[dict[str, Any]],
) -> int:
    updated = 0
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            old_tensor = onnx.helper.get_attribute_value(attr)
            old_array = numpy_helper.to_array(old_tensor)
            new_array = _replace_row_count_values(old_array, old_row_count, new_row_count)
            if new_array is None:
                continue
            _replace_constant_tensor(
                node,
                attr,
                old_array,
                new_array,
                "row_count_values_updated",
                rows,
            )
            updated += 1
    return updated


def inspect_task157_row_count_values(
    model_path: str = DEFAULT_SOURCE,
) -> dict[str, Any]:
    """List initializers and Constant nodes containing the placement row count."""
    model = onnx.load(model_path)
    hits: list[dict[str, Any]] = []
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer)
        if any(int(dim) == ROW_COUNT for dim in array.shape):
            hits.append(
                {
                    "value_name": initializer.name,
                    "value_kind": "initializer_dim",
                    "shape": _shape_text(array.shape),
                    "dtype": str(array.dtype),
                    "count_1305": "",
                }
            )
        if np.issubdtype(array.dtype, np.integer) and np.any(array == ROW_COUNT):
            hits.append(
                {
                    "value_name": initializer.name,
                    "value_kind": "initializer",
                    "shape": _shape_text(array.shape),
                    "dtype": str(array.dtype),
                    "count_1305": int(np.count_nonzero(array == ROW_COUNT)),
                }
            )
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name != "value":
                continue
            array = numpy_helper.to_array(onnx.helper.get_attribute_value(attr))
            if any(int(dim) == ROW_COUNT for dim in array.shape):
                hits.append(
                    {
                        "value_name": node.output[0] if node.output else node.name,
                        "value_kind": "constant_node_dim",
                        "shape": _shape_text(array.shape),
                        "dtype": str(array.dtype),
                        "count_1305": "",
                    }
                )
            if np.issubdtype(array.dtype, np.integer) and np.any(array == ROW_COUNT):
                hits.append(
                    {
                        "value_name": node.output[0] if node.output else node.name,
                        "value_kind": "constant_node",
                        "shape": _shape_text(array.shape),
                        "dtype": str(array.dtype),
                        "count_1305": int(np.count_nonzero(array == ROW_COUNT)),
                    }
                )
    summary = {"model_path": model_path, "row_count_hits": hits}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def prune_task157_placements(
    source_model: str = DEFAULT_SOURCE,
    output_model: str | None = None,
    report_path: str = DEFAULT_REPORT,
    observed_report: str = DEFAULT_OBSERVED_REPORT,
    mode: str = "conservative",
    row_list: str | None = None,
) -> dict[str, Any]:
    """Slice task157 placement rows and update row-count constants."""
    mode_names = {
        "conservative": "PlacementConservative",
        "medium": "PlacementMedium",
        "observed": "PlacementObserved",
        "component": "PlacementPruneComponent",
        "drop-list": "PlacementPruneDropList",
        "keep-list": "PlacementPruneKeepList",
    }
    if mode not in mode_names:
        raise ValueError(
            "mode must be one of "
            f"{', '.join(sorted(mode_names))}, got {mode}"
        )
    source_path = Path(source_model)
    if not source_path.is_file():
        raise FileNotFoundError(f"source model does not exist: {source_model}")
    model = onnx.load(str(source_path))
    arrays = _initializer_arrays(model)
    missing = [name for name in PLACEMENT_TABLES if name not in arrays]
    if missing:
        raise ValueError(f"source model is missing task157 placement tables: {missing}")
    source_row_count = int(arrays["plac_idx_963"].shape[0])
    if source_row_count <= 0:
        raise ValueError(f"unexpected plac_idx row count: {arrays['plac_idx_963'].shape}")
    if int(arrays["expand_idx_983"].shape[0]) != source_row_count:
        raise ValueError(
            "placement tables have inconsistent first dimensions: "
            f"plac_idx={arrays['plac_idx_963'].shape}, "
            f"expand_idx={arrays['expand_idx_983'].shape}"
        )

    observed_rows = _read_observed_rows(observed_report)
    out_of_range_observed = [
        row for row in sorted(observed_rows) if row < 0 or row >= source_row_count
    ]
    if out_of_range_observed:
        raise ValueError(
            "observed placement rows are outside source model row range: "
            f"{out_of_range_observed[:10]}"
        )
    keep_indices = _keep_indices_for_mode(mode, arrays, observed_rows, row_list)
    new_row_count = int(keep_indices.size)

    rows: list[dict[str, Any]] = []
    for initializer in model.graph.initializer:
        if initializer.name == "plac_idx_963":
            _replace_initializer(
                initializer,
                arrays["plac_idx_963"][keep_indices],
                "sliced_rows",
                rows,
            )
        elif initializer.name == "expand_idx_983":
            _replace_initializer(
                initializer,
                arrays["expand_idx_983"][keep_indices],
                "sliced_rows",
                rows,
            )

    updated_initializer_axes = _slice_row_count_initializer_axes(
        model,
        keep_indices,
        source_row_count,
        rows,
    )
    updated_constant_axes = _slice_row_count_constant_axes(
        model,
        keep_indices,
        source_row_count,
        rows,
    )
    updated_initializer_values = _replace_row_count_initializer_values(
        model,
        source_row_count,
        new_row_count,
        rows,
    )
    updated_constant_values = _replace_row_count_constant_values(
        model,
        source_row_count,
        new_row_count,
        rows,
    )

    del model.graph.value_info[:]
    onnx.checker.check_model(model)

    if output_model is None:
        output_model = str(
            Path(DEFAULT_CANDIDATE_DIR)
            / f"task157_{mode_names[mode]}.onnx"
        )
    output_path = Path(output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    onnx.checker.check_model(str(output_path))

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PRUNE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    source_cost = estimate_model_cost(str(source_path))
    output_cost = estimate_model_cost(str(output_path))
    summary = {
        "source_model": str(source_path),
        "output_model": str(output_path),
        "report_path": str(report),
        "observed_report": observed_report,
        "mode": mode,
        "original_rows": source_row_count,
        "kept_rows": new_row_count,
        "observed_rows": len(observed_rows),
        "updated_initializer_axes": updated_initializer_axes,
        "updated_constant_axes": updated_constant_axes,
        "updated_initializer_values": updated_initializer_values,
        "updated_constant_values": updated_constant_values,
        "source_estimated_cost": source_cost["estimated_cost"],
        "output_estimated_cost": output_cost["estimated_cost"],
        "source_file_size_bytes": source_cost["file_size_bytes"],
        "output_file_size_bytes": output_cost["file_size_bytes"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def generate_task157_placement_candidates(
    source_model: str = DEFAULT_SOURCE,
    task_path: str = DEFAULT_TASK,
    candidate_dir: str = DEFAULT_CANDIDATE_DIR,
    observed_report: str = DEFAULT_OBSERVED_REPORT,
    observed_summary: str = DEFAULT_SUMMARY,
    report_dir: str = "outputs/reports",
) -> dict[str, Any]:
    """Observe task157 placement rows, then build the three requested candidates."""
    observe_summary = observe_task157_selected_rows(
        model_path=source_model,
        task_path=task_path,
        report_path=observed_report,
        summary_path=observed_summary,
    )
    mode_to_name = {
        "conservative": "PlacementConservative",
        "medium": "PlacementMedium",
        "observed": "PlacementObserved",
    }
    candidates: list[dict[str, Any]] = []
    for mode, name in mode_to_name.items():
        output_model = str(Path(candidate_dir) / f"task157_{name}.onnx")
        report_path = str(Path(report_dir) / f"task157_placement_{mode}.csv")
        candidates.append(
            prune_task157_placements(
                source_model=source_model,
                output_model=output_model,
                report_path=report_path,
                observed_report=observed_report,
                mode=mode,
            )
        )
    summary = {
        "source_model": source_model,
        "task_path": task_path,
        "candidate_dir": candidate_dir,
        "observed_report": observed_report,
        "observed_summary": observed_summary,
        "observe_summary": observe_summary,
        "candidates": candidates,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    observe = subparsers.add_parser("observe")
    observe.add_argument("--model", default=DEFAULT_SOURCE)
    observe.add_argument("--task", default=DEFAULT_TASK)
    observe.add_argument("--report", default=DEFAULT_OBSERVED_REPORT)
    observe.add_argument("--summary", default=DEFAULT_SUMMARY)

    inspect = subparsers.add_parser("inspect-row-counts")
    inspect.add_argument("--model", default=DEFAULT_SOURCE)

    prune = subparsers.add_parser("prune")
    prune.add_argument("--source", default=DEFAULT_SOURCE)
    prune.add_argument("--output")
    prune.add_argument("--report", default=DEFAULT_REPORT)
    prune.add_argument("--observed-report", default=DEFAULT_OBSERVED_REPORT)
    prune.add_argument(
        "--mode",
        default="conservative",
        choices=(
            "conservative",
            "medium",
            "observed",
            "component",
            "drop-list",
            "keep-list",
        ),
    )
    prune.add_argument("--row-list")

    generate = subparsers.add_parser("generate-candidates")
    generate.add_argument("--source", default=DEFAULT_SOURCE)
    generate.add_argument("--task", default=DEFAULT_TASK)
    generate.add_argument("--candidate-dir", default=DEFAULT_CANDIDATE_DIR)
    generate.add_argument("--observed-report", default=DEFAULT_OBSERVED_REPORT)
    generate.add_argument("--observed-summary", default=DEFAULT_SUMMARY)
    generate.add_argument("--report-dir", default="outputs/reports")

    args = parser.parse_args()
    if args.command == "observe":
        observe_task157_selected_rows(
            model_path=args.model,
            task_path=args.task,
            report_path=args.report,
            summary_path=args.summary,
        )
    elif args.command == "inspect-row-counts":
        inspect_task157_row_count_values(args.model)
    elif args.command == "prune":
        prune_task157_placements(
            source_model=args.source,
            output_model=args.output,
            report_path=args.report,
            observed_report=args.observed_report,
            mode=args.mode,
            row_list=args.row_list,
        )
    elif args.command == "generate-candidates":
        generate_task157_placement_candidates(
            source_model=args.source,
            task_path=args.task,
            candidate_dir=args.candidate_dir,
            observed_report=args.observed_report,
            observed_summary=args.observed_summary,
            report_dir=args.report_dir,
        )


if __name__ == "__main__":
    main()
