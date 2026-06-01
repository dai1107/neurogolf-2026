"""Create a repaired archive copy by masking fixed-shape padding outputs."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from .arc_io import load_all_tasks
from .encoding import DEFAULT_HEIGHT, DEFAULT_WIDTH


def _train_output_shapes(task: dict) -> set[tuple[int, int]]:
    return {(len(case["output"]), len(case["output"][0])) for case in task["train"]}


def _graph_output_elem_type(model: onnx.ModelProto) -> int:
    output_type = model.graph.output[0].type.tensor_type
    if not model.graph.output[0].type.HasField("tensor_type"):
        raise ValueError("graph output is not a tensor")
    return int(output_type.elem_type)


def _np_dtype_for_elem_type(elem_type: int) -> np.dtype:
    return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(elem_type))


def mask_model_output(source_path: Path, output_path: Path, active_height: int, active_width: int) -> None:
    """Append output *= active_mask for a fixed active output rectangle."""
    model = onnx.load(str(source_path))
    if len(model.graph.output) != 1:
        raise ValueError(f"expected exactly one graph output: {source_path}")
    original_output = model.graph.output[0].name
    unmasked_output = f"{original_output}_unmasked"
    if any(unmasked_output in node.output for node in model.graph.node):
        unmasked_output = f"{original_output}_unmasked_0"
    renamed = False
    for node in model.graph.node:
        for index, output_name in enumerate(node.output):
            if output_name == original_output:
                node.output[index] = unmasked_output
                renamed = True
    if not renamed:
        raise ValueError(f"could not find producer for graph output {original_output!r}")

    elem_type = _graph_output_elem_type(model)
    mask = np.zeros((1, 1, DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=_np_dtype_for_elem_type(elem_type))
    mask[:, :, :active_height, :active_width] = 1.0
    model.graph.initializer.append(numpy_helper.from_array(mask, name="ActiveOutputMask"))
    model.graph.node.append(
        helper.make_node(
            "Mul",
            [unmasked_output, "ActiveOutputMask"],
            [original_output],
            name="mask_padding_output",
        )
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))


def dynamic_active_mask_model_output(source_path: Path, output_path: Path) -> None:
    """Append output *= (sum(input channels) > 0), preserving the graph output name."""
    model = onnx.load(str(source_path))
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError(f"expected exactly one input and one output: {source_path}")
    input_name = model.graph.input[0].name
    original_output = model.graph.output[0].name
    unmasked_output = f"{original_output}_unmasked"
    if any(unmasked_output in node.output for node in model.graph.node):
        unmasked_output = f"{original_output}_unmasked_0"
    renamed = False
    for node in model.graph.node:
        for index, output_name in enumerate(node.output):
            if output_name == original_output:
                node.output[index] = unmasked_output
                renamed = True
    if not renamed:
        raise ValueError(f"could not find producer for graph output {original_output!r}")

    active_weights_name = "ActiveInputW"
    if any(initializer.name == active_weights_name for initializer in model.graph.initializer):
        active_weights_name = "ActiveInputW_0"
    active_zero_name = "ActiveZero"
    if any(initializer.name == active_zero_name for initializer in model.graph.initializer):
        active_zero_name = "ActiveZero_0"
    elem_type = _graph_output_elem_type(model)
    model.graph.initializer.append(numpy_helper.from_array(np.ones((1, 10, 1, 1), dtype=np.float32), name=active_weights_name))
    model.graph.initializer.append(numpy_helper.from_array(np.array(0.0, dtype=np.float32), name=active_zero_name))
    model.graph.node.extend(
        [
            helper.make_node(
                "Conv",
                [input_name, active_weights_name],
                ["active_sum"],
                name="active_sum",
                kernel_shape=[1, 1],
                strides=[1, 1],
            ),
            helper.make_node("Greater", ["active_sum", active_zero_name], ["active_bool"], name="active_bool"),
            helper.make_node("Cast", ["active_bool"], ["active_float"], name="active_float", to=elem_type),
            helper.make_node("Mul", [unmasked_output, "active_float"], [original_output], name="mask_padding_output"),
        ]
    )
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))


def repair_archive_padding(
    data_dir: str,
    archive_dir: str,
    blend_report: str,
    output_dir: str,
    repair_report: str,
    task_ids: set[str] | None = None,
    mode: str = "static",
) -> list[dict[str, str]]:
    tasks = load_all_tasks(data_dir)
    archive_root = Path(archive_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for source in archive_root.glob("task*.onnx"):
        shutil.copyfile(source, output_root / source.name)

    rows: list[dict[str, str]] = []
    if task_ids is None:
        with Path(blend_report).open("r", newline="", encoding="utf-8") as handle:
            selected_task_ids = {
                row["task_id"]
                for row in csv.DictReader(handle)
                if row.get("archive_failure_reason", "").startswith("nonzero_padding_cells")
            }
    else:
        selected_task_ids = set(task_ids)

    for task_id in sorted(selected_task_ids):
        shapes = _train_output_shapes(tasks[task_id])
        if mode == "active":
            if any((len(case["input"]), len(case["input"][0])) != (len(case["output"]), len(case["output"][0])) for case in tasks[task_id]["train"]):
                rows.append({"task_id": task_id, "status": "skipped", "reason": "active_mask_requires_same_size"})
                continue
            try:
                dynamic_active_mask_model_output(
                    archive_root / f"{task_id}.onnx",
                    output_root / f"{task_id}.onnx",
                )
            except Exception as exc:
                rows.append({"task_id": task_id, "status": "failed", "reason": str(exc)})
                continue
            rows.append({"task_id": task_id, "status": "repaired", "reason": "masked_to_input_active_area"})
            continue
        if len(shapes) != 1:
            rows.append({"task_id": task_id, "status": "skipped", "reason": "variable_output_shapes"})
            continue
        active_height, active_width = next(iter(shapes))
        try:
            mask_model_output(
                archive_root / f"{task_id}.onnx",
                output_root / f"{task_id}.onnx",
                active_height,
                active_width,
            )
        except Exception as exc:
            rows.append({"task_id": task_id, "status": "failed", "reason": str(exc)})
            continue
        rows.append({"task_id": task_id, "status": "repaired", "reason": f"masked_to_{active_height}x{active_width}"})

    Path(repair_report).parent.mkdir(parents=True, exist_ok=True)
    with Path(repair_report).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["task_id", "status", "reason"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"repair_rows = {len(rows)}")
    print(f"repair_report = {repair_report}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="task")
    parser.add_argument("--archive-dir", default="archive")
    parser.add_argument("--blend-report", default="outputs/reports/archive_blend_report.csv")
    parser.add_argument("--output-dir", default="outputs/archive_repaired")
    parser.add_argument("--repair-report", default="outputs/reports/archive_padding_repair_report.csv")
    parser.add_argument("--task-ids", default="", help="Optional comma-separated task ids to repair regardless of blend report")
    parser.add_argument("--mode", choices=["static", "active"], default="static")
    args = parser.parse_args()
    task_ids = {item.strip() for item in args.task_ids.split(",") if item.strip()} or None
    repair_archive_padding(
        data_dir=args.data_dir,
        archive_dir=args.archive_dir,
        blend_report=args.blend_report,
        output_dir=args.output_dir,
        repair_report=args.repair_report,
        task_ids=task_ids,
        mode=args.mode,
    )


if __name__ == "__main__":
    main()
