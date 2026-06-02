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


def _set_static_tensor_shape(value_info: onnx.ValueInfoProto, dims: list[int]) -> None:
    tensor_type = value_info.type.tensor_type
    if not value_info.type.HasField("tensor_type"):
        raise ValueError(f"{value_info.name} is not a tensor")
    del tensor_type.shape.dim[:]
    for dim in dims:
        tensor_type.shape.dim.add().dim_value = int(dim)


def _static_shape_map(model: onnx.ModelProto) -> dict[str, list[int]]:
    inferred = onnx.shape_inference.infer_shapes(model)
    shapes: dict[str, list[int]] = {}
    values = list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)
    for value_info in values:
        if not value_info.type.HasField("tensor_type"):
            continue
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims: list[int] = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value") or dim.dim_value <= 0:
                break
            dims.append(int(dim.dim_value))
        else:
            shapes[value_info.name] = dims
    return shapes


def _get_ints_attribute(node: onnx.NodeProto, name: str) -> list[int] | None:
    for attribute in node.attribute:
        if attribute.name == name:
            return [int(value) for value in attribute.ints]
    return None


def _set_ints_attribute(node: onnx.NodeProto, name: str, values: list[int]) -> None:
    for attribute in node.attribute:
        if attribute.name == name:
            del attribute.ints[:]
            attribute.ints.extend(int(value) for value in values)
            return
    node.attribute.append(helper.make_attribute(name, values))


def _unique_name(base: str, used_names: set[str]) -> str:
    candidate = base
    suffix = 0
    while candidate in used_names:
        suffix += 1
        candidate = f"{base}_{suffix}"
    used_names.add(candidate)
    return candidate


def repair_negative_conv_pads(source_path: Path, output_path: Path) -> int:
    """Rewrite Conv nodes with negative pads as Slice + non-negative-pad Conv."""
    model = onnx.load(str(source_path))
    shape_by_name = _static_shape_map(model)
    used_names = {
        value.name
        for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    }
    for initializer in model.graph.initializer:
        used_names.add(initializer.name)
    for node in model.graph.node:
        if node.name:
            used_names.add(node.name)
        used_names.update(node.output)

    new_nodes: list[onnx.NodeProto] = []
    initializer_cache: dict[tuple[str, tuple[int, ...]], str] = {}

    def add_int64_initializer(base_name: str, values: np.ndarray) -> str:
        key = (values.dtype.str, tuple(int(value) for value in values.tolist()))
        if key in initializer_cache:
            return initializer_cache[key]
        name = _unique_name(base_name, used_names)
        model.graph.initializer.append(numpy_helper.from_array(values, name=name))
        initializer_cache[key] = name
        return name

    rewritten = 0
    for node in model.graph.node:
        pads = _get_ints_attribute(node, "pads")
        if node.op_type != "Conv" or pads is None or len(pads) != 4 or all(value >= 0 for value in pads):
            new_nodes.append(node)
            continue
        if len(node.input) < 2:
            raise ValueError(f"Conv node {node.name!r} does not have data and weight inputs")
        input_shape = shape_by_name.get(node.input[0])
        if input_shape is None or len(input_shape) != 4:
            raise ValueError(f"missing static NCHW input shape for Conv node {node.name!r}")

        top_crop = max(-pads[0], 0)
        left_crop = max(-pads[1], 0)
        bottom_crop = max(-pads[2], 0)
        right_crop = max(-pads[3], 0)
        height = input_shape[2]
        width = input_shape[3]
        if top_crop + bottom_crop >= height or left_crop + right_crop >= width:
            raise ValueError(f"negative pads crop away Conv input for node {node.name!r}")

        starts = np.array([top_crop, left_crop], dtype=np.int64)
        ends = np.array([height - bottom_crop, width - right_crop], dtype=np.int64)
        axes = np.array([2, 3], dtype=np.int64)
        steps = np.array([1, 1], dtype=np.int64)
        safe_node_name = (node.name or f"conv_{rewritten}").strip("/").replace("/", "_")
        starts_name = add_int64_initializer(f"{safe_node_name}_crop_starts", starts)
        ends_name = add_int64_initializer(f"{safe_node_name}_crop_ends", ends)
        axes_name = add_int64_initializer(f"{safe_node_name}_crop_axes", axes)
        steps_name = add_int64_initializer(f"{safe_node_name}_crop_steps", steps)
        crop_output = _unique_name(f"{safe_node_name}_cropped_input", used_names)
        new_nodes.append(
            helper.make_node(
                "Slice",
                [node.input[0], starts_name, ends_name, axes_name, steps_name],
                [crop_output],
                name=_unique_name(f"{safe_node_name}_crop", used_names),
            )
        )
        node.input[0] = crop_output
        _set_ints_attribute(
            node,
            "pads",
            [max(pads[0], 0), max(pads[1], 0), max(pads[2], 0), max(pads[3], 0)],
        )
        new_nodes.append(node)
        rewritten += 1

    if rewritten == 0:
        raise ValueError(f"no Conv node with negative pads found: {source_path}")

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    onnx.checker.check_model(onnx.shape_inference.infer_shapes(model))
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    return rewritten


def repair_task277_static_pads(source_path: Path, output_path: Path) -> None:
    """Replace task277's constant pad subgraphs with static pad initializers."""
    model = onnx.load(str(source_path))
    replacements = {
        "/Pad": np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64),
        "/Pad_1": np.array([0, 0, 0, 0, 0, 0, 20, 20], dtype=np.int64),
    }
    replaced: set[str] = set()
    existing_initializer_names = {initializer.name for initializer in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Pad" or node.name not in replacements:
            continue
        if len(node.input) < 2:
            raise ValueError(f"Pad node {node.name!r} does not have a pads input")
        initializer_name = f"{node.name.strip('/').replace('/', '_')}_static_pads"
        if initializer_name in existing_initializer_names:
            raise ValueError(f"initializer already exists: {initializer_name}")
        model.graph.initializer.append(
            numpy_helper.from_array(replacements[node.name], name=initializer_name)
        )
        node.input[1] = initializer_name
        replaced.add(node.name)

    missing = sorted(set(replacements) - replaced)
    if missing:
        raise ValueError(f"missing expected Pad nodes: {missing}")

    if len(model.graph.output) != 1:
        raise ValueError(f"expected exactly one graph output: {source_path}")
    _set_static_tensor_shape(model.graph.output[0], [1, 10, DEFAULT_HEIGHT, DEFAULT_WIDTH])

    onnx.checker.check_model(onnx.shape_inference.infer_shapes(model))
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
            if mode == "negative_conv_pads":
                selected_task_ids = {
                    row["task_id"]
                    for row in csv.DictReader(handle)
                    if row.get("archive_failure_reason", "").startswith("evaluation_subprocess_failed")
                }
            else:
                selected_task_ids = {
                    row["task_id"]
                    for row in csv.DictReader(handle)
                    if row.get("archive_failure_reason", "").startswith("nonzero_padding_cells")
                }
    else:
        selected_task_ids = set(task_ids)

    for task_id in sorted(selected_task_ids):
        shapes = _train_output_shapes(tasks[task_id])
        if mode == "negative_conv_pads":
            try:
                rewritten = repair_negative_conv_pads(
                    archive_root / f"{task_id}.onnx",
                    output_root / f"{task_id}.onnx",
                )
            except Exception as exc:
                rows.append({"task_id": task_id, "status": "failed", "reason": str(exc)})
                continue
            rows.append({"task_id": task_id, "status": "repaired", "reason": f"rewritten_negative_conv_pads={rewritten}"})
            continue
        if mode == "task277_static_pads":
            if task_id != "task277":
                rows.append({"task_id": task_id, "status": "skipped", "reason": "not_task277"})
                continue
            try:
                repair_task277_static_pads(
                    archive_root / f"{task_id}.onnx",
                    output_root / f"{task_id}.onnx",
                )
            except Exception as exc:
                rows.append({"task_id": task_id, "status": "failed", "reason": str(exc)})
                continue
            rows.append({"task_id": task_id, "status": "repaired", "reason": "static_pad_inputs"})
            continue
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
    parser.add_argument("--mode", choices=["static", "active", "task277_static_pads", "negative_conv_pads"], default="static")
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
