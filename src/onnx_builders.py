"""Reusable ONNX builders for first-version NeuroGolf rules."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .encoding import DEFAULT_COLORS, DEFAULT_HEIGHT, DEFAULT_WIDTH


def _value_info(name: str, channels: int, height: int, width: int) -> onnx.ValueInfoProto:
    return helper.make_tensor_value_info(
        name,
        TensorProto.FLOAT,
        [1, channels, height, width],
    )


def _save_checked_model(
    output_path: str,
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    graph_name: str,
    num_channels: int,
    height: int,
    width: int,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    graph = helper.make_graph(
        nodes=nodes,
        name=graph_name,
        inputs=[_value_info("input", num_channels, height, width)],
        outputs=[_value_info("output", num_channels, height, width)],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="neurogolf-2026",
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 10)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(output))
    onnx.checker.check_model(str(output))


def _bool_mask(mask: np.ndarray, name: str) -> onnx.TensorProto:
    return numpy_helper.from_array(np.asarray(mask, dtype=np.bool_), name=name)


def _cast_to_float(input_name: str, output_name: str) -> onnx.NodeProto:
    return helper.make_node(
        "Cast",
        [input_name],
        [output_name],
        name=output_name,
        to=TensorProto.FLOAT,
    )


def build_identity_model(
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a zero-parameter Identity model."""
    node = helper.make_node("Identity", ["input"], ["output"], name="output")
    _save_checked_model(
        output_path,
        [node],
        [],
        "identity",
        num_channels,
        height,
        width,
    )


def build_color_map_model(
    color_map: dict[int, int],
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a compact model that applies a global color map."""
    full_map: list[int] = []
    for old_color in range(num_channels):
        new_color = int(color_map.get(old_color, old_color))
        if new_color < 0 or new_color >= num_channels:
            raise ValueError(f"mapped color {new_color} is outside 0..{num_channels - 1}")
        full_map.append(new_color)

    if len(set(full_map)) == num_channels:
        inverse = np.zeros((num_channels,), dtype=np.int32)
        for old_color, new_color in enumerate(full_map):
            inverse[new_color] = old_color
        node = helper.make_node(
            "Gather",
            ["input", "ChannelIndices"],
            ["output"],
            name="output",
            axis=1,
        )
        _save_checked_model(
            output_path,
            [node],
            [numpy_helper.from_array(inverse, name="ChannelIndices")],
            "color_map_gather",
            num_channels,
            height,
            width,
        )
        return

    weights = np.zeros((num_channels, num_channels, 1, 1), dtype=np.float32)
    for old_color, new_color in enumerate(full_map):
        weights[new_color, old_color, 0, 0] = 1.0

    weight_initializer = numpy_helper.from_array(weights, name="W")
    node = helper.make_node(
        "Conv",
        ["input", "W"],
        ["output"],
        name="output",
        kernel_shape=[1, 1],
        strides=[1, 1],
    )
    _save_checked_model(
        output_path,
        [node],
        [weight_initializer],
        "color_map",
        num_channels,
        height,
        width,
    )


def build_small_translation_model(
    dy: int,
    dx: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    active_height: int | None = None,
    active_width: int | None = None,
    fill_color: int = 0,
) -> None:
    """Build a translation model with deterministic color-0 fill."""
    active_height = height if active_height is None else active_height
    active_width = width if active_width is None else active_width
    if active_height <= 0 or active_height > height:
        raise ValueError("active_height must be within 1..height")
    if active_width <= 0 or active_width > width:
        raise ValueError("active_width must be within 1..width")
    if abs(dy) >= active_height or abs(dx) >= active_width:
        raise ValueError("dy/dx must leave at least one source row or column in bounds")
    if fill_color < 0 or fill_color >= num_channels:
        raise ValueError(f"fill_color must be within 0..{num_channels - 1}")

    rows: list[int] = []
    cols: list[int] = []
    for row in range(active_height):
        source_row = row - dy
        rows.append(source_row if 0 <= source_row < active_height else 0)
    for col in range(active_width):
        source_col = col - dx
        cols.append(source_col if 0 <= source_col < active_width else 0)

    valid_mask = np.zeros((1, 1, height, width), dtype=np.float32)
    fill_mask = np.zeros((1, 1, height, width), dtype=np.float32)
    for row in range(active_height):
        for col in range(active_width):
            source_row = row - dy
            source_col = col - dx
            if 0 <= source_row < active_height and 0 <= source_col < active_width:
                valid_mask[0, 0, row, col] = 1.0
            else:
                fill_mask[0, 0, row, col] = 1.0

    fill_value = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    fill_value[:, fill_color, :, :] = 1.0

    initializers = [
        numpy_helper.from_array(_padded_indices(rows, height, height), name="RowIndices"),
        numpy_helper.from_array(_padded_indices(cols, width, width), name="ColIndices"),
        _bool_mask(valid_mask, "ValidMask"),
        _bool_mask(fill_mask, "FillMask"),
        numpy_helper.from_array(fill_value, name="FillColor"),
    ]
    nodes = [
        helper.make_node("Gather", ["input", "RowIndices"], ["gather_rows"], name="gather_rows", axis=2),
        helper.make_node("Gather", ["gather_rows", "ColIndices"], ["remapped"], name="remapped", axis=3),
        _cast_to_float("ValidMask", "ValidMaskFloat"),
        _cast_to_float("FillMask", "FillMaskFloat"),
        helper.make_node("Mul", ["remapped", "ValidMaskFloat"], ["shifted"], name="shifted"),
        helper.make_node("Mul", ["FillColor", "FillMaskFloat"], ["edge_fill"], name="edge_fill"),
        helper.make_node("Add", ["shifted", "edge_fill"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "one_step_translation",
        num_channels,
        height,
        width,
    )


def _padded_indices(indices: list[int], size: int, limit: int, pad_value: int = 0) -> np.ndarray:
    if len(indices) > size:
        raise ValueError(f"too many indices: {len(indices)} > {size}")
    padded = list(indices) + [pad_value] * (size - len(indices))
    if any(index < 0 or index >= limit for index in padded):
        raise ValueError(f"indices must be within 0..{limit - 1}: {indices}")
    return np.asarray(padded, dtype=np.int32)


class SpatialRemapBuilder:
    """Build a static per-cell spatial gather with optional color remapping.

    The remap is expressed as flattened spatial indices, so each output cell can
    read from an arbitrary input row/column. Cells outside ``active_mask`` are
    zeroed and therefore do not need meaningful source coordinates.
    """

    def __init__(
        self,
        row_indices: Sequence[int] | Sequence[Sequence[int]],
        col_indices: Sequence[int] | Sequence[Sequence[int]],
        active_mask: np.ndarray | None = None,
        color_map: dict[int, int] | None = None,
        num_channels: int = DEFAULT_COLORS,
        height: int = DEFAULT_HEIGHT,
        width: int = DEFAULT_WIDTH,
    ) -> None:
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.color_map = color_map
        self.rows, self.cols, self.active_mask = self._normalize_indices(
            row_indices,
            col_indices,
            active_mask,
        )

    def _normalize_indices(
        self,
        row_indices: Sequence[int] | Sequence[Sequence[int]],
        col_indices: Sequence[int] | Sequence[Sequence[int]],
        active_mask: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = np.asarray(row_indices, dtype=np.int32)
        cols = np.asarray(col_indices, dtype=np.int32)
        if rows.ndim not in {1, 2} or cols.ndim not in {1, 2}:
            raise ValueError("row_indices and col_indices must be 1D or 2D")
        if rows.ndim != cols.ndim:
            raise ValueError("row_indices and col_indices must have the same rank")

        if rows.ndim == 1:
            if len(rows) > self.height or len(cols) > self.width:
                raise ValueError("1D remap indices exceed static output shape")
            grid_rows = np.zeros((self.height, self.width), dtype=np.int32)
            grid_cols = np.zeros((self.height, self.width), dtype=np.int32)
            grid_rows[: len(rows), : len(cols)] = rows[:, None]
            grid_cols[: len(rows), : len(cols)] = cols[None, :]
            default_mask = np.zeros((self.height, self.width), dtype=np.float32)
            default_mask[: len(rows), : len(cols)] = 1.0
        else:
            if rows.shape != cols.shape:
                raise ValueError("2D row_indices and col_indices must have the same shape")
            if rows.shape[0] > self.height or rows.shape[1] > self.width:
                raise ValueError("2D remap indices exceed static output shape")
            grid_rows = np.zeros((self.height, self.width), dtype=np.int32)
            grid_cols = np.zeros((self.height, self.width), dtype=np.int32)
            grid_rows[: rows.shape[0], : rows.shape[1]] = rows
            grid_cols[: rows.shape[0], : rows.shape[1]] = cols
            default_mask = np.zeros((self.height, self.width), dtype=np.float32)
            default_mask[: rows.shape[0], : rows.shape[1]] = 1.0

        mask = default_mask if active_mask is None else np.asarray(active_mask, dtype=np.float32)
        if mask.shape == (1, 1, self.height, self.width):
            mask_2d = mask[0, 0]
        elif mask.shape == (self.height, self.width):
            mask_2d = mask
        else:
            raise ValueError("active_mask must be HxW or 1x1xHxW")

        inactive = mask_2d <= 0.0
        grid_rows[inactive] = 0
        grid_cols[inactive] = 0
        if np.any((grid_rows < 0) | (grid_rows >= self.height) | (grid_cols < 0) | (grid_cols >= self.width)):
            raise ValueError("active remap indices must be within the static input shape")
        return grid_rows, grid_cols, mask_2d.reshape(1, 1, self.height, self.width)

    def _color_weights(self) -> np.ndarray:
        weights = np.zeros((self.num_channels, self.num_channels, 1, 1), dtype=np.float32)
        for old_color in range(self.num_channels):
            new_color = int((self.color_map or {}).get(old_color, old_color))
            if new_color < 0 or new_color >= self.num_channels:
                raise ValueError(f"mapped color {new_color} is outside 0..{self.num_channels - 1}")
            weights[new_color, old_color, 0, 0] = 1.0
        return weights

    def save(self, output_path: str, graph_name: str = "spatial_remap") -> None:
        flat_indices = (self.rows * self.width + self.cols).reshape(self.height * self.width).astype(np.int32)
        uses_active_mask = not np.all(self.active_mask > 0.0)
        initializers = [
            numpy_helper.from_array(np.asarray([1, self.num_channels, self.height * self.width], dtype=np.int64), name="FlatShape"),
            numpy_helper.from_array(flat_indices, name="FlatIndices"),
            numpy_helper.from_array(np.asarray([1, self.num_channels, self.height, self.width], dtype=np.int64), name="GridShape"),
        ]
        if uses_active_mask:
            initializers.append(_bool_mask(self.active_mask, "ActiveMask"))
        nodes = [
            helper.make_node("Reshape", ["input", "FlatShape"], ["flat_input"], name="flat_input"),
            helper.make_node("Gather", ["flat_input", "FlatIndices"], ["flat_remapped"], name="flat_remapped", axis=2),
            helper.make_node("Reshape", ["flat_remapped", "GridShape"], ["remapped"], name="remapped"),
        ]
        final_output = "remapped"
        if self.color_map is not None:
            initializers.append(numpy_helper.from_array(self._color_weights(), name="ColorMapW"))
            nodes.append(
                helper.make_node(
                    "Conv",
                    ["remapped", "ColorMapW"],
                    ["mapped"],
                    name="mapped",
                    kernel_shape=[1, 1],
                    strides=[1, 1],
                )
            )
            final_output = "mapped"
        if uses_active_mask:
            nodes.append(_cast_to_float("ActiveMask", "ActiveMaskFloat"))
            nodes.append(helper.make_node("Mul", [final_output, "ActiveMaskFloat"], ["output"], name="output"))
        else:
            nodes.append(helper.make_node("Identity", [final_output], ["output"], name="output"))
        _save_checked_model(
            output_path,
            nodes,
            initializers,
            graph_name,
            self.num_channels,
            self.height,
            self.width,
        )


def _padding_coordinate(
    input_active_height: int,
    input_active_width: int,
    height: int,
    width: int,
) -> tuple[int, int] | None:
    if input_active_height < height and input_active_width < width:
        return input_active_height, input_active_width
    if input_active_height < height:
        return input_active_height, 0
    if input_active_width < width:
        return 0, input_active_width
    return None


def build_spatial_remap_model(
    row_indices: list[int],
    col_indices: list[int],
    output_path: str,
    output_active_height: int | None = None,
    output_active_width: int | None = None,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    pad_row_index: int = 0,
    pad_col_index: int = 0,
    color_map: dict[int, int] | None = None,
) -> None:
    """Build a separable spatial remap: output[r, c] = input[row[r], col[c]]."""
    rows = list(row_indices) + [pad_row_index] * (height - len(row_indices))
    cols = list(col_indices) + [pad_col_index] * (width - len(col_indices))
    active_height = height if output_active_height is None else output_active_height
    active_width = width if output_active_width is None else output_active_width
    active_mask = np.ones((height, width), dtype=np.float32)
    if output_active_height is not None or output_active_width is not None:
        if active_height <= 0 or active_height > height:
            raise ValueError("output_active_height must be within 1..height")
        if active_width <= 0 or active_width > width:
            raise ValueError("output_active_width must be within 1..width")
        active_mask = np.zeros((height, width), dtype=np.float32)
        active_mask[:active_height, :active_width] = 1.0
    SpatialRemapBuilder(
        rows,
        cols,
        active_mask=active_mask,
        color_map=color_map,
        num_channels=num_channels,
        height=height,
        width=width,
    ).save(output_path)


def build_dynamic_non_background_bbox_crop_model(
    background_color: int,
    color_map: dict[int, int],
    output_path: str,
    transform: str = "identity",
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a dynamic crop of the bbox covering all non-background cells."""
    if background_color < 0 or background_color >= num_channels:
        raise ValueError(f"background_color must be within 0..{num_channels - 1}")
    if transform not in {"identity", "mirror_horizontal", "mirror_vertical"}:
        raise ValueError("transform must be identity, mirror_horizontal, or mirror_vertical")

    full_map: list[int] = []
    for old_color in range(num_channels):
        new_color = int(color_map.get(old_color, old_color))
        if new_color < 0 or new_color >= num_channels:
            raise ValueError(f"mapped color {new_color} is outside 0..{num_channels - 1}")
        full_map.append(new_color)

    select_weights = np.ones((1, num_channels, 1, 1), dtype=np.float32)
    select_weights[0, background_color, 0, 0] = 0.0
    color_weights = np.zeros((num_channels, num_channels, 1, 1), dtype=np.float32)
    for old_color, new_color in enumerate(full_map):
        color_weights[new_color, old_color, 0, 0] = 1.0

    offsets = np.arange(height, dtype=np.int64)
    reverse_indices = np.arange(height - 1, -1, -1, dtype=np.int64)
    one = np.array(1, dtype=np.int64)
    last_index = np.array(height - 1, dtype=np.int64)
    zero_threshold = np.array(0.0, dtype=np.float32)

    initializers = [
        numpy_helper.from_array(select_weights, name="SelectW"),
        numpy_helper.from_array(color_weights, name="ColorW"),
        numpy_helper.from_array(zero_threshold, name="Zero"),
        numpy_helper.from_array(offsets, name="Offsets"),
        numpy_helper.from_array(reverse_indices, name="ReverseIndices"),
        numpy_helper.from_array(one, name="One"),
        numpy_helper.from_array(last_index, name="LastIndex"),
    ]
    if transform == "mirror_vertical":
        row_source_nodes = [
            helper.make_node("Sub", ["bottom", "Offsets"], ["raw_rows"], name="raw_rows"),
            helper.make_node("Where", ["valid_rows", "raw_rows", "top"], ["source_rows"], name="source_rows"),
        ]
    else:
        row_source_nodes = [
            helper.make_node("Add", ["Offsets", "top"], ["raw_rows"], name="raw_rows"),
            helper.make_node("Where", ["valid_rows", "raw_rows", "top"], ["source_rows"], name="source_rows"),
        ]
    if transform == "mirror_horizontal":
        col_source_nodes = [
            helper.make_node("Sub", ["right", "Offsets"], ["raw_cols"], name="raw_cols"),
            helper.make_node("Where", ["valid_cols", "raw_cols", "left"], ["source_cols"], name="source_cols"),
        ]
    else:
        col_source_nodes = [
            helper.make_node("Add", ["Offsets", "left"], ["raw_cols"], name="raw_cols"),
            helper.make_node("Where", ["valid_cols", "raw_cols", "left"], ["source_cols"], name="source_cols"),
        ]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "SelectW"],
            ["selected_sum"],
            name="selected_sum",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Greater", ["selected_sum", "Zero"], ["selected_bool"], name="selected_bool"),
        _cast_to_float("selected_bool", "selected_float"),
        helper.make_node("ReduceSum", ["selected_float"], ["row_sum"], name="row_sum", axes=[3], keepdims=0),
        helper.make_node("ReduceSum", ["selected_float"], ["col_sum"], name="col_sum", axes=[2], keepdims=0),
        helper.make_node("Greater", ["row_sum", "Zero"], ["row_any"], name="row_any"),
        helper.make_node("Greater", ["col_sum", "Zero"], ["col_any"], name="col_any"),
        _cast_to_float("row_any", "row_any_float"),
        _cast_to_float("col_any", "col_any_float"),
        helper.make_node("ArgMax", ["row_any_float"], ["top_keep"], name="top_keep", axis=2, keepdims=0),
        helper.make_node("ArgMax", ["col_any_float"], ["left_keep"], name="left_keep", axis=2, keepdims=0),
        helper.make_node("Gather", ["row_any_float", "ReverseIndices"], ["row_any_reverse"], name="row_any_reverse", axis=2),
        helper.make_node("Gather", ["col_any_float", "ReverseIndices"], ["col_any_reverse"], name="col_any_reverse", axis=2),
        helper.make_node("ArgMax", ["row_any_reverse"], ["bottom_distance_keep"], name="bottom_distance_keep", axis=2, keepdims=0),
        helper.make_node("ArgMax", ["col_any_reverse"], ["right_distance_keep"], name="right_distance_keep", axis=2, keepdims=0),
        helper.make_node("Squeeze", ["top_keep"], ["top"], name="top", axes=[0, 1]),
        helper.make_node("Squeeze", ["left_keep"], ["left"], name="left", axes=[0, 1]),
        helper.make_node("Squeeze", ["bottom_distance_keep"], ["bottom_distance"], name="bottom_distance", axes=[0, 1]),
        helper.make_node("Squeeze", ["right_distance_keep"], ["right_distance"], name="right_distance", axes=[0, 1]),
        helper.make_node("Sub", ["LastIndex", "bottom_distance"], ["bottom"], name="bottom"),
        helper.make_node("Sub", ["LastIndex", "right_distance"], ["right"], name="right"),
        helper.make_node("Sub", ["bottom", "top"], ["bbox_height_minus_one"], name="bbox_height_minus_one"),
        helper.make_node("Sub", ["right", "left"], ["bbox_width_minus_one"], name="bbox_width_minus_one"),
        helper.make_node("Add", ["bbox_height_minus_one", "One"], ["bbox_height"], name="bbox_height"),
        helper.make_node("Add", ["bbox_width_minus_one", "One"], ["bbox_width"], name="bbox_width"),
        helper.make_node("Less", ["Offsets", "bbox_height"], ["valid_rows"], name="valid_rows"),
        helper.make_node("Less", ["Offsets", "bbox_width"], ["valid_cols"], name="valid_cols"),
        *row_source_nodes,
        *col_source_nodes,
        helper.make_node("Unsqueeze", ["valid_rows"], ["valid_rows_2d"], name="valid_rows_2d", axes=[1]),
        helper.make_node("Unsqueeze", ["valid_cols"], ["valid_cols_2d"], name="valid_cols_2d", axes=[0]),
        helper.make_node("And", ["valid_rows_2d", "valid_cols_2d"], ["active_bool"], name="active_bool"),
        helper.make_node("Gather", ["input", "source_rows"], ["gather_rows"], name="gather_rows", axis=2),
        helper.make_node("Gather", ["gather_rows", "source_cols"], ["remapped"], name="remapped", axis=3),
        helper.make_node(
            "Conv",
            ["remapped", "ColorW"],
            ["mapped"],
            name="mapped",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        _cast_to_float("active_bool", "active_float"),
        helper.make_node("Mul", ["mapped", "active_float"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        f"dynamic_non_background_bbox_crop_{transform}",
        num_channels,
        height,
        width,
    )


def build_dynamic_color_bbox_crop_model(
    selected_color: int,
    color_map: dict[int, int],
    output_path: str,
    transform: str = "identity",
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a dynamic crop of the bbox covering all cells of one color."""
    if selected_color < 0 or selected_color >= num_channels:
        raise ValueError(f"selected_color must be within 0..{num_channels - 1}")
    if transform not in {"identity", "mirror_horizontal", "mirror_vertical"}:
        raise ValueError("transform must be identity, mirror_horizontal, or mirror_vertical")

    full_map: list[int] = []
    for old_color in range(num_channels):
        new_color = int(color_map.get(old_color, old_color))
        if new_color < 0 or new_color >= num_channels:
            raise ValueError(f"mapped color {new_color} is outside 0..{num_channels - 1}")
        full_map.append(new_color)

    select_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    select_weights[0, selected_color, 0, 0] = 1.0
    color_weights = np.zeros((num_channels, num_channels, 1, 1), dtype=np.float32)
    for old_color, new_color in enumerate(full_map):
        color_weights[new_color, old_color, 0, 0] = 1.0

    offsets = np.arange(height, dtype=np.int64)
    reverse_indices = np.arange(height - 1, -1, -1, dtype=np.int64)
    one = np.array(1, dtype=np.int64)
    last_index = np.array(height - 1, dtype=np.int64)
    zero_threshold = np.array(0.0, dtype=np.float32)

    initializers = [
        numpy_helper.from_array(select_weights, name="SelectW"),
        numpy_helper.from_array(color_weights, name="ColorW"),
        numpy_helper.from_array(zero_threshold, name="Zero"),
        numpy_helper.from_array(offsets, name="Offsets"),
        numpy_helper.from_array(reverse_indices, name="ReverseIndices"),
        numpy_helper.from_array(one, name="One"),
        numpy_helper.from_array(last_index, name="LastIndex"),
    ]
    if transform == "mirror_vertical":
        row_source_nodes = [
            helper.make_node("Sub", ["bottom", "Offsets"], ["raw_rows"], name="raw_rows"),
            helper.make_node("Where", ["valid_rows", "raw_rows", "top"], ["source_rows"], name="source_rows"),
        ]
    else:
        row_source_nodes = [
            helper.make_node("Add", ["Offsets", "top"], ["raw_rows"], name="raw_rows"),
            helper.make_node("Where", ["valid_rows", "raw_rows", "top"], ["source_rows"], name="source_rows"),
        ]
    if transform == "mirror_horizontal":
        col_source_nodes = [
            helper.make_node("Sub", ["right", "Offsets"], ["raw_cols"], name="raw_cols"),
            helper.make_node("Where", ["valid_cols", "raw_cols", "left"], ["source_cols"], name="source_cols"),
        ]
    else:
        col_source_nodes = [
            helper.make_node("Add", ["Offsets", "left"], ["raw_cols"], name="raw_cols"),
            helper.make_node("Where", ["valid_cols", "raw_cols", "left"], ["source_cols"], name="source_cols"),
        ]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "SelectW"],
            ["selected_sum"],
            name="selected_sum",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Greater", ["selected_sum", "Zero"], ["selected_bool"], name="selected_bool"),
        _cast_to_float("selected_bool", "selected_float"),
        helper.make_node("ReduceSum", ["selected_float"], ["row_sum"], name="row_sum", axes=[3], keepdims=0),
        helper.make_node("ReduceSum", ["selected_float"], ["col_sum"], name="col_sum", axes=[2], keepdims=0),
        helper.make_node("Greater", ["row_sum", "Zero"], ["row_any"], name="row_any"),
        helper.make_node("Greater", ["col_sum", "Zero"], ["col_any"], name="col_any"),
        _cast_to_float("row_any", "row_any_float"),
        _cast_to_float("col_any", "col_any_float"),
        helper.make_node("ArgMax", ["row_any_float"], ["top_keep"], name="top_keep", axis=2, keepdims=0),
        helper.make_node("ArgMax", ["col_any_float"], ["left_keep"], name="left_keep", axis=2, keepdims=0),
        helper.make_node("Gather", ["row_any_float", "ReverseIndices"], ["row_any_reverse"], name="row_any_reverse", axis=2),
        helper.make_node("Gather", ["col_any_float", "ReverseIndices"], ["col_any_reverse"], name="col_any_reverse", axis=2),
        helper.make_node("ArgMax", ["row_any_reverse"], ["bottom_distance_keep"], name="bottom_distance_keep", axis=2, keepdims=0),
        helper.make_node("ArgMax", ["col_any_reverse"], ["right_distance_keep"], name="right_distance_keep", axis=2, keepdims=0),
        helper.make_node("Squeeze", ["top_keep"], ["top"], name="top", axes=[0, 1]),
        helper.make_node("Squeeze", ["left_keep"], ["left"], name="left", axes=[0, 1]),
        helper.make_node("Squeeze", ["bottom_distance_keep"], ["bottom_distance"], name="bottom_distance", axes=[0, 1]),
        helper.make_node("Squeeze", ["right_distance_keep"], ["right_distance"], name="right_distance", axes=[0, 1]),
        helper.make_node("Sub", ["LastIndex", "bottom_distance"], ["bottom"], name="bottom"),
        helper.make_node("Sub", ["LastIndex", "right_distance"], ["right"], name="right"),
        helper.make_node("Sub", ["bottom", "top"], ["bbox_height_minus_one"], name="bbox_height_minus_one"),
        helper.make_node("Sub", ["right", "left"], ["bbox_width_minus_one"], name="bbox_width_minus_one"),
        helper.make_node("Add", ["bbox_height_minus_one", "One"], ["bbox_height"], name="bbox_height"),
        helper.make_node("Add", ["bbox_width_minus_one", "One"], ["bbox_width"], name="bbox_width"),
        helper.make_node("Less", ["Offsets", "bbox_height"], ["valid_rows"], name="valid_rows"),
        helper.make_node("Less", ["Offsets", "bbox_width"], ["valid_cols"], name="valid_cols"),
        *row_source_nodes,
        *col_source_nodes,
        helper.make_node("Unsqueeze", ["valid_rows"], ["valid_rows_2d"], name="valid_rows_2d", axes=[1]),
        helper.make_node("Unsqueeze", ["valid_cols"], ["valid_cols_2d"], name="valid_cols_2d", axes=[0]),
        helper.make_node("And", ["valid_rows_2d", "valid_cols_2d"], ["active_bool"], name="active_bool"),
        helper.make_node("Gather", ["input", "source_rows"], ["gather_rows"], name="gather_rows", axis=2),
        helper.make_node("Gather", ["gather_rows", "source_cols"], ["remapped"], name="remapped", axis=3),
        helper.make_node(
            "Conv",
            ["remapped", "ColorW"],
            ["mapped"],
            name="mapped",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        _cast_to_float("active_bool", "active_float"),
        helper.make_node("Mul", ["mapped", "active_float"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        f"dynamic_color_bbox_crop_{transform}",
        num_channels,
        height,
        width,
    )


def build_dynamic_active_mirror_model(
    mode: str,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a mirror over the top-left active input rectangle."""
    if mode not in {"horizontal", "vertical"}:
        raise ValueError("mode must be 'horizontal' or 'vertical'")

    select_weights = np.ones((1, num_channels, 1, 1), dtype=np.float32)
    offsets = np.arange(height, dtype=np.int64)
    reverse_indices = np.arange(height - 1, -1, -1, dtype=np.int64)
    one = np.array(1, dtype=np.int64)
    last_index = np.array(height - 1, dtype=np.int64)
    zero_int = np.array(0, dtype=np.int64)
    zero_threshold = np.array(0.0, dtype=np.float32)

    initializers = [
        numpy_helper.from_array(select_weights, name="SelectW"),
        numpy_helper.from_array(zero_threshold, name="Zero"),
        numpy_helper.from_array(offsets, name="Offsets"),
        numpy_helper.from_array(reverse_indices, name="ReverseIndices"),
        numpy_helper.from_array(one, name="One"),
        numpy_helper.from_array(last_index, name="LastIndex"),
        numpy_helper.from_array(zero_int, name="ZeroIndex"),
    ]

    if mode == "horizontal":
        row_nodes = [
            helper.make_node("Where", ["valid_rows", "Offsets", "ZeroIndex"], ["source_rows"], name="source_rows")
        ]
        col_nodes = [
            helper.make_node("Sub", ["right", "Offsets"], ["raw_cols"], name="raw_cols"),
            helper.make_node("Where", ["valid_cols", "raw_cols", "ZeroIndex"], ["source_cols"], name="source_cols"),
        ]
    else:
        row_nodes = [
            helper.make_node("Sub", ["bottom", "Offsets"], ["raw_rows"], name="raw_rows"),
            helper.make_node("Where", ["valid_rows", "raw_rows", "ZeroIndex"], ["source_rows"], name="source_rows"),
        ]
        col_nodes = [
            helper.make_node("Where", ["valid_cols", "Offsets", "ZeroIndex"], ["source_cols"], name="source_cols")
        ]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "SelectW"],
            ["active_sum"],
            name="active_sum",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Greater", ["active_sum", "Zero"], ["active_bool"], name="active_bool"),
        _cast_to_float("active_bool", "active_float_full"),
        helper.make_node("ReduceSum", ["active_float_full"], ["row_sum"], name="row_sum", axes=[3], keepdims=0),
        helper.make_node("ReduceSum", ["active_float_full"], ["col_sum"], name="col_sum", axes=[2], keepdims=0),
        helper.make_node("Greater", ["row_sum", "Zero"], ["row_any"], name="row_any"),
        helper.make_node("Greater", ["col_sum", "Zero"], ["col_any"], name="col_any"),
        _cast_to_float("row_any", "row_any_float"),
        _cast_to_float("col_any", "col_any_float"),
        helper.make_node("Gather", ["row_any_float", "ReverseIndices"], ["row_any_reverse"], name="row_any_reverse", axis=2),
        helper.make_node("Gather", ["col_any_float", "ReverseIndices"], ["col_any_reverse"], name="col_any_reverse", axis=2),
        helper.make_node("ArgMax", ["row_any_reverse"], ["bottom_distance_keep"], name="bottom_distance_keep", axis=2, keepdims=0),
        helper.make_node("ArgMax", ["col_any_reverse"], ["right_distance_keep"], name="right_distance_keep", axis=2, keepdims=0),
        helper.make_node("Squeeze", ["bottom_distance_keep"], ["bottom_distance"], name="bottom_distance", axes=[0, 1]),
        helper.make_node("Squeeze", ["right_distance_keep"], ["right_distance"], name="right_distance", axes=[0, 1]),
        helper.make_node("Sub", ["LastIndex", "bottom_distance"], ["bottom"], name="bottom"),
        helper.make_node("Sub", ["LastIndex", "right_distance"], ["right"], name="right"),
        helper.make_node("Add", ["bottom", "One"], ["active_height"], name="active_height"),
        helper.make_node("Add", ["right", "One"], ["active_width"], name="active_width"),
        helper.make_node("Less", ["Offsets", "active_height"], ["valid_rows"], name="valid_rows"),
        helper.make_node("Less", ["Offsets", "active_width"], ["valid_cols"], name="valid_cols"),
        *row_nodes,
        *col_nodes,
        helper.make_node("Unsqueeze", ["valid_rows"], ["valid_rows_2d"], name="valid_rows_2d", axes=[1]),
        helper.make_node("Unsqueeze", ["valid_cols"], ["valid_cols_2d"], name="valid_cols_2d", axes=[0]),
        helper.make_node("And", ["valid_rows_2d", "valid_cols_2d"], ["output_active_bool"], name="output_active_bool"),
        helper.make_node("Gather", ["input", "source_rows"], ["gather_rows"], name="gather_rows", axis=2),
        helper.make_node("Gather", ["gather_rows", "source_cols"], ["mirrored"], name="mirrored", axis=3),
        _cast_to_float("output_active_bool", "output_active_float"),
        helper.make_node("Mul", ["mirrored", "output_active_float"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        f"dynamic_active_mirror_{mode}",
        num_channels,
        height,
        width,
    )


def build_dynamic_left_column_diagonal_bottom_fill_model(
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build task084-style left-column preserve plus anti-diagonal and bottom fill."""
    if height != width:
        raise ValueError("dynamic diagonal-bottom fill requires square static dimensions")
    select_weights = np.ones((1, num_channels, 1, 1), dtype=np.float32)
    offsets = np.arange(height, dtype=np.int64)
    reverse_indices = np.arange(height - 1, -1, -1, dtype=np.int64)
    one = np.array(1, dtype=np.int64)
    last_index = np.array(height - 1, dtype=np.int64)
    zero_int = np.array(0, dtype=np.int64)
    zero_threshold = np.array(0.0, dtype=np.float32)

    initializers = [
        numpy_helper.from_array(select_weights, name="SelectW"),
        numpy_helper.from_array(zero_threshold, name="Zero"),
        numpy_helper.from_array(offsets, name="Offsets"),
        numpy_helper.from_array(reverse_indices, name="ReverseIndices"),
        numpy_helper.from_array(one, name="One"),
        numpy_helper.from_array(last_index, name="LastIndex"),
        numpy_helper.from_array(zero_int, name="ZeroIndex"),
        numpy_helper.from_array(_one_hot_color(2, num_channels), name="DiagonalColor"),
        numpy_helper.from_array(_one_hot_color(4, num_channels), name="BottomColor"),
    ]

    nodes = [
        helper.make_node(
            "Conv",
            ["input", "SelectW"],
            ["active_sum"],
            name="active_sum",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Greater", ["active_sum", "Zero"], ["active_cell_bool"], name="active_cell_bool"),
        _cast_to_float("active_cell_bool", "active_cell_float"),
        helper.make_node("ReduceSum", ["active_cell_float"], ["row_sum"], name="row_sum", axes=[3], keepdims=0),
        helper.make_node("ReduceSum", ["active_cell_float"], ["col_sum"], name="col_sum", axes=[2], keepdims=0),
        helper.make_node("Greater", ["row_sum", "Zero"], ["row_any"], name="row_any"),
        helper.make_node("Greater", ["col_sum", "Zero"], ["col_any"], name="col_any"),
        _cast_to_float("row_any", "row_any_float"),
        _cast_to_float("col_any", "col_any_float"),
        helper.make_node("Gather", ["row_any_float", "ReverseIndices"], ["row_any_reverse"], name="row_any_reverse", axis=2),
        helper.make_node("Gather", ["col_any_float", "ReverseIndices"], ["col_any_reverse"], name="col_any_reverse", axis=2),
        helper.make_node("ArgMax", ["row_any_reverse"], ["bottom_distance_keep"], name="bottom_distance_keep", axis=2, keepdims=0),
        helper.make_node("ArgMax", ["col_any_reverse"], ["right_distance_keep"], name="right_distance_keep", axis=2, keepdims=0),
        helper.make_node("Squeeze", ["bottom_distance_keep"], ["bottom_distance"], name="bottom_distance", axes=[0, 1]),
        helper.make_node("Squeeze", ["right_distance_keep"], ["right_distance"], name="right_distance", axes=[0, 1]),
        helper.make_node("Sub", ["LastIndex", "bottom_distance"], ["bottom"], name="bottom"),
        helper.make_node("Sub", ["LastIndex", "right_distance"], ["right"], name="right"),
        helper.make_node("Add", ["bottom", "One"], ["active_height"], name="active_height"),
        helper.make_node("Add", ["right", "One"], ["active_width"], name="active_width"),
        helper.make_node("Less", ["Offsets", "active_height"], ["valid_rows"], name="valid_rows"),
        helper.make_node("Less", ["Offsets", "active_width"], ["valid_cols"], name="valid_cols"),
        helper.make_node("Unsqueeze", ["valid_rows"], ["valid_rows_2d"], name="valid_rows_2d", axes=[1]),
        helper.make_node("Unsqueeze", ["valid_cols"], ["valid_cols_2d"], name="valid_cols_2d", axes=[0]),
        helper.make_node("And", ["valid_rows_2d", "valid_cols_2d"], ["active_bool"], name="active_bool"),
        helper.make_node("Greater", ["Offsets", "ZeroIndex"], ["col_positive"], name="col_positive"),
        helper.make_node("Less", ["Offsets", "bottom"], ["row_before_bottom"], name="row_before_bottom"),
        helper.make_node("Equal", ["Offsets", "bottom"], ["row_is_bottom"], name="row_is_bottom"),
        helper.make_node("Sub", ["right", "Offsets"], ["anti_diag_cols"], name="anti_diag_cols"),
        helper.make_node("Unsqueeze", ["Offsets"], ["col_offsets_2d"], name="col_offsets_2d", axes=[0]),
        helper.make_node("Unsqueeze", ["anti_diag_cols"], ["anti_diag_cols_2d"], name="anti_diag_cols_2d", axes=[1]),
        helper.make_node("Unsqueeze", ["col_positive"], ["col_positive_2d"], name="col_positive_2d", axes=[0]),
        helper.make_node("Unsqueeze", ["row_before_bottom"], ["row_before_bottom_2d"], name="row_before_bottom_2d", axes=[1]),
        helper.make_node("Unsqueeze", ["row_is_bottom"], ["row_is_bottom_2d"], name="row_is_bottom_2d", axes=[1]),
        helper.make_node("Equal", ["col_offsets_2d", "anti_diag_cols_2d"], ["anti_diag_position"], name="anti_diag_position"),
        helper.make_node("And", ["anti_diag_position", "row_before_bottom_2d"], ["anti_diag_rows"], name="anti_diag_rows"),
        helper.make_node("And", ["anti_diag_rows", "col_positive_2d"], ["anti_diag_inside_cols"], name="anti_diag_inside_cols"),
        helper.make_node("And", ["anti_diag_inside_cols", "active_bool"], ["diagonal_bool"], name="diagonal_bool"),
        helper.make_node("And", ["row_is_bottom_2d", "col_positive_2d"], ["bottom_candidate"], name="bottom_candidate"),
        helper.make_node("And", ["bottom_candidate", "active_bool"], ["bottom_bool"], name="bottom_bool"),
        helper.make_node("Or", ["diagonal_bool", "bottom_bool"], ["draw_bool"], name="draw_bool"),
        helper.make_node("Not", ["draw_bool"], ["not_draw_bool"], name="not_draw_bool"),
        helper.make_node("And", ["active_bool", "not_draw_bool"], ["keep_bool"], name="keep_bool"),
        _cast_to_float("keep_bool", "keep_float"),
        _cast_to_float("diagonal_bool", "diagonal_float"),
        _cast_to_float("bottom_bool", "bottom_float"),
        helper.make_node("Mul", ["input", "keep_float"], ["kept_input"], name="kept_input"),
        helper.make_node("Mul", ["DiagonalColor", "diagonal_float"], ["diagonal_draw"], name="diagonal_draw"),
        helper.make_node("Mul", ["BottomColor", "bottom_float"], ["bottom_draw"], name="bottom_draw"),
        helper.make_node("Add", ["kept_input", "diagonal_draw"], ["with_diagonal"], name="with_diagonal"),
        helper.make_node("Add", ["with_diagonal", "bottom_draw"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "dynamic_left_column_diagonal_bottom_fill",
        num_channels,
        height,
        width,
    )


def _append_dynamic_column_mask_nodes(
    nodes: list[onnx.NodeProto],
    initializers: list[onnx.TensorProto],
    marker_col_name: str,
    offsets_name: str,
    prefix: str,
    deltas: Sequence[int],
) -> str:
    mask_names: list[str] = []
    for index, delta in enumerate(deltas):
        delta_name = f"{prefix}_Delta_{index}"
        target_name = f"{prefix}_Target_{index}"
        mask_name = f"{prefix}_Mask_{index}"
        initializers.append(numpy_helper.from_array(np.array(delta, dtype=np.int64), name=delta_name))
        nodes.append(helper.make_node("Add", [marker_col_name, delta_name], [target_name], name=target_name))
        nodes.append(helper.make_node("Equal", [offsets_name, target_name], [mask_name], name=mask_name))
        mask_names.append(mask_name)
    if not mask_names:
        raise ValueError("deltas must not be empty")
    accum = mask_names[0]
    for index, mask_name in enumerate(mask_names[1:], start=1):
        next_name = f"{prefix}_Any_{index}"
        nodes.append(helper.make_node("Or", [accum, mask_name], [next_name], name=next_name))
        accum = next_name
    return accum


def build_dynamic_bottom_marker_vertical_stripes_model(
    output_path: str,
    fill_color: int = 5,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build task200-style marker-color vertical stripes with color-5 connectors."""
    if fill_color < 0 or fill_color >= num_channels:
        raise ValueError(f"fill_color must be within 0..{num_channels - 1}")
    any_color_weights = np.ones((1, num_channels, 1, 1), dtype=np.float32)
    nonzero_weights = np.ones((1, num_channels, 1, 1), dtype=np.float32)
    nonzero_weights[:, 0, :, :] = 0.0
    offsets = np.arange(height, dtype=np.int64)
    reverse_indices = np.arange(height - 1, -1, -1, dtype=np.int64)
    one = np.array(1, dtype=np.int64)
    last_index = np.array(height - 1, dtype=np.int64)
    zero_int = np.array(0, dtype=np.int64)
    zero_threshold = np.array(0.0, dtype=np.float32)

    initializers = [
        numpy_helper.from_array(any_color_weights, name="AnyColorW"),
        numpy_helper.from_array(nonzero_weights, name="NonzeroW"),
        numpy_helper.from_array(zero_threshold, name="Zero"),
        numpy_helper.from_array(offsets, name="Offsets"),
        numpy_helper.from_array(reverse_indices, name="ReverseIndices"),
        numpy_helper.from_array(one, name="One"),
        numpy_helper.from_array(last_index, name="LastIndex"),
        numpy_helper.from_array(zero_int, name="ZeroIndex"),
        numpy_helper.from_array(_one_hot_color(fill_color, num_channels), name="FillColor"),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "Conv",
            ["input", "AnyColorW"],
            ["active_sum"],
            name="active_sum",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Greater", ["active_sum", "Zero"], ["active_cell_bool"], name="active_cell_bool"),
        _cast_to_float("active_cell_bool", "active_cell_float"),
        helper.make_node("ReduceSum", ["active_cell_float"], ["row_sum"], name="row_sum", axes=[3], keepdims=0),
        helper.make_node("ReduceSum", ["active_cell_float"], ["col_sum"], name="col_sum", axes=[2], keepdims=0),
        helper.make_node("Greater", ["row_sum", "Zero"], ["row_any"], name="row_any"),
        helper.make_node("Greater", ["col_sum", "Zero"], ["col_any"], name="col_any"),
        _cast_to_float("row_any", "row_any_float"),
        _cast_to_float("col_any", "col_any_float"),
        helper.make_node("Gather", ["row_any_float", "ReverseIndices"], ["row_any_reverse"], name="row_any_reverse", axis=2),
        helper.make_node("Gather", ["col_any_float", "ReverseIndices"], ["col_any_reverse"], name="col_any_reverse", axis=2),
        helper.make_node("ArgMax", ["row_any_reverse"], ["bottom_distance_keep"], name="bottom_distance_keep", axis=2, keepdims=0),
        helper.make_node("ArgMax", ["col_any_reverse"], ["right_distance_keep"], name="right_distance_keep", axis=2, keepdims=0),
        helper.make_node("Squeeze", ["bottom_distance_keep"], ["bottom_distance"], name="bottom_distance", axes=[0, 1]),
        helper.make_node("Squeeze", ["right_distance_keep"], ["right_distance"], name="right_distance", axes=[0, 1]),
        helper.make_node("Sub", ["LastIndex", "bottom_distance"], ["bottom"], name="bottom"),
        helper.make_node("Sub", ["LastIndex", "right_distance"], ["right"], name="right"),
        helper.make_node("Add", ["bottom", "One"], ["active_height"], name="active_height"),
        helper.make_node("Add", ["right", "One"], ["active_width"], name="active_width"),
        helper.make_node("Less", ["Offsets", "active_height"], ["valid_rows"], name="valid_rows"),
        helper.make_node("Less", ["Offsets", "active_width"], ["valid_cols"], name="valid_cols"),
        helper.make_node("Unsqueeze", ["valid_rows"], ["valid_rows_2d"], name="valid_rows_2d", axes=[1]),
        helper.make_node("Unsqueeze", ["valid_cols"], ["valid_cols_2d"], name="valid_cols_2d", axes=[0]),
        helper.make_node("And", ["valid_rows_2d", "valid_cols_2d"], ["active_bool"], name="active_bool"),
        helper.make_node(
            "Conv",
            ["input", "NonzeroW"],
            ["nonzero_sum"],
            name="nonzero_sum",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Greater", ["nonzero_sum", "Zero"], ["nonzero_bool"], name="nonzero_bool"),
        _cast_to_float("nonzero_bool", "nonzero_float"),
        helper.make_node("ReduceSum", ["nonzero_float"], ["nonzero_col_sum"], name="nonzero_col_sum", axes=[2], keepdims=0),
        helper.make_node("ArgMax", ["nonzero_col_sum"], ["marker_col_keep"], name="marker_col_keep", axis=2, keepdims=0),
        helper.make_node("Squeeze", ["marker_col_keep"], ["marker_col"], name="marker_col", axes=[0, 1]),
        helper.make_node("Mul", ["input", "nonzero_float"], ["marker_cells"], name="marker_cells"),
        helper.make_node("ReduceMax", ["marker_cells"], ["MarkerColor"], name="MarkerColor", axes=[2, 3], keepdims=1),
    ]

    vertical_col_bool = _append_dynamic_column_mask_nodes(
        nodes,
        initializers,
        "marker_col",
        "Offsets",
        "VerticalCols",
        range(0, width, 2),
    )
    top_connector_col_bool = _append_dynamic_column_mask_nodes(
        nodes,
        initializers,
        "marker_col",
        "Offsets",
        "TopConnectorCols",
        range(1, width, 4),
    )
    bottom_connector_col_bool = _append_dynamic_column_mask_nodes(
        nodes,
        initializers,
        "marker_col",
        "Offsets",
        "BottomConnectorCols",
        range(3, width, 4),
    )

    nodes.extend(
        [
            helper.make_node("Equal", ["Offsets", "ZeroIndex"], ["top_row"], name="top_row"),
            helper.make_node("Equal", ["Offsets", "bottom"], ["bottom_row"], name="bottom_row"),
            helper.make_node("Unsqueeze", [vertical_col_bool], ["vertical_cols_2d"], name="vertical_cols_2d", axes=[0]),
            helper.make_node("Unsqueeze", [top_connector_col_bool], ["top_connector_cols_2d"], name="top_connector_cols_2d", axes=[0]),
            helper.make_node(
                "Unsqueeze",
                [bottom_connector_col_bool],
                ["bottom_connector_cols_2d"],
                name="bottom_connector_cols_2d",
                axes=[0],
            ),
            helper.make_node("Unsqueeze", ["top_row"], ["top_row_2d"], name="top_row_2d", axes=[1]),
            helper.make_node("Unsqueeze", ["bottom_row"], ["bottom_row_2d"], name="bottom_row_2d", axes=[1]),
            helper.make_node("And", ["vertical_cols_2d", "active_bool"], ["vertical_bool"], name="vertical_bool"),
            helper.make_node("And", ["top_row_2d", "top_connector_cols_2d"], ["top_connector_raw"], name="top_connector_raw"),
            helper.make_node("And", ["top_connector_raw", "active_bool"], ["top_connector_bool"], name="top_connector_bool"),
            helper.make_node(
                "And",
                ["bottom_row_2d", "bottom_connector_cols_2d"],
                ["bottom_connector_raw"],
                name="bottom_connector_raw",
            ),
            helper.make_node(
                "And",
                ["bottom_connector_raw", "active_bool"],
                ["bottom_connector_bool"],
                name="bottom_connector_bool",
            ),
            helper.make_node("Or", ["top_connector_bool", "bottom_connector_bool"], ["connector_bool"], name="connector_bool"),
            helper.make_node("Or", ["vertical_bool", "connector_bool"], ["draw_bool"], name="draw_bool"),
            helper.make_node("Not", ["draw_bool"], ["not_draw_bool"], name="not_draw_bool"),
            helper.make_node("And", ["active_bool", "not_draw_bool"], ["keep_bool"], name="keep_bool"),
            _cast_to_float("keep_bool", "keep_float"),
            _cast_to_float("vertical_bool", "vertical_float"),
            _cast_to_float("connector_bool", "connector_float"),
            helper.make_node("Mul", ["input", "keep_float"], ["kept_input"], name="kept_input"),
            helper.make_node("Mul", ["MarkerColor", "vertical_float"], ["vertical_draw"], name="vertical_draw"),
            helper.make_node("Mul", ["FillColor", "connector_float"], ["connector_draw"], name="connector_draw"),
            helper.make_node("Add", ["kept_input", "vertical_draw"], ["with_vertical"], name="with_vertical"),
            helper.make_node("Add", ["with_vertical", "connector_draw"], ["output"], name="output"),
        ]
    )

    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "dynamic_bottom_marker_vertical_stripes",
        num_channels,
        height,
        width,
    )


def build_dynamic_line_projection_model(
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build same-shape projection of stray same-color cells next to full lines."""
    any_color_weights = np.ones((1, num_channels, 1, 1), dtype=np.float32)
    offsets = np.arange(height, dtype=np.int64)
    zero_float = np.array(0.0, dtype=np.float32)
    half_float = np.array(0.5, dtype=np.float32)
    one_int = np.array(1, dtype=np.int64)

    initializers = [
        numpy_helper.from_array(any_color_weights, name="AnyColorW"),
        numpy_helper.from_array(zero_float, name="Zero"),
        numpy_helper.from_array(offsets, name="Offsets"),
        numpy_helper.from_array(one_int, name="One"),
        numpy_helper.from_array(half_float, name="Half"),
        numpy_helper.from_array(_one_hot_color(0, num_channels), name="BackgroundColor"),
    ]

    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "Conv",
            ["input", "AnyColorW"],
            ["active_sum"],
            name="active_sum",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Greater", ["active_sum", "Zero"], ["active_cell_bool"], name="active_cell_bool"),
        _cast_to_float("active_cell_bool", "active_cell_float"),
        helper.make_node("ReduceSum", ["active_cell_float"], ["active_row_sum"], name="active_row_sum", axes=[3], keepdims=0),
        helper.make_node("ReduceSum", ["active_cell_float"], ["active_col_sum"], name="active_col_sum", axes=[2], keepdims=0),
        helper.make_node("Greater", ["active_row_sum", "Zero"], ["active_row_bool"], name="active_row_bool"),
        helper.make_node("Greater", ["active_col_sum", "Zero"], ["active_col_bool"], name="active_col_bool"),
        _cast_to_float("active_row_bool", "active_row_float"),
        _cast_to_float("active_col_bool", "active_col_float"),
        helper.make_node("ReduceSum", ["active_row_float"], ["active_height"], name="active_height", axes=[2], keepdims=0),
        helper.make_node("ReduceSum", ["active_col_float"], ["active_width"], name="active_width", axes=[2], keepdims=0),
    ]

    color_outputs: list[str] = []
    for color in range(1, num_channels):
        prefix = f"line_project_c{color}"
        select_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
        select_weights[0, color, 0, 0] = 1.0
        initializers.extend(
            [
                numpy_helper.from_array(select_weights, name=f"{prefix}_SelectW"),
                numpy_helper.from_array(_one_hot_color(color, num_channels), name=f"{prefix}_Color"),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    ["input", f"{prefix}_SelectW"],
                    [f"{prefix}_mask"],
                    name=f"{prefix}_mask",
                    kernel_shape=[1, 1],
                    strides=[1, 1],
                ),
                helper.make_node("ReduceSum", [f"{prefix}_mask"], [f"{prefix}_row_sum"], name=f"{prefix}_row_sum", axes=[3], keepdims=0),
                helper.make_node("ReduceSum", [f"{prefix}_mask"], [f"{prefix}_col_sum"], name=f"{prefix}_col_sum", axes=[2], keepdims=0),
                helper.make_node("Sub", [f"{prefix}_row_sum", "active_width"], [f"{prefix}_hline_diff_raw"], name=f"{prefix}_hline_diff_raw"),
                helper.make_node("Abs", [f"{prefix}_hline_diff_raw"], [f"{prefix}_hline_diff"], name=f"{prefix}_hline_diff"),
                helper.make_node("Less", [f"{prefix}_hline_diff", "Half"], [f"{prefix}_hline_bool"], name=f"{prefix}_hline_bool"),
                helper.make_node("Sub", [f"{prefix}_col_sum", "active_height"], [f"{prefix}_vline_diff_raw"], name=f"{prefix}_vline_diff_raw"),
                helper.make_node("Abs", [f"{prefix}_vline_diff_raw"], [f"{prefix}_vline_diff"], name=f"{prefix}_vline_diff"),
                helper.make_node("Less", [f"{prefix}_vline_diff", "Half"], [f"{prefix}_vline_bool"], name=f"{prefix}_vline_bool"),
                _cast_to_float(f"{prefix}_hline_bool", f"{prefix}_hline_float"),
                _cast_to_float(f"{prefix}_vline_bool", f"{prefix}_vline_float"),
                helper.make_node(
                    "ReduceSum",
                    [f"{prefix}_hline_float"],
                    [f"{prefix}_hline_count"],
                    name=f"{prefix}_hline_count",
                    axes=[2],
                    keepdims=0,
                ),
                helper.make_node(
                    "ReduceSum",
                    [f"{prefix}_vline_float"],
                    [f"{prefix}_vline_count"],
                    name=f"{prefix}_vline_count",
                    axes=[2],
                    keepdims=0,
                ),
                helper.make_node("Greater", [f"{prefix}_hline_count", "Zero"], [f"{prefix}_h_exists"], name=f"{prefix}_h_exists"),
                helper.make_node("Greater", [f"{prefix}_vline_count", "Zero"], [f"{prefix}_v_exists"], name=f"{prefix}_v_exists"),
                helper.make_node("ArgMax", [f"{prefix}_hline_float"], [f"{prefix}_hline_index_keep"], name=f"{prefix}_hline_index_keep", axis=2, keepdims=0),
                helper.make_node("ArgMax", [f"{prefix}_vline_float"], [f"{prefix}_vline_index_keep"], name=f"{prefix}_vline_index_keep", axis=2, keepdims=0),
                helper.make_node("Squeeze", [f"{prefix}_hline_index_keep"], [f"{prefix}_hline_index"], name=f"{prefix}_hline_index", axes=[0, 1]),
                helper.make_node("Squeeze", [f"{prefix}_vline_index_keep"], [f"{prefix}_vline_index"], name=f"{prefix}_vline_index", axes=[0, 1]),
                helper.make_node("Sub", [f"{prefix}_hline_index", "One"], [f"{prefix}_hline_above_index"], name=f"{prefix}_hline_above_index"),
                helper.make_node("Add", [f"{prefix}_hline_index", "One"], [f"{prefix}_hline_below_index"], name=f"{prefix}_hline_below_index"),
                helper.make_node("Sub", [f"{prefix}_vline_index", "One"], [f"{prefix}_vline_left_index"], name=f"{prefix}_vline_left_index"),
                helper.make_node("Add", [f"{prefix}_vline_index", "One"], [f"{prefix}_vline_right_index"], name=f"{prefix}_vline_right_index"),
                helper.make_node("Less", ["Offsets", f"{prefix}_hline_index"], [f"{prefix}_row_above_bool"], name=f"{prefix}_row_above_bool"),
                helper.make_node("Greater", ["Offsets", f"{prefix}_hline_index"], [f"{prefix}_row_below_bool"], name=f"{prefix}_row_below_bool"),
                helper.make_node("Less", ["Offsets", f"{prefix}_vline_index"], [f"{prefix}_col_left_bool"], name=f"{prefix}_col_left_bool"),
                helper.make_node("Greater", ["Offsets", f"{prefix}_vline_index"], [f"{prefix}_col_right_bool"], name=f"{prefix}_col_right_bool"),
                helper.make_node("Equal", ["Offsets", f"{prefix}_hline_above_index"], [f"{prefix}_target_above_row_bool"], name=f"{prefix}_target_above_row_bool"),
                helper.make_node("Equal", ["Offsets", f"{prefix}_hline_below_index"], [f"{prefix}_target_below_row_bool"], name=f"{prefix}_target_below_row_bool"),
                helper.make_node("Equal", ["Offsets", f"{prefix}_vline_left_index"], [f"{prefix}_target_left_col_bool"], name=f"{prefix}_target_left_col_bool"),
                helper.make_node("Equal", ["Offsets", f"{prefix}_vline_right_index"], [f"{prefix}_target_right_col_bool"], name=f"{prefix}_target_right_col_bool"),
                helper.make_node("Unsqueeze", [f"{prefix}_hline_bool"], [f"{prefix}_hline_rows"], name=f"{prefix}_hline_rows", axes=[3]),
                helper.make_node("Unsqueeze", [f"{prefix}_vline_bool"], [f"{prefix}_vline_cols"], name=f"{prefix}_vline_cols", axes=[2]),
                helper.make_node("And", [f"{prefix}_hline_rows", "active_cell_bool"], [f"{prefix}_hline_active"], name=f"{prefix}_hline_active"),
                helper.make_node("And", [f"{prefix}_hline_active", f"{prefix}_h_exists"], [f"{prefix}_hline_draw"], name=f"{prefix}_hline_draw"),
                helper.make_node("And", [f"{prefix}_vline_cols", "active_cell_bool"], [f"{prefix}_vline_active"], name=f"{prefix}_vline_active"),
                helper.make_node("And", [f"{prefix}_vline_active", f"{prefix}_v_exists"], [f"{prefix}_vline_draw"], name=f"{prefix}_vline_draw"),
                helper.make_node("Unsqueeze", [f"{prefix}_row_above_bool"], [f"{prefix}_row_above"], name=f"{prefix}_row_above", axes=[0, 1, 3]),
                helper.make_node("Unsqueeze", [f"{prefix}_row_below_bool"], [f"{prefix}_row_below"], name=f"{prefix}_row_below", axes=[0, 1, 3]),
                _cast_to_float(f"{prefix}_row_above", f"{prefix}_row_above_float"),
                _cast_to_float(f"{prefix}_row_below", f"{prefix}_row_below_float"),
                helper.make_node("Mul", [f"{prefix}_mask", f"{prefix}_row_above_float"], [f"{prefix}_above_cells"], name=f"{prefix}_above_cells"),
                helper.make_node("Mul", [f"{prefix}_mask", f"{prefix}_row_below_float"], [f"{prefix}_below_cells"], name=f"{prefix}_below_cells"),
                helper.make_node(
                    "ReduceSum",
                    [f"{prefix}_above_cells"],
                    [f"{prefix}_above_col_sum"],
                    name=f"{prefix}_above_col_sum",
                    axes=[2],
                    keepdims=0,
                ),
                helper.make_node(
                    "ReduceSum",
                    [f"{prefix}_below_cells"],
                    [f"{prefix}_below_col_sum"],
                    name=f"{prefix}_below_col_sum",
                    axes=[2],
                    keepdims=0,
                ),
                helper.make_node("Greater", [f"{prefix}_above_col_sum", "Zero"], [f"{prefix}_above_cols_bool"], name=f"{prefix}_above_cols_bool"),
                helper.make_node("Greater", [f"{prefix}_below_col_sum", "Zero"], [f"{prefix}_below_cols_bool"], name=f"{prefix}_below_cols_bool"),
                helper.make_node(
                    "Unsqueeze",
                    [f"{prefix}_target_above_row_bool"],
                    [f"{prefix}_target_above_row"],
                    name=f"{prefix}_target_above_row",
                    axes=[0, 1, 3],
                ),
                helper.make_node(
                    "Unsqueeze",
                    [f"{prefix}_target_below_row_bool"],
                    [f"{prefix}_target_below_row"],
                    name=f"{prefix}_target_below_row",
                    axes=[0, 1, 3],
                ),
                helper.make_node("Unsqueeze", [f"{prefix}_above_cols_bool"], [f"{prefix}_above_cols"], name=f"{prefix}_above_cols", axes=[2]),
                helper.make_node("Unsqueeze", [f"{prefix}_below_cols_bool"], [f"{prefix}_below_cols"], name=f"{prefix}_below_cols", axes=[2]),
                helper.make_node("And", [f"{prefix}_target_above_row", f"{prefix}_above_cols"], [f"{prefix}_hproj_above_raw"], name=f"{prefix}_hproj_above_raw"),
                helper.make_node("And", [f"{prefix}_target_below_row", f"{prefix}_below_cols"], [f"{prefix}_hproj_below_raw"], name=f"{prefix}_hproj_below_raw"),
                helper.make_node("And", [f"{prefix}_hproj_above_raw", "active_cell_bool"], [f"{prefix}_hproj_above_active"], name=f"{prefix}_hproj_above_active"),
                helper.make_node("And", [f"{prefix}_hproj_below_raw", "active_cell_bool"], [f"{prefix}_hproj_below_active"], name=f"{prefix}_hproj_below_active"),
                helper.make_node("And", [f"{prefix}_hproj_above_active", f"{prefix}_h_exists"], [f"{prefix}_hproj_above"], name=f"{prefix}_hproj_above"),
                helper.make_node("And", [f"{prefix}_hproj_below_active", f"{prefix}_h_exists"], [f"{prefix}_hproj_below"], name=f"{prefix}_hproj_below"),
                helper.make_node("Unsqueeze", [f"{prefix}_col_left_bool"], [f"{prefix}_col_left"], name=f"{prefix}_col_left", axes=[0, 1, 2]),
                helper.make_node("Unsqueeze", [f"{prefix}_col_right_bool"], [f"{prefix}_col_right"], name=f"{prefix}_col_right", axes=[0, 1, 2]),
                _cast_to_float(f"{prefix}_col_left", f"{prefix}_col_left_float"),
                _cast_to_float(f"{prefix}_col_right", f"{prefix}_col_right_float"),
                helper.make_node("Mul", [f"{prefix}_mask", f"{prefix}_col_left_float"], [f"{prefix}_left_cells"], name=f"{prefix}_left_cells"),
                helper.make_node("Mul", [f"{prefix}_mask", f"{prefix}_col_right_float"], [f"{prefix}_right_cells"], name=f"{prefix}_right_cells"),
                helper.make_node(
                    "ReduceSum",
                    [f"{prefix}_left_cells"],
                    [f"{prefix}_left_row_sum"],
                    name=f"{prefix}_left_row_sum",
                    axes=[3],
                    keepdims=0,
                ),
                helper.make_node(
                    "ReduceSum",
                    [f"{prefix}_right_cells"],
                    [f"{prefix}_right_row_sum"],
                    name=f"{prefix}_right_row_sum",
                    axes=[3],
                    keepdims=0,
                ),
                helper.make_node("Greater", [f"{prefix}_left_row_sum", "Zero"], [f"{prefix}_left_rows_bool"], name=f"{prefix}_left_rows_bool"),
                helper.make_node("Greater", [f"{prefix}_right_row_sum", "Zero"], [f"{prefix}_right_rows_bool"], name=f"{prefix}_right_rows_bool"),
                helper.make_node(
                    "Unsqueeze",
                    [f"{prefix}_target_left_col_bool"],
                    [f"{prefix}_target_left_col"],
                    name=f"{prefix}_target_left_col",
                    axes=[0, 1, 2],
                ),
                helper.make_node(
                    "Unsqueeze",
                    [f"{prefix}_target_right_col_bool"],
                    [f"{prefix}_target_right_col"],
                    name=f"{prefix}_target_right_col",
                    axes=[0, 1, 2],
                ),
                helper.make_node("Unsqueeze", [f"{prefix}_left_rows_bool"], [f"{prefix}_left_rows"], name=f"{prefix}_left_rows", axes=[3]),
                helper.make_node("Unsqueeze", [f"{prefix}_right_rows_bool"], [f"{prefix}_right_rows"], name=f"{prefix}_right_rows", axes=[3]),
                helper.make_node("And", [f"{prefix}_target_left_col", f"{prefix}_left_rows"], [f"{prefix}_vproj_left_raw"], name=f"{prefix}_vproj_left_raw"),
                helper.make_node("And", [f"{prefix}_target_right_col", f"{prefix}_right_rows"], [f"{prefix}_vproj_right_raw"], name=f"{prefix}_vproj_right_raw"),
                helper.make_node("And", [f"{prefix}_vproj_left_raw", "active_cell_bool"], [f"{prefix}_vproj_left_active"], name=f"{prefix}_vproj_left_active"),
                helper.make_node("And", [f"{prefix}_vproj_right_raw", "active_cell_bool"], [f"{prefix}_vproj_right_active"], name=f"{prefix}_vproj_right_active"),
                helper.make_node("And", [f"{prefix}_vproj_left_active", f"{prefix}_v_exists"], [f"{prefix}_vproj_left"], name=f"{prefix}_vproj_left"),
                helper.make_node("And", [f"{prefix}_vproj_right_active", f"{prefix}_v_exists"], [f"{prefix}_vproj_right"], name=f"{prefix}_vproj_right"),
                helper.make_node("Or", [f"{prefix}_hline_draw", f"{prefix}_hproj_above"], [f"{prefix}_hdraw_0"], name=f"{prefix}_hdraw_0"),
                helper.make_node("Or", [f"{prefix}_hdraw_0", f"{prefix}_hproj_below"], [f"{prefix}_hdraw"], name=f"{prefix}_hdraw"),
                helper.make_node("Or", [f"{prefix}_vline_draw", f"{prefix}_vproj_left"], [f"{prefix}_vdraw_0"], name=f"{prefix}_vdraw_0"),
                helper.make_node("Or", [f"{prefix}_vdraw_0", f"{prefix}_vproj_right"], [f"{prefix}_vdraw"], name=f"{prefix}_vdraw"),
                helper.make_node("Or", [f"{prefix}_hdraw", f"{prefix}_vdraw"], [f"{prefix}_draw_bool"], name=f"{prefix}_draw_bool"),
                _cast_to_float(f"{prefix}_draw_bool", f"{prefix}_draw_float"),
                helper.make_node("Mul", [f"{prefix}_draw_float", f"{prefix}_Color"], [f"{prefix}_output"], name=f"{prefix}_output"),
            ]
        )
        color_outputs.append(f"{prefix}_output")

    rolling = color_outputs[0]
    for index, color_output in enumerate(color_outputs[1:], start=1):
        next_name = f"line_project_color_sum_{index}"
        nodes.append(helper.make_node("Add", [rolling, color_output], [next_name], name=next_name))
        rolling = next_name
    nodes.extend(
        [
            helper.make_node("ReduceSum", [rolling], ["draw_any"], name="draw_any", axes=[1], keepdims=1),
            helper.make_node("Sub", ["active_cell_float", "draw_any"], ["background_mask"], name="background_mask"),
            helper.make_node("Mul", ["background_mask", "BackgroundColor"], ["background_output"], name="background_output"),
            helper.make_node("Add", [rolling, "background_output"], ["output"], name="output"),
        ]
    )
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "dynamic_line_projection",
        num_channels,
        height,
        width,
    )


def build_dynamic_rectangular_cavity_fill_model(
    output_path: str,
    background_color: int = 0,
    wall_color: int = 5,
    fill_color: int = 4,
    max_cavity_height: int = 10,
    max_cavity_width: int = 10,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a dynamic fill for rectangular background cavities bounded by wall color.

    The supported cavities are solid background rectangles with wall-colored top
    and bottom borders. Left/right borders may be wall-colored or the grid
    boundary. Boundary-side cavities require the horizontal wall to terminate at
    the side wall, which prevents filling exterior gaps next to longer bars.
    """
    if len({background_color, wall_color, fill_color}) != 3:
        raise ValueError("background, wall, and fill colors must be distinct")
    for color in (background_color, wall_color, fill_color):
        if color < 0 or color >= num_channels:
            raise ValueError(f"color {color} is outside 0..{num_channels - 1}")
    if max_cavity_height <= 0 or max_cavity_height > height - 2:
        raise ValueError("max_cavity_height must be within 1..height-2")
    if max_cavity_width <= 0 or max_cavity_width > width:
        raise ValueError("max_cavity_width must be within 1..width")

    select_weights = np.zeros((2, num_channels, 1, 1), dtype=np.float32)
    select_weights[0, background_color, 0, 0] = 1.0
    select_weights[1, wall_color, 0, 0] = 1.0
    background_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    background_weights[0, background_color, 0, 0] = 1.0

    initializers: list[onnx.TensorProto] = [
        numpy_helper.from_array(select_weights, name="RectCavitySelectBW"),
        numpy_helper.from_array(background_weights, name="RectCavityBackgroundW"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="RectCavityHalf"),
        numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="RectCavityOne"),
        numpy_helper.from_array(_one_hot_color(fill_color, num_channels), name="RectCavityFillColor"),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "Conv",
            ["input", "RectCavitySelectBW"],
            ["rect_cavity_bw"],
            name="rect_cavity_bw",
            kernel_shape=[1, 1],
        ),
        helper.make_node(
            "Conv",
            ["input", "RectCavityBackgroundW"],
            ["rect_cavity_background"],
            name="rect_cavity_background",
            kernel_shape=[1, 1],
        ),
    ]

    spread_outputs: list[str] = []
    spread_names: dict[tuple[int, int], str] = {}

    def spread_name(cavity_height: int, cavity_width: int) -> str:
        key = (cavity_height, cavity_width)
        existing = spread_names.get(key)
        if existing is not None:
            return existing
        name = f"RectCavitySpread_{cavity_height}_{cavity_width}"
        spread_names[key] = name
        initializers.append(
            numpy_helper.from_array(
                np.ones((1, 1, cavity_height, cavity_width), dtype=np.float32),
                name=name,
            )
        )
        return name

    def add_pattern(
        prefix: str,
        kernel: np.ndarray,
        required_count: int,
        match_pads: list[int],
        spread_pads: list[int],
        cavity_height: int,
        cavity_width: int,
    ) -> None:
        kernel_name = f"{prefix}_Kernel"
        required_name = f"{prefix}_Required"
        initializers.append(numpy_helper.from_array(kernel.astype(np.float32, copy=False), name=kernel_name))
        initializers.append(numpy_helper.from_array(np.array(float(required_count), dtype=np.float32), name=required_name))

        conv_attrs: dict[str, object] = {"name": f"{prefix}_sum", "kernel_shape": list(kernel.shape[2:])}
        if any(match_pads):
            conv_attrs["pads"] = match_pads
        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    ["rect_cavity_bw", kernel_name],
                    [f"{prefix}_sum"],
                    **conv_attrs,
                ),
                helper.make_node("Sub", [f"{prefix}_sum", required_name], [f"{prefix}_diff_raw"], name=f"{prefix}_diff_raw"),
                helper.make_node("Abs", [f"{prefix}_diff_raw"], [f"{prefix}_diff"], name=f"{prefix}_diff"),
                helper.make_node(
                    "Less",
                    [f"{prefix}_diff", "RectCavityHalf"],
                    [f"{prefix}_match_bool"],
                    name=f"{prefix}_match_bool",
                ),
                _cast_to_float(f"{prefix}_match_bool", f"{prefix}_match"),
                helper.make_node(
                    "Conv",
                    [f"{prefix}_match", spread_name(cavity_height, cavity_width)],
                    [f"{prefix}_spread"],
                    name=f"{prefix}_spread",
                    kernel_shape=[cavity_height, cavity_width],
                    pads=spread_pads,
                ),
            ]
        )
        spread_outputs.append(f"{prefix}_spread")

    for cavity_height in range(1, max_cavity_height + 1):
        for cavity_width in range(1, max_cavity_width + 1):
            positive_count = cavity_height * cavity_width + 2 * cavity_width + 2 * cavity_height
            full_kernel = np.zeros((1, 2, cavity_height + 2, cavity_width + 2), dtype=np.float32)
            full_kernel[0, 0, 1 : cavity_height + 1, 1 : cavity_width + 1] = 1.0
            full_kernel[0, 1, 0, 1 : cavity_width + 1] = 1.0
            full_kernel[0, 1, cavity_height + 1, 1 : cavity_width + 1] = 1.0
            full_kernel[0, 1, 1 : cavity_height + 1, 0] = 1.0
            full_kernel[0, 1, 1 : cavity_height + 1, cavity_width + 1] = 1.0
            add_pattern(
                f"rect_cavity_full_{cavity_height}_{cavity_width}",
                full_kernel,
                positive_count,
                [0, 0, 0, 0],
                [cavity_height, cavity_width, cavity_height, cavity_width],
                cavity_height,
                cavity_width,
            )

            boundary_positive_count = cavity_height * cavity_width + 2 * cavity_width + cavity_height

            left_kernel = np.zeros((1, 2, cavity_height + 2, cavity_width + 3), dtype=np.float32)
            left_kernel[0, :, :, 0] = -1.0
            left_kernel[0, 0, 1 : cavity_height + 1, 1 : cavity_width + 1] = 1.0
            left_kernel[0, 1, 0, 1 : cavity_width + 1] = 1.0
            left_kernel[0, 1, cavity_height + 1, 1 : cavity_width + 1] = 1.0
            left_kernel[0, 1, 1 : cavity_height + 1, cavity_width + 1] = 1.0
            left_kernel[0, 1, 0, cavity_width + 2] = -1.0
            left_kernel[0, 1, cavity_height + 1, cavity_width + 2] = -1.0
            add_pattern(
                f"rect_cavity_left_{cavity_height}_{cavity_width}",
                left_kernel,
                boundary_positive_count,
                [0, 1, 0, 0],
                [cavity_height, cavity_width - 1, cavity_height, cavity_width + 1],
                cavity_height,
                cavity_width,
            )

            right_kernel = np.zeros((1, 2, cavity_height + 2, cavity_width + 3), dtype=np.float32)
            right_kernel[0, 1, 0, 0] = -1.0
            right_kernel[0, 1, cavity_height + 1, 0] = -1.0
            right_kernel[0, 1, 1 : cavity_height + 1, 1] = 1.0
            right_kernel[0, 0, 1 : cavity_height + 1, 2 : cavity_width + 2] = 1.0
            right_kernel[0, 1, 0, 2 : cavity_width + 2] = 1.0
            right_kernel[0, 1, cavity_height + 1, 2 : cavity_width + 2] = 1.0
            right_kernel[0, :, :, cavity_width + 2] = -1.0
            add_pattern(
                f"rect_cavity_right_{cavity_height}_{cavity_width}",
                right_kernel,
                boundary_positive_count,
                [0, 1, 0, 1],
                [cavity_height, cavity_width, cavity_height, cavity_width - 1],
                cavity_height,
                cavity_width,
            )

    rolling = spread_outputs[0]
    for index, spread_output in enumerate(spread_outputs[1:], start=1):
        next_name = f"rect_cavity_spread_sum_{index}"
        nodes.append(helper.make_node("Add", [rolling, spread_output], [next_name], name=next_name))
        rolling = next_name

    nodes.extend(
        [
            helper.make_node("Clip", [rolling], ["rect_cavity_any"], name="rect_cavity_any", min=0.0, max=1.0),
            helper.make_node(
                "Mul",
                ["rect_cavity_any", "rect_cavity_background"],
                ["rect_cavity_fill_mask"],
                name="rect_cavity_fill_mask",
            ),
            helper.make_node(
                "Sub",
                ["RectCavityOne", "rect_cavity_fill_mask"],
                ["rect_cavity_keep_mask"],
                name="rect_cavity_keep_mask",
            ),
            helper.make_node("Mul", ["input", "rect_cavity_keep_mask"], ["rect_cavity_kept"], name="rect_cavity_kept"),
            helper.make_node(
                "Mul",
                ["RectCavityFillColor", "rect_cavity_fill_mask"],
                ["rect_cavity_filled"],
                name="rect_cavity_filled",
            ),
            helper.make_node("Add", ["rect_cavity_kept", "rect_cavity_filled"], ["output"], name="output"),
        ]
    )
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "dynamic_rectangular_cavity_fill",
        num_channels,
        height,
        width,
    )


def build_two_marker_horizontal_bands_model(
    output_path: str,
    active_height: int,
    active_width: int,
    top_marker_row: int,
    bottom_marker_row: int,
    background_color: int = 0,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build fixed-shape horizontal bands whose colors come from two marker rows."""
    if active_height <= 0 or active_height > height:
        raise ValueError("active_height must be within 1..height")
    if active_width <= 0 or active_width > width:
        raise ValueError("active_width must be within 1..width")
    if not (0 < top_marker_row < bottom_marker_row < active_height - 1):
        raise ValueError("marker rows must be interior rows with top < bottom")
    if background_color < 0 or background_color >= num_channels:
        raise ValueError(f"background_color must be within 0..{num_channels - 1}")

    split_row = (top_marker_row + bottom_marker_row) // 2
    non_background_mask = np.ones((1, num_channels, 1, 1), dtype=np.float32)
    non_background_mask[:, background_color, :, :] = 0.0
    top_marker_mask = np.zeros((1, 1, height, width), dtype=np.float32)
    top_marker_mask[:, :, top_marker_row, :active_width] = 1.0
    bottom_marker_mask = np.zeros((1, 1, height, width), dtype=np.float32)
    bottom_marker_mask[:, :, bottom_marker_row, :active_width] = 1.0
    top_draw_mask = np.zeros((1, 1, height, width), dtype=np.float32)
    bottom_draw_mask = np.zeros((1, 1, height, width), dtype=np.float32)
    active_mask = np.zeros((1, 1, height, width), dtype=np.float32)
    active_mask[:, :, :active_height, :active_width] = 1.0

    for row in range(0, split_row + 1):
        top_draw_mask[:, :, row, 0] = 1.0
        top_draw_mask[:, :, row, active_width - 1] = 1.0
    top_draw_mask[:, :, 0, :active_width] = 1.0
    top_draw_mask[:, :, top_marker_row, :active_width] = 1.0

    for row in range(split_row + 1, active_height):
        bottom_draw_mask[:, :, row, 0] = 1.0
        bottom_draw_mask[:, :, row, active_width - 1] = 1.0
    bottom_draw_mask[:, :, bottom_marker_row, :active_width] = 1.0
    bottom_draw_mask[:, :, active_height - 1, :active_width] = 1.0

    draw_mask = np.clip(top_draw_mask + bottom_draw_mask, 0.0, 1.0)
    background_mask = active_mask - draw_mask

    initializers = [
        numpy_helper.from_array(non_background_mask, name="BandNonBackgroundMask"),
        numpy_helper.from_array(top_marker_mask, name="BandTopMarkerMask"),
        numpy_helper.from_array(bottom_marker_mask, name="BandBottomMarkerMask"),
        numpy_helper.from_array(top_draw_mask, name="BandTopDrawMask"),
        numpy_helper.from_array(bottom_draw_mask, name="BandBottomDrawMask"),
        numpy_helper.from_array(background_mask, name="BandBackgroundMask"),
        numpy_helper.from_array(_one_hot_color(background_color, num_channels), name="BandBackgroundColor"),
    ]
    nodes = [
        helper.make_node("Mul", ["input", "BandNonBackgroundMask"], ["band_non_background"], name="band_non_background"),
        helper.make_node("Mul", ["band_non_background", "BandTopMarkerMask"], ["band_top_marker_cells"], name="band_top_marker_cells"),
        helper.make_node(
            "ReduceSum",
            ["band_top_marker_cells"],
            ["band_top_color"],
            name="band_top_color",
            axes=[2, 3],
            keepdims=1,
        ),
        helper.make_node(
            "Mul",
            ["band_non_background", "BandBottomMarkerMask"],
            ["band_bottom_marker_cells"],
            name="band_bottom_marker_cells",
        ),
        helper.make_node(
            "ReduceSum",
            ["band_bottom_marker_cells"],
            ["band_bottom_color"],
            name="band_bottom_color",
            axes=[2, 3],
            keepdims=1,
        ),
        helper.make_node("Mul", ["band_top_color", "BandTopDrawMask"], ["band_top_output"], name="band_top_output"),
        helper.make_node(
            "Mul",
            ["band_bottom_color", "BandBottomDrawMask"],
            ["band_bottom_output"],
            name="band_bottom_output",
        ),
        helper.make_node("Mul", ["BandBackgroundColor", "BandBackgroundMask"], ["band_background"], name="band_background"),
        helper.make_node("Add", ["band_top_output", "band_bottom_output"], ["band_foreground"], name="band_foreground"),
        helper.make_node("Add", ["band_foreground", "band_background"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "two_marker_horizontal_bands",
        num_channels,
        height,
        width,
    )


def build_dynamic_quadrant_panel_select_model(
    color_map: dict[int, int],
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a dynamic 2x2 center-cross panel selector.

    The selected panel is the quadrant whose pattern has a unique maximum
    summed difference from the other three quadrants. Supported active inputs
    are odd square grids up to 29x29 with a one-cell center separator.
    """
    full_map: list[int] = []
    for old_color in range(num_channels):
        new_color = int(color_map.get(old_color, old_color))
        if new_color < 0 or new_color >= num_channels:
            raise ValueError(f"mapped color {new_color} is outside 0..{num_channels - 1}")
        full_map.append(new_color)

    color_weights = np.zeros((num_channels, num_channels, 1, 1), dtype=np.float32)
    for old_color, new_color in enumerate(full_map):
        color_weights[new_color, old_color, 0, 0] = 1.0

    ones = np.ones((1, num_channels, 1, 1), dtype=np.float32)
    zero = np.array(0.0, dtype=np.float32)
    initializers = [
        numpy_helper.from_array(ones, name="ActiveW"),
        numpy_helper.from_array(zero, name="Zero"),
        numpy_helper.from_array(color_weights, name="ColorW"),
    ]
    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "Conv",
            ["input", "ActiveW"],
            ["active_sum"],
            name="active_sum",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Greater", ["active_sum", "Zero"], ["active_bool"], name="active_bool"),
        _cast_to_float("active_bool", "active_float"),
        helper.make_node("ReduceSum", ["active_float"], ["active_total"], name="active_total", axes=[1, 2, 3], keepdims=0),
    ]

    selected_by_size: list[str] = []
    max_panel_size = (min(height, width) - 1) // 2
    for panel_size in range(1, max_panel_size + 1):
        active_size = panel_size * 2 + 1
        low_name = f"Size{panel_size}Low"
        high_name = f"Size{panel_size}High"
        active_area = float(active_size * active_size)
        initializers.append(numpy_helper.from_array(np.array(active_area - 0.5, dtype=np.float32), name=low_name))
        initializers.append(numpy_helper.from_array(np.array(active_area + 0.5, dtype=np.float32), name=high_name))
        nodes.append(helper.make_node("Greater", ["active_total", low_name], [f"p{panel_size}_above_low"], name=f"p{panel_size}_above_low"))
        nodes.append(helper.make_node("Less", ["active_total", high_name], [f"p{panel_size}_below_high"], name=f"p{panel_size}_below_high"))
        nodes.append(
            helper.make_node(
                "And",
                [f"p{panel_size}_above_low", f"p{panel_size}_below_high"],
                [f"p{panel_size}_active_bool"],
                name=f"p{panel_size}_active_bool",
            )
        )
        nodes.append(_cast_to_float(f"p{panel_size}_active_bool", f"p{panel_size}_active"))

        mask = np.zeros((1, 1, height, width), dtype=np.float32)
        mask[:, :, :panel_size, :panel_size] = 1.0
        initializers.append(numpy_helper.from_array(mask, name=f"PanelMask{panel_size}"))

        panel_names: list[str] = []
        starts = [
            (0, 0),
            (0, panel_size + 1),
            (panel_size + 1, 0),
            (panel_size + 1, panel_size + 1),
        ]
        for panel_index, (row_start, col_start) in enumerate(starts):
            rows = np.zeros((height,), dtype=np.int64)
            cols = np.zeros((width,), dtype=np.int64)
            rows[:panel_size] = np.arange(row_start, row_start + panel_size, dtype=np.int64)
            cols[:panel_size] = np.arange(col_start, col_start + panel_size, dtype=np.int64)
            initializers.append(numpy_helper.from_array(rows, name=f"P{panel_size}Q{panel_index}Rows"))
            initializers.append(numpy_helper.from_array(cols, name=f"P{panel_size}Q{panel_index}Cols"))
            nodes.append(
                helper.make_node(
                    "Gather",
                    ["input", f"P{panel_size}Q{panel_index}Rows"],
                    [f"p{panel_size}_q{panel_index}_rows"],
                    name=f"p{panel_size}_q{panel_index}_rows",
                    axis=2,
                )
            )
            nodes.append(
                helper.make_node(
                    "Gather",
                    [f"p{panel_size}_q{panel_index}_rows", f"P{panel_size}Q{panel_index}Cols"],
                    [f"p{panel_size}_q{panel_index}_raw"],
                    name=f"p{panel_size}_q{panel_index}_raw",
                    axis=3,
                )
            )
            panel_name = f"p{panel_size}_q{panel_index}"
            nodes.append(
                helper.make_node(
                    "Mul",
                    [f"p{panel_size}_q{panel_index}_raw", f"PanelMask{panel_size}"],
                    [panel_name],
                    name=panel_name,
                )
            )
            panel_names.append(panel_name)

        pair_diffs: dict[tuple[int, int], str] = {}
        for left in range(4):
            for right in range(left + 1, 4):
                diff_name = f"p{panel_size}_d{left}{right}"
                nodes.append(
                    helper.make_node(
                        "Sub",
                        [panel_names[left], panel_names[right]],
                        [f"{diff_name}_raw"],
                        name=f"{diff_name}_raw",
                    )
                )
                nodes.append(helper.make_node("Abs", [f"{diff_name}_raw"], [f"{diff_name}_abs"], name=f"{diff_name}_abs"))
                nodes.append(
                    helper.make_node(
                        "ReduceSum",
                        [f"{diff_name}_abs"],
                        [diff_name],
                        name=diff_name,
                        axes=[1, 2, 3],
                        keepdims=0,
                    )
                )
                pair_diffs[(left, right)] = diff_name

        scores: list[str] = []
        for panel_index in range(4):
            pieces = [
                pair_diffs[(min(panel_index, other), max(panel_index, other))]
                for other in range(4)
                if other != panel_index
            ]
            score_name = f"p{panel_size}_score{panel_index}"
            nodes.append(helper.make_node("Add", [pieces[0], pieces[1]], [f"{score_name}_partial"], name=f"{score_name}_partial"))
            nodes.append(helper.make_node("Add", [f"{score_name}_partial", pieces[2]], [score_name], name=score_name))
            scores.append(score_name)

        selected_quadrants: list[str] = []
        for panel_index in range(4):
            comparisons: list[str] = []
            for other in range(4):
                if other == panel_index:
                    continue
                cmp_name = f"p{panel_size}_q{panel_index}_gt{other}"
                nodes.append(helper.make_node("Greater", [scores[panel_index], scores[other]], [cmp_name], name=cmp_name))
                comparisons.append(cmp_name)
            nodes.append(
                helper.make_node(
                    "And",
                    [comparisons[0], comparisons[1]],
                    [f"p{panel_size}_q{panel_index}_gt01"],
                    name=f"p{panel_size}_q{panel_index}_gt01",
                )
            )
            nodes.append(
                helper.make_node(
                    "And",
                    [f"p{panel_size}_q{panel_index}_gt01", comparisons[2]],
                    [f"p{panel_size}_q{panel_index}_selected_bool"],
                    name=f"p{panel_size}_q{panel_index}_selected_bool",
                )
            )
            nodes.append(_cast_to_float(f"p{panel_size}_q{panel_index}_selected_bool", f"p{panel_size}_q{panel_index}_selected"))
            selected_name = f"p{panel_size}_q{panel_index}_weighted"
            nodes.append(
                helper.make_node(
                    "Mul",
                    [panel_names[panel_index], f"p{panel_size}_q{panel_index}_selected"],
                    [selected_name],
                    name=selected_name,
                )
            )
            selected_quadrants.append(selected_name)

        nodes.append(helper.make_node("Add", [selected_quadrants[0], selected_quadrants[1]], [f"p{panel_size}_sum01"], name=f"p{panel_size}_sum01"))
        nodes.append(helper.make_node("Add", [selected_quadrants[2], selected_quadrants[3]], [f"p{panel_size}_sum23"], name=f"p{panel_size}_sum23"))
        nodes.append(helper.make_node("Add", [f"p{panel_size}_sum01", f"p{panel_size}_sum23"], [f"p{panel_size}_selected"], name=f"p{panel_size}_selected"))
        nodes.append(
            helper.make_node(
                "Mul",
                [f"p{panel_size}_selected", f"p{panel_size}_active"],
                [f"p{panel_size}_selected_active"],
                name=f"p{panel_size}_selected_active",
            )
        )
        selected_by_size.append(f"p{panel_size}_selected_active")

    rolling = selected_by_size[0]
    for index, selected_name in enumerate(selected_by_size[1:], start=2):
        output_name = f"panel_size_sum_{index}"
        nodes.append(helper.make_node("Add", [rolling, selected_name], [output_name], name=output_name))
        rolling = output_name

    nodes.append(
        helper.make_node(
            "Conv",
            [rolling, "ColorW"],
            ["output"],
            name="output",
            kernel_shape=[1, 1],
            strides=[1, 1],
        )
    )
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "dynamic_quadrant_panel_select",
        num_channels,
        height,
        width,
    )


def build_dynamic_frame_interior_crop_model(
    frame_color: int,
    color_map: dict[int, int],
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a dynamic crop of the interior of a color-specific rectangular frame."""
    if frame_color < 0 or frame_color >= num_channels:
        raise ValueError(f"frame_color must be within 0..{num_channels - 1}")

    full_map: list[int] = []
    for old_color in range(num_channels):
        new_color = int(color_map.get(old_color, old_color))
        if new_color < 0 or new_color >= num_channels:
            raise ValueError(f"mapped color {new_color} is outside 0..{num_channels - 1}")
        full_map.append(new_color)

    select_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    select_weights[0, frame_color, 0, 0] = 1.0
    color_weights = np.zeros((num_channels, num_channels, 1, 1), dtype=np.float32)
    for old_color, new_color in enumerate(full_map):
        color_weights[new_color, old_color, 0, 0] = 1.0

    offsets = np.arange(height, dtype=np.int64)
    reverse_indices = np.arange(height - 1, -1, -1, dtype=np.int64)
    one = np.array(1, dtype=np.int64)
    last_index = np.array(height - 1, dtype=np.int64)
    zero_threshold = np.array(0.0, dtype=np.float32)

    initializers = [
        numpy_helper.from_array(select_weights, name="SelectW"),
        numpy_helper.from_array(color_weights, name="ColorW"),
        numpy_helper.from_array(zero_threshold, name="Zero"),
        numpy_helper.from_array(offsets, name="Offsets"),
        numpy_helper.from_array(reverse_indices, name="ReverseIndices"),
        numpy_helper.from_array(one, name="One"),
        numpy_helper.from_array(last_index, name="LastIndex"),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "SelectW"],
            ["selected_sum"],
            name="selected_sum",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Greater", ["selected_sum", "Zero"], ["selected_bool"], name="selected_bool"),
        _cast_to_float("selected_bool", "selected_float"),
        helper.make_node("ReduceSum", ["selected_float"], ["row_sum"], name="row_sum", axes=[3], keepdims=0),
        helper.make_node("ReduceSum", ["selected_float"], ["col_sum"], name="col_sum", axes=[2], keepdims=0),
        helper.make_node("Greater", ["row_sum", "Zero"], ["row_any"], name="row_any"),
        helper.make_node("Greater", ["col_sum", "Zero"], ["col_any"], name="col_any"),
        _cast_to_float("row_any", "row_any_float"),
        _cast_to_float("col_any", "col_any_float"),
        helper.make_node("ArgMax", ["row_any_float"], ["top_keep"], name="top_keep", axis=2, keepdims=0),
        helper.make_node("ArgMax", ["col_any_float"], ["left_keep"], name="left_keep", axis=2, keepdims=0),
        helper.make_node("Gather", ["row_any_float", "ReverseIndices"], ["row_any_reverse"], name="row_any_reverse", axis=2),
        helper.make_node("Gather", ["col_any_float", "ReverseIndices"], ["col_any_reverse"], name="col_any_reverse", axis=2),
        helper.make_node("ArgMax", ["row_any_reverse"], ["bottom_distance_keep"], name="bottom_distance_keep", axis=2, keepdims=0),
        helper.make_node("ArgMax", ["col_any_reverse"], ["right_distance_keep"], name="right_distance_keep", axis=2, keepdims=0),
        helper.make_node("Squeeze", ["top_keep"], ["top"], name="top", axes=[0, 1]),
        helper.make_node("Squeeze", ["left_keep"], ["left"], name="left", axes=[0, 1]),
        helper.make_node("Squeeze", ["bottom_distance_keep"], ["bottom_distance"], name="bottom_distance", axes=[0, 1]),
        helper.make_node("Squeeze", ["right_distance_keep"], ["right_distance"], name="right_distance", axes=[0, 1]),
        helper.make_node("Sub", ["LastIndex", "bottom_distance"], ["bottom"], name="bottom"),
        helper.make_node("Sub", ["LastIndex", "right_distance"], ["right"], name="right"),
        helper.make_node("Sub", ["bottom", "top"], ["bbox_height_minus_one"], name="bbox_height_minus_one"),
        helper.make_node("Sub", ["right", "left"], ["bbox_width_minus_one"], name="bbox_width_minus_one"),
        helper.make_node("Sub", ["bbox_height_minus_one", "One"], ["interior_height"], name="interior_height"),
        helper.make_node("Sub", ["bbox_width_minus_one", "One"], ["interior_width"], name="interior_width"),
        helper.make_node("Add", ["top", "One"], ["interior_top"], name="interior_top"),
        helper.make_node("Add", ["left", "One"], ["interior_left"], name="interior_left"),
        helper.make_node("Less", ["Offsets", "interior_height"], ["valid_rows"], name="valid_rows"),
        helper.make_node("Less", ["Offsets", "interior_width"], ["valid_cols"], name="valid_cols"),
        helper.make_node("Add", ["Offsets", "interior_top"], ["raw_rows"], name="raw_rows"),
        helper.make_node("Add", ["Offsets", "interior_left"], ["raw_cols"], name="raw_cols"),
        helper.make_node("Where", ["valid_rows", "raw_rows", "interior_top"], ["source_rows"], name="source_rows"),
        helper.make_node("Where", ["valid_cols", "raw_cols", "interior_left"], ["source_cols"], name="source_cols"),
        helper.make_node("Unsqueeze", ["valid_rows"], ["valid_rows_2d"], name="valid_rows_2d", axes=[1]),
        helper.make_node("Unsqueeze", ["valid_cols"], ["valid_cols_2d"], name="valid_cols_2d", axes=[0]),
        helper.make_node("And", ["valid_rows_2d", "valid_cols_2d"], ["active_bool"], name="active_bool"),
        helper.make_node("Gather", ["input", "source_rows"], ["gather_rows"], name="gather_rows", axis=2),
        helper.make_node("Gather", ["gather_rows", "source_cols"], ["remapped"], name="remapped", axis=3),
        helper.make_node(
            "Conv",
            ["remapped", "ColorW"],
            ["mapped"],
            name="mapped",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        _cast_to_float("active_bool", "active_float"),
        helper.make_node("Mul", ["mapped", "active_float"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "dynamic_frame_interior_crop",
        num_channels,
        height,
        width,
    )


def build_dynamic_bbox_extreme_color_swap_model(
    background_color: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build bbox crop where the most and least frequent bbox colors are swapped."""
    if background_color < 0 or background_color >= num_channels:
        raise ValueError(f"background_color must be within 0..{num_channels - 1}")

    select_weights = np.ones((1, num_channels, 1, 1), dtype=np.float32)
    select_weights[0, background_color, 0, 0] = 0.0
    offsets = np.arange(height, dtype=np.int64)
    reverse_indices = np.arange(height - 1, -1, -1, dtype=np.int64)
    channel_indices = np.arange(num_channels, dtype=np.int64)
    one = np.array(1, dtype=np.int64)
    last_index = np.array(height - 1, dtype=np.int64)
    zero_threshold = np.array(0.0, dtype=np.float32)
    half_threshold = np.array(0.5, dtype=np.float32)
    large_penalty = np.array(10000.0, dtype=np.float32)

    initializers = [
        numpy_helper.from_array(select_weights, name="SelectW"),
        numpy_helper.from_array(zero_threshold, name="Zero"),
        numpy_helper.from_array(half_threshold, name="Half"),
        numpy_helper.from_array(large_penalty, name="LargePenalty"),
        numpy_helper.from_array(offsets, name="Offsets"),
        numpy_helper.from_array(reverse_indices, name="ReverseIndices"),
        numpy_helper.from_array(channel_indices, name="ChannelIndices"),
        numpy_helper.from_array(one, name="One"),
        numpy_helper.from_array(last_index, name="LastIndex"),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "SelectW"],
            ["selected_sum"],
            name="selected_sum",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Greater", ["selected_sum", "Zero"], ["selected_bool"], name="selected_bool"),
        _cast_to_float("selected_bool", "selected_float"),
        helper.make_node("ReduceSum", ["selected_float"], ["row_sum"], name="row_sum", axes=[3], keepdims=0),
        helper.make_node("ReduceSum", ["selected_float"], ["col_sum"], name="col_sum", axes=[2], keepdims=0),
        helper.make_node("Greater", ["row_sum", "Zero"], ["row_any"], name="row_any"),
        helper.make_node("Greater", ["col_sum", "Zero"], ["col_any"], name="col_any"),
        _cast_to_float("row_any", "row_any_float"),
        _cast_to_float("col_any", "col_any_float"),
        helper.make_node("ArgMax", ["row_any_float"], ["top_keep"], name="top_keep", axis=2, keepdims=0),
        helper.make_node("ArgMax", ["col_any_float"], ["left_keep"], name="left_keep", axis=2, keepdims=0),
        helper.make_node("Gather", ["row_any_float", "ReverseIndices"], ["row_any_reverse"], name="row_any_reverse", axis=2),
        helper.make_node("Gather", ["col_any_float", "ReverseIndices"], ["col_any_reverse"], name="col_any_reverse", axis=2),
        helper.make_node("ArgMax", ["row_any_reverse"], ["bottom_distance_keep"], name="bottom_distance_keep", axis=2, keepdims=0),
        helper.make_node("ArgMax", ["col_any_reverse"], ["right_distance_keep"], name="right_distance_keep", axis=2, keepdims=0),
        helper.make_node("Squeeze", ["top_keep"], ["top"], name="top", axes=[0, 1]),
        helper.make_node("Squeeze", ["left_keep"], ["left"], name="left", axes=[0, 1]),
        helper.make_node("Squeeze", ["bottom_distance_keep"], ["bottom_distance"], name="bottom_distance", axes=[0, 1]),
        helper.make_node("Squeeze", ["right_distance_keep"], ["right_distance"], name="right_distance", axes=[0, 1]),
        helper.make_node("Sub", ["LastIndex", "bottom_distance"], ["bottom"], name="bottom"),
        helper.make_node("Sub", ["LastIndex", "right_distance"], ["right"], name="right"),
        helper.make_node("Sub", ["bottom", "top"], ["bbox_height_minus_one"], name="bbox_height_minus_one"),
        helper.make_node("Sub", ["right", "left"], ["bbox_width_minus_one"], name="bbox_width_minus_one"),
        helper.make_node("Add", ["bbox_height_minus_one", "One"], ["bbox_height"], name="bbox_height"),
        helper.make_node("Add", ["bbox_width_minus_one", "One"], ["bbox_width"], name="bbox_width"),
        helper.make_node("Less", ["Offsets", "bbox_height"], ["valid_rows"], name="valid_rows"),
        helper.make_node("Less", ["Offsets", "bbox_width"], ["valid_cols"], name="valid_cols"),
        helper.make_node("Add", ["Offsets", "top"], ["raw_rows"], name="raw_rows"),
        helper.make_node("Add", ["Offsets", "left"], ["raw_cols"], name="raw_cols"),
        helper.make_node("Where", ["valid_rows", "raw_rows", "top"], ["source_rows"], name="source_rows"),
        helper.make_node("Where", ["valid_cols", "raw_cols", "left"], ["source_cols"], name="source_cols"),
        helper.make_node("Unsqueeze", ["valid_rows"], ["valid_rows_2d"], name="valid_rows_2d", axes=[1]),
        helper.make_node("Unsqueeze", ["valid_cols"], ["valid_cols_2d"], name="valid_cols_2d", axes=[0]),
        helper.make_node("And", ["valid_rows_2d", "valid_cols_2d"], ["active_bool"], name="active_bool"),
        helper.make_node("Gather", ["input", "source_rows"], ["gather_rows"], name="gather_rows", axis=2),
        helper.make_node("Gather", ["gather_rows", "source_cols"], ["remapped"], name="remapped", axis=3),
        _cast_to_float("active_bool", "active_float"),
        helper.make_node("Mul", ["remapped", "active_float"], ["remapped_active"], name="remapped_active"),
        helper.make_node("ReduceSum", ["remapped_active"], ["counts"], name="counts", axes=[2, 3], keepdims=0),
        helper.make_node("ArgMax", ["counts"], ["most_index"], name="most_index", axis=1, keepdims=0),
        helper.make_node("Less", ["counts", "Half"], ["zero_count_bool"], name="zero_count_bool"),
        _cast_to_float("zero_count_bool", "zero_count_float"),
        helper.make_node("Mul", ["zero_count_float", "LargePenalty"], ["zero_penalty"], name="zero_penalty"),
        helper.make_node("Add", ["counts", "zero_penalty"], ["penalized_counts"], name="penalized_counts"),
        helper.make_node("ArgMin", ["penalized_counts"], ["least_index"], name="least_index", axis=1, keepdims=0),
        helper.make_node("Equal", ["ChannelIndices", "most_index"], ["most_channel_bool"], name="most_channel_bool"),
        helper.make_node("Equal", ["ChannelIndices", "least_index"], ["least_channel_bool"], name="least_channel_bool"),
        _cast_to_float("most_channel_bool", "most_channel_float"),
        _cast_to_float("least_channel_bool", "least_channel_float"),
        helper.make_node("Unsqueeze", ["most_channel_float"], ["most_channel"], name="most_channel", axes=[0, 2, 3]),
        helper.make_node("Unsqueeze", ["least_channel_float"], ["least_channel"], name="least_channel", axes=[0, 2, 3]),
        helper.make_node("Mul", ["remapped_active", "most_channel"], ["most_cells"], name="most_cells"),
        helper.make_node("Mul", ["remapped_active", "least_channel"], ["least_cells"], name="least_cells"),
        helper.make_node("ReduceSum", ["most_cells"], ["most_cell_mask"], name="most_cell_mask", axes=[1], keepdims=1),
        helper.make_node("ReduceSum", ["least_cells"], ["least_cell_mask"], name="least_cell_mask", axes=[1], keepdims=1),
        helper.make_node("Mul", ["most_cell_mask", "least_channel"], ["most_to_least"], name="most_to_least"),
        helper.make_node("Mul", ["least_cell_mask", "most_channel"], ["least_to_most"], name="least_to_most"),
        helper.make_node("Add", ["most_to_least", "least_to_most"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "dynamic_bbox_extreme_color_swap",
        num_channels,
        height,
        width,
    )


def build_dynamic_largest_frame_recolor_crop_model(
    output_path: str,
    min_frame_size: int = 4,
    max_frame_size: int = 8,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Crop the largest dynamic frame and recolor its nonzero cells to the marker color."""
    if min_frame_size < 3:
        raise ValueError("min_frame_size must be at least 3")
    if max_frame_size < min_frame_size:
        raise ValueError("max_frame_size must be >= min_frame_size")
    if max_frame_size > min(height, width):
        raise ValueError("max_frame_size cannot exceed the model grid size")

    channel_mask = np.ones((1, num_channels), dtype=np.float32)
    channel_mask[0, 0] = 0.0
    nonzero_weights = np.ones((1, num_channels, 1, 1), dtype=np.float32)
    nonzero_weights[0, 0, 0, 0] = 0.0
    zero_channel = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    zero_channel[0, 0, 0, 0] = 1.0
    channel_indices = np.arange(num_channels, dtype=np.int64)
    offsets = np.arange(height, dtype=np.int64)

    initializers = [
        numpy_helper.from_array(channel_mask, name="ChannelMask"),
        numpy_helper.from_array(nonzero_weights, name="NonZeroW"),
        numpy_helper.from_array(zero_channel, name="ZeroChannel"),
        numpy_helper.from_array(channel_indices, name="ChannelIndices"),
        numpy_helper.from_array(offsets, name="Offsets"),
        numpy_helper.from_array(np.array([1, -1], dtype=np.int64), name="FlatShape"),
        numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="BestScore0"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="BestTop0"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="BestLeft0"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="BestHeight0"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), name="BestWidth0"),
        numpy_helper.from_array(np.array([0.5], dtype=np.float32), name="Half"),
        numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="OneFloat"),
        numpy_helper.from_array(np.array([10000.0], dtype=np.float32), name="LargePenalty"),
    ]

    nodes = [
        helper.make_node("ReduceSum", ["input"], ["color_counts"], name="color_counts", axes=[2, 3], keepdims=0),
        helper.make_node("Mul", ["color_counts", "ChannelMask"], ["nonzero_counts"], name="nonzero_counts"),
        helper.make_node("ArgMax", ["nonzero_counts"], ["source_index"], name="source_index", axis=1, keepdims=0),
        helper.make_node("Less", ["nonzero_counts", "Half"], ["absent_bool"], name="absent_bool"),
        helper.make_node("Equal", ["ChannelIndices", "source_index"], ["source_channel_bool"], name="source_channel_bool"),
        _cast_to_float("source_channel_bool", "source_channel_float"),
        _cast_to_float("absent_bool", "absent_float"),
        helper.make_node("Add", ["absent_float", "source_channel_float"], ["marker_penalty_mask"], name="marker_penalty_mask"),
        helper.make_node("Mul", ["marker_penalty_mask", "LargePenalty"], ["marker_penalty"], name="marker_penalty"),
        helper.make_node("Add", ["nonzero_counts", "marker_penalty"], ["marker_scores"], name="marker_scores"),
        helper.make_node("ArgMin", ["marker_scores"], ["marker_index"], name="marker_index", axis=1, keepdims=0),
        helper.make_node("Equal", ["ChannelIndices", "marker_index"], ["marker_channel_bool"], name="marker_channel_bool"),
        _cast_to_float("marker_channel_bool", "marker_channel_float"),
        helper.make_node("Unsqueeze", ["source_channel_float"], ["source_channel"], name="source_channel", axes=[0, 2, 3]),
        helper.make_node("Unsqueeze", ["marker_channel_float"], ["marker_channel"], name="marker_channel", axes=[0, 2, 3]),
        helper.make_node("Mul", ["input", "source_channel"], ["source_cells"], name="source_cells"),
        helper.make_node("ReduceSum", ["source_cells"], ["source_mask"], name="source_mask", axes=[1], keepdims=1),
    ]

    best_score = "BestScore0"
    best_top = "BestTop0"
    best_left = "BestLeft0"
    best_height = "BestHeight0"
    best_width = "BestWidth0"
    for frame_height in range(min_frame_size, max_frame_size + 1):
        for frame_width in range(min_frame_size, max_frame_size + 1):
            prefix = f"frame_{frame_height}_{frame_width}"
            kernel = np.zeros((1, 1, frame_height, frame_width), dtype=np.float32)
            kernel[:, :, 0, :] = 1.0
            kernel[:, :, -1, :] = 1.0
            kernel[:, :, :, 0] = 1.0
            kernel[:, :, :, -1] = 1.0
            border_count = float((2 * frame_height) + (2 * frame_width) - 4)
            output_width = width - frame_width + 1
            initializers.extend(
                [
                    numpy_helper.from_array(kernel, name=f"{prefix}_kernel"),
                    numpy_helper.from_array(np.array([border_count - 0.5], dtype=np.float32), name=f"{prefix}_threshold"),
                    numpy_helper.from_array(np.array([float(frame_height * frame_width)], dtype=np.float32), name=f"{prefix}_area"),
                    numpy_helper.from_array(np.array([output_width], dtype=np.int64), name=f"{prefix}_out_width"),
                    numpy_helper.from_array(np.array([frame_height], dtype=np.int64), name=f"{prefix}_height"),
                    numpy_helper.from_array(np.array([frame_width], dtype=np.int64), name=f"{prefix}_width"),
                ]
            )
            nodes.extend(
                [
                    helper.make_node(
                        "Conv",
                        ["source_mask", f"{prefix}_kernel"],
                        [f"{prefix}_border_sum"],
                        name=f"{prefix}_border_sum",
                        kernel_shape=[frame_height, frame_width],
                        strides=[1, 1],
                    ),
                    helper.make_node(
                        "Greater",
                        [f"{prefix}_border_sum", f"{prefix}_threshold"],
                        [f"{prefix}_valid_bool"],
                        name=f"{prefix}_valid_bool",
                    ),
                    _cast_to_float(f"{prefix}_valid_bool", f"{prefix}_valid"),
                    helper.make_node(
                        "Mul",
                        [f"{prefix}_valid", f"{prefix}_area"],
                        [f"{prefix}_score_map"],
                        name=f"{prefix}_score_map",
                    ),
                    helper.make_node(
                        "ReduceMax",
                        [f"{prefix}_score_map"],
                        [f"{prefix}_score_keep"],
                        name=f"{prefix}_score_keep",
                        axes=[2, 3],
                        keepdims=0,
                    ),
                    helper.make_node("Squeeze", [f"{prefix}_score_keep"], [f"{prefix}_score"], name=f"{prefix}_score", axes=[1]),
                    helper.make_node("Reshape", [f"{prefix}_score_map", "FlatShape"], [f"{prefix}_flat"], name=f"{prefix}_flat"),
                    helper.make_node("ArgMax", [f"{prefix}_flat"], [f"{prefix}_flat_index"], name=f"{prefix}_flat_index", axis=1, keepdims=0),
                    helper.make_node(
                        "Div",
                        [f"{prefix}_flat_index", f"{prefix}_out_width"],
                        [f"{prefix}_top"],
                        name=f"{prefix}_top",
                    ),
                    helper.make_node(
                        "Mod",
                        [f"{prefix}_flat_index", f"{prefix}_out_width"],
                        [f"{prefix}_left"],
                        name=f"{prefix}_left",
                    ),
                    helper.make_node("Greater", [f"{prefix}_score", best_score], [f"{prefix}_better"], name=f"{prefix}_better"),
                    helper.make_node("Where", [f"{prefix}_better", f"{prefix}_score", best_score], [f"{prefix}_best_score"], name=f"{prefix}_best_score"),
                    helper.make_node("Where", [f"{prefix}_better", f"{prefix}_top", best_top], [f"{prefix}_best_top"], name=f"{prefix}_best_top"),
                    helper.make_node("Where", [f"{prefix}_better", f"{prefix}_left", best_left], [f"{prefix}_best_left"], name=f"{prefix}_best_left"),
                    helper.make_node(
                        "Where",
                        [f"{prefix}_better", f"{prefix}_height", best_height],
                        [f"{prefix}_best_height"],
                        name=f"{prefix}_best_height",
                    ),
                    helper.make_node(
                        "Where",
                        [f"{prefix}_better", f"{prefix}_width", best_width],
                        [f"{prefix}_best_width"],
                        name=f"{prefix}_best_width",
                    ),
                ]
            )
            best_score = f"{prefix}_best_score"
            best_top = f"{prefix}_best_top"
            best_left = f"{prefix}_best_left"
            best_height = f"{prefix}_best_height"
            best_width = f"{prefix}_best_width"

    nodes.extend(
        [
            helper.make_node("Less", ["Offsets", best_height], ["valid_rows"], name="valid_rows"),
            helper.make_node("Less", ["Offsets", best_width], ["valid_cols"], name="valid_cols"),
            helper.make_node("Add", ["Offsets", best_top], ["raw_rows"], name="raw_rows"),
            helper.make_node("Add", ["Offsets", best_left], ["raw_cols"], name="raw_cols"),
            helper.make_node("Where", ["valid_rows", "raw_rows", best_top], ["source_rows"], name="source_rows"),
            helper.make_node("Where", ["valid_cols", "raw_cols", best_left], ["source_cols"], name="source_cols"),
            helper.make_node("Unsqueeze", ["valid_rows"], ["valid_rows_2d"], name="valid_rows_2d", axes=[1]),
            helper.make_node("Unsqueeze", ["valid_cols"], ["valid_cols_2d"], name="valid_cols_2d", axes=[0]),
            helper.make_node("And", ["valid_rows_2d", "valid_cols_2d"], ["active_bool"], name="active_bool"),
            helper.make_node("Gather", ["input", "source_rows"], ["gather_rows"], name="gather_rows", axis=2),
            helper.make_node("Gather", ["gather_rows", "source_cols"], ["crop"], name="crop", axis=3),
            helper.make_node(
                "Conv",
                ["crop", "NonZeroW"],
                ["crop_nonzero_sum"],
                name="crop_nonzero_sum",
                kernel_shape=[1, 1],
                strides=[1, 1],
            ),
            helper.make_node("Greater", ["crop_nonzero_sum", "Half"], ["crop_nonzero_bool"], name="crop_nonzero_bool"),
            _cast_to_float("crop_nonzero_bool", "crop_nonzero"),
            helper.make_node("Sub", ["OneFloat", "crop_nonzero"], ["crop_zero"], name="crop_zero"),
            helper.make_node("Mul", ["crop_nonzero", "marker_channel"], ["nonzero_recolored"], name="nonzero_recolored"),
            helper.make_node("Mul", ["crop_zero", "ZeroChannel"], ["zero_cells"], name="zero_cells"),
            helper.make_node("Add", ["nonzero_recolored", "zero_cells"], ["colored_crop"], name="colored_crop"),
            _cast_to_float("active_bool", "active_float"),
            helper.make_node("Mul", ["colored_crop", "active_float"], ["output"], name="output"),
        ]
    )
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "dynamic_largest_frame_recolor_crop",
        num_channels,
        height,
        width,
    )


def build_periodic_extension_color_map_model(
    period_y: int,
    period_x: int,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    color_map: dict[int, int],
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build output[r, c] = color_map(input[r % period_y, c % period_x])."""
    if period_y <= 0 or period_y > input_height:
        raise ValueError("period_y must be within 1..input_height")
    if period_x <= 0 or period_x > input_width:
        raise ValueError("period_x must be within 1..input_width")
    if input_height <= 0 or input_height > height:
        raise ValueError("input_height must be within 1..height")
    if input_width <= 0 or input_width > width:
        raise ValueError("input_width must be within 1..width")
    if output_height <= 0 or output_height > height:
        raise ValueError("output_height must be within 1..height")
    if output_width <= 0 or output_width > width:
        raise ValueError("output_width must be within 1..width")

    pad_row = input_height if input_height < height else 0
    pad_col = input_width if input_width < width else 0
    row_init = _padded_indices(
        [row % period_y for row in range(output_height)],
        height,
        height,
        pad_row,
    )
    col_init = _padded_indices(
        [col % period_x for col in range(output_width)],
        width,
        width,
        pad_col,
    )
    weights = np.zeros((num_channels, num_channels, 1, 1), dtype=np.float32)
    for old_color in range(num_channels):
        new_color = int(color_map.get(old_color, old_color))
        if new_color < 0 or new_color >= num_channels:
            raise ValueError(f"mapped color {new_color} is outside 0..{num_channels - 1}")
        weights[new_color, old_color, 0, 0] = 1.0
    active_mask = np.zeros((1, 1, height, width), dtype=np.float32)
    active_mask[:, :, :output_height, :output_width] = 1.0

    initializers = [
        numpy_helper.from_array(row_init, name="RowIndices"),
        numpy_helper.from_array(col_init, name="ColIndices"),
        numpy_helper.from_array(weights, name="W"),
        _bool_mask(active_mask, "ActiveMask"),
    ]
    nodes = [
        helper.make_node("Gather", ["input", "RowIndices"], ["gather_rows"], name="gather_rows", axis=2),
        helper.make_node("Gather", ["gather_rows", "ColIndices"], ["remapped"], name="remapped", axis=3),
        helper.make_node(
            "Conv",
            ["remapped", "W"],
            ["mapped"],
            name="mapped",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        _cast_to_float("ActiveMask", "ActiveMaskFloat"),
        helper.make_node("Mul", ["mapped", "ActiveMaskFloat"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "periodic_extension_color_map",
        num_channels,
        height,
        width,
    )


def build_auto_periodic_extension_color_map_model(
    axis: str,
    input_height: int,
    input_width: int,
    output_height: int,
    output_width: int,
    color_map: dict[int, int],
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a single-axis periodic extension with an input-inferred period.

    The graph tests all possible periods for the chosen axis and selects the
    first period that exactly explains the active input rectangle. This keeps
    the model static while allowing different cases to use different periods.
    """
    if axis not in {"row", "col"}:
        raise ValueError("axis must be 'row' or 'col'")
    if input_height <= 0 or input_height > height:
        raise ValueError("input_height must be within 1..height")
    if input_width <= 0 or input_width > width:
        raise ValueError("input_width must be within 1..width")
    if output_height <= 0 or output_height > height:
        raise ValueError("output_height must be within 1..height")
    if output_width <= 0 or output_width > width:
        raise ValueError("output_width must be within 1..width")
    if axis == "row" and output_width != input_width:
        raise ValueError("row auto-period extension requires output_width == input_width")
    if axis == "col" and output_height != input_height:
        raise ValueError("col auto-period extension requires output_height == input_height")

    max_period = input_height if axis == "row" else input_width
    pad_row = input_height if input_height < height else 0
    pad_col = input_width if input_width < width else 0

    weights = np.zeros((num_channels, num_channels, 1, 1), dtype=np.float32)
    for old_color in range(num_channels):
        new_color = int(color_map.get(old_color, old_color))
        if new_color < 0 or new_color >= num_channels:
            raise ValueError(f"mapped color {new_color} is outside 0..{num_channels - 1}")
        weights[new_color, old_color, 0, 0] = 1.0

    active_mask = np.zeros((1, 1, height, width), dtype=np.float32)
    active_mask[:, :, :output_height, :output_width] = 1.0
    scalar_shape = (1, 1, 1, 1)
    initializers: list[onnx.TensorProto] = [
        numpy_helper.from_array(weights, name="W"),
        _bool_mask(active_mask, "ActiveMask"),
        numpy_helper.from_array(np.ones(scalar_shape, dtype=np.float32), name="One"),
        numpy_helper.from_array(np.full(scalar_shape, 0.5, dtype=np.float32), name="Half"),
        numpy_helper.from_array(np.zeros((1, num_channels, height, width), dtype=np.float32), name="ZeroImage"),
    ]
    nodes: list[onnx.NodeProto] = []
    valid_names: list[str] = []
    candidate_names: list[str] = []

    for period in range(1, max_period + 1):
        if axis == "row":
            candidate_rows = [row % period for row in range(output_height)]
            candidate_cols = list(range(output_width))
        else:
            candidate_rows = list(range(output_height))
            candidate_cols = [col % period for col in range(output_width)]
        initializers.extend(
            [
                numpy_helper.from_array(
                    _padded_indices(candidate_rows, height, height, pad_row),
                    name=f"CandidateRowsP{period}",
                ),
                numpy_helper.from_array(
                    _padded_indices(candidate_cols, width, width, pad_col),
                    name=f"CandidateColsP{period}",
                ),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "Gather",
                    ["input", f"CandidateRowsP{period}"],
                    [f"candidate_rows_p{period}"],
                    name=f"candidate_rows_p{period}",
                    axis=2,
                ),
                helper.make_node(
                    "Gather",
                    [f"candidate_rows_p{period}", f"CandidateColsP{period}"],
                    [f"candidate_p{period}"],
                    name=f"candidate_p{period}",
                    axis=3,
                ),
            ]
        )
        candidate_names.append(f"candidate_p{period}")

        if period == max_period:
            valid_names.append("One")
            continue

        if axis == "row":
            compare_rows = list(range(period, input_height))
            base_rows = [row % period for row in compare_rows]
            compare_cols = list(range(input_width))
            if not compare_rows:
                valid_names.append("One")
                continue
            initializers.extend(
                [
                    numpy_helper.from_array(
                        _padded_indices(compare_rows, len(compare_rows), height),
                        name=f"CompareRowsP{period}",
                    ),
                    numpy_helper.from_array(
                        _padded_indices(base_rows, len(base_rows), height),
                        name=f"BaseRowsP{period}",
                    ),
                    numpy_helper.from_array(
                        _padded_indices(compare_cols, input_width, width),
                        name=f"CompareColsP{period}",
                    ),
                ]
            )
            nodes.extend(
                [
                    helper.make_node(
                        "Gather",
                        ["input", f"CompareRowsP{period}"],
                        [f"compare_rows_p{period}"],
                        name=f"compare_rows_p{period}",
                        axis=2,
                    ),
                    helper.make_node(
                        "Gather",
                        [f"compare_rows_p{period}", f"CompareColsP{period}"],
                        [f"compare_p{period}"],
                        name=f"compare_p{period}",
                        axis=3,
                    ),
                    helper.make_node(
                        "Gather",
                        ["input", f"BaseRowsP{period}"],
                        [f"base_rows_p{period}"],
                        name=f"base_rows_p{period}",
                        axis=2,
                    ),
                    helper.make_node(
                        "Gather",
                        [f"base_rows_p{period}", f"CompareColsP{period}"],
                        [f"base_p{period}"],
                        name=f"base_p{period}",
                        axis=3,
                    ),
                ]
            )
        else:
            compare_cols = list(range(period, input_width))
            base_cols = [col % period for col in compare_cols]
            compare_rows = list(range(input_height))
            if not compare_cols:
                valid_names.append("One")
                continue
            initializers.extend(
                [
                    numpy_helper.from_array(
                        _padded_indices(compare_rows, input_height, height),
                        name=f"CompareRowsP{period}",
                    ),
                    numpy_helper.from_array(
                        _padded_indices(compare_cols, len(compare_cols), width),
                        name=f"CompareColsP{period}",
                    ),
                    numpy_helper.from_array(
                        _padded_indices(base_cols, len(base_cols), width),
                        name=f"BaseColsP{period}",
                    ),
                ]
            )
            nodes.extend(
                [
                    helper.make_node(
                        "Gather",
                        ["input", f"CompareRowsP{period}"],
                        [f"compare_rows_p{period}"],
                        name=f"compare_rows_p{period}",
                        axis=2,
                    ),
                    helper.make_node(
                        "Gather",
                        [f"compare_rows_p{period}", f"CompareColsP{period}"],
                        [f"compare_p{period}"],
                        name=f"compare_p{period}",
                        axis=3,
                    ),
                    helper.make_node(
                        "Gather",
                        ["input", f"CompareRowsP{period}"],
                        [f"base_rows_p{period}"],
                        name=f"base_rows_p{period}",
                        axis=2,
                    ),
                    helper.make_node(
                        "Gather",
                        [f"base_rows_p{period}", f"BaseColsP{period}"],
                        [f"base_p{period}"],
                        name=f"base_p{period}",
                        axis=3,
                    ),
                ]
            )

        nodes.extend(
            [
                helper.make_node(
                    "Sub",
                    [f"compare_p{period}", f"base_p{period}"],
                    [f"diff_p{period}"],
                    name=f"diff_p{period}",
                ),
                helper.make_node("Abs", [f"diff_p{period}"], [f"abs_p{period}"], name=f"abs_p{period}"),
                helper.make_node(
                    "ReduceSum",
                    [f"abs_p{period}"],
                    [f"score_p{period}"],
                    name=f"score_p{period}",
                    axes=[1, 2, 3],
                    keepdims=1,
                ),
                helper.make_node(
                    "Less",
                    [f"score_p{period}", "Half"],
                    [f"valid_bool_p{period}"],
                    name=f"valid_bool_p{period}",
                ),
                _cast_to_float(f"valid_bool_p{period}", f"valid_p{period}"),
            ]
        )
        valid_names.append(f"valid_p{period}")

    selected_terms: list[str] = []
    unresolved = "One"
    for period, valid_name in enumerate(valid_names, start=1):
        if period == 1:
            select_name = valid_name
        else:
            select_name = f"select_p{period}"
            nodes.append(
                helper.make_node(
                    "Mul",
                    [valid_name, unresolved],
                    [select_name],
                    name=select_name,
                )
            )
        term_name = f"selected_candidate_p{period}"
        nodes.append(
            helper.make_node(
                "Mul",
                [candidate_names[period - 1], select_name],
                [term_name],
                name=term_name,
            )
        )
        selected_terms.append(term_name)

        if period < max_period:
            not_valid = f"not_valid_p{period}"
            next_unresolved = f"unresolved_p{period}"
            nodes.extend(
                [
                    helper.make_node("Sub", ["One", valid_name], [not_valid], name=not_valid),
                    helper.make_node(
                        "Mul",
                        [unresolved, not_valid],
                        [next_unresolved],
                        name=next_unresolved,
                    ),
                ]
            )
            unresolved = next_unresolved

    accum = "ZeroImage"
    for index, term_name in enumerate(selected_terms, start=1):
        next_accum = f"periodic_mix_{index}"
        nodes.append(helper.make_node("Add", [accum, term_name], [next_accum], name=next_accum))
        accum = next_accum

    nodes.extend(
        [
            helper.make_node(
                "Conv",
                [accum, "W"],
                ["mapped"],
                name="mapped",
                kernel_shape=[1, 1],
                strides=[1, 1],
            ),
            _cast_to_float("ActiveMask", "ActiveMaskFloat"),
            helper.make_node("Mul", ["mapped", "ActiveMaskFloat"], ["output"], name="output"),
        ]
    )
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "auto_periodic_extension_color_map",
        num_channels,
        height,
        width,
    )


def _make_active_mask(active_height: int, active_width: int, height: int, width: int) -> np.ndarray:
    if active_height <= 0 or active_height > height:
        raise ValueError("active_height must be within 1..height")
    if active_width <= 0 or active_width > width:
        raise ValueError("active_width must be within 1..width")
    mask = np.zeros((1, 1, height, width), dtype=np.float32)
    mask[:, :, :active_height, :active_width] = 1.0
    return mask


def _one_hot_color(color: int, num_channels: int) -> np.ndarray:
    if color < 0 or color >= num_channels:
        raise ValueError(f"color {color} is outside 0..{num_channels - 1}")
    value = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    value[:, color, :, :] = 1.0
    return value


def build_panel_binary_op_model(
    orientation: str,
    operation: str,
    panel_height: int,
    panel_width: int,
    input_false_color: int,
    true_color: int,
    false_color: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a two-panel binary operation model with a one-cell separator."""
    operations = {
        "AND",
        "OR",
        "XOR",
        "LEFT_MINUS_RIGHT",
        "RIGHT_MINUS_LEFT",
        "EQUAL",
        "NOT_EQUAL",
    }
    if orientation not in {"vertical", "horizontal"}:
        raise ValueError("orientation must be vertical or horizontal")
    if operation not in operations:
        raise ValueError(f"unsupported panel operation: {operation}")
    if panel_height <= 0 or panel_height > height:
        raise ValueError("panel_height must be within 1..height")
    if panel_width <= 0 or panel_width > width:
        raise ValueError("panel_width must be within 1..width")

    if orientation == "vertical":
        input_height = panel_height
        input_width = panel_width * 2 + 1
        left_rows = list(range(panel_height))
        left_cols = list(range(panel_width))
        right_rows = list(range(panel_height))
        right_cols = list(range(panel_width + 1, panel_width * 2 + 1))
    else:
        input_height = panel_height * 2 + 1
        input_width = panel_width
        left_rows = list(range(panel_height))
        left_cols = list(range(panel_width))
        right_rows = list(range(panel_height + 1, panel_height * 2 + 1))
        right_cols = list(range(panel_width))
    if input_height > height or input_width > width:
        raise ValueError("panel layout exceeds static input shape")

    pad_row = input_height if input_height < height else 0
    pad_col = input_width if input_width < width else 0
    bool_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    for color in range(num_channels):
        if color != input_false_color:
            bool_weights[0, color, 0, 0] = 1.0

    initializers = [
        numpy_helper.from_array(_padded_indices(left_rows, height, height, pad_row), name="LeftRows"),
        numpy_helper.from_array(_padded_indices(left_cols, width, width, pad_col), name="LeftCols"),
        numpy_helper.from_array(_padded_indices(right_rows, height, height, pad_row), name="RightRows"),
        numpy_helper.from_array(_padded_indices(right_cols, width, width, pad_col), name="RightCols"),
        numpy_helper.from_array(bool_weights, name="BoolW"),
        _bool_mask(_make_active_mask(panel_height, panel_width, height, width), "ActiveMask"),
        numpy_helper.from_array(_one_hot_color(true_color, num_channels), name="TrueColor"),
        numpy_helper.from_array(_one_hot_color(false_color, num_channels), name="FalseColor"),
    ]
    nodes = [
        _cast_to_float("ActiveMask", "ActiveMaskFloat"),
        helper.make_node("Gather", ["input", "LeftRows"], ["left_rows"], name="left_rows", axis=2),
        helper.make_node("Gather", ["left_rows", "LeftCols"], ["left_panel"], name="left_panel", axis=3),
        helper.make_node("Gather", ["input", "RightRows"], ["right_rows"], name="right_rows", axis=2),
        helper.make_node("Gather", ["right_rows", "RightCols"], ["right_panel"], name="right_panel", axis=3),
        helper.make_node("Conv", ["left_panel", "BoolW"], ["left_mask"], name="left_mask", kernel_shape=[1, 1]),
        helper.make_node("Conv", ["right_panel", "BoolW"], ["right_mask"], name="right_mask", kernel_shape=[1, 1]),
    ]

    def add_or(output_name: str) -> None:
        nodes.append(helper.make_node("Add", ["left_mask", "right_mask"], ["or_sum"], name="or_sum"))
        nodes.append(helper.make_node("Clip", ["or_sum"], [output_name], name=output_name, min=0.0, max=1.0))

    def add_xor(output_name: str) -> None:
        initializers.append(numpy_helper.from_array(np.full((1, 1, 1, 1), 2.0, dtype=np.float32), name="Two"))
        nodes.append(helper.make_node("Add", ["left_mask", "right_mask"], ["xor_sum"], name="xor_sum"))
        nodes.append(helper.make_node("Mul", ["left_mask", "right_mask"], ["xor_both"], name="xor_both"))
        nodes.append(helper.make_node("Mul", ["xor_both", "Two"], ["xor_twice"], name="xor_twice"))
        nodes.append(helper.make_node("Sub", ["xor_sum", "xor_twice"], [output_name], name=output_name))

    def add_one() -> None:
        if not any(initializer.name == "One" for initializer in initializers):
            initializers.append(numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="One"))

    if operation == "AND":
        nodes.append(helper.make_node("Mul", ["left_mask", "right_mask"], ["raw_mask"], name="raw_mask"))
    elif operation == "OR":
        add_or("raw_mask")
    elif operation in {"XOR", "NOT_EQUAL"}:
        add_xor("raw_mask")
    elif operation == "LEFT_MINUS_RIGHT":
        add_one()
        nodes.append(helper.make_node("Sub", ["One", "right_mask"], ["not_right"], name="not_right"))
        nodes.append(helper.make_node("Mul", ["left_mask", "not_right"], ["raw_mask"], name="raw_mask"))
    elif operation == "RIGHT_MINUS_LEFT":
        add_one()
        nodes.append(helper.make_node("Sub", ["One", "left_mask"], ["not_left"], name="not_left"))
        nodes.append(helper.make_node("Mul", ["right_mask", "not_left"], ["raw_mask"], name="raw_mask"))
    else:
        add_xor("not_equal_mask")
        add_one()
        nodes.append(helper.make_node("Sub", ["One", "not_equal_mask"], ["raw_mask"], name="raw_mask"))

    nodes.extend(
        [
            helper.make_node("Mul", ["raw_mask", "ActiveMaskFloat"], ["true_mask"], name="true_mask"),
            helper.make_node("Sub", ["ActiveMaskFloat", "true_mask"], ["false_mask"], name="false_mask"),
            helper.make_node("Mul", ["TrueColor", "true_mask"], ["true_part"], name="true_part"),
            helper.make_node("Mul", ["FalseColor", "false_mask"], ["false_part"], name="false_part"),
            helper.make_node("Add", ["true_part", "false_part"], ["output"], name="output"),
        ]
    )
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "panel_binary_op",
        num_channels,
        height,
        width,
    )


def build_generalized_panel_op_model(
    panel_specs: list[dict[str, int]],
    operation: str,
    input_false_color: int,
    true_color: int,
    false_color: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build boolean operations over arbitrary fixed panel rectangles."""
    if not panel_specs:
        raise ValueError("panel_specs must be non-empty")
    panel_height = int(panel_specs[0]["height"])
    panel_width = int(panel_specs[0]["width"])
    if panel_height <= 0 or panel_height > height or panel_width <= 0 or panel_width > width:
        raise ValueError("panel shape must fit static output shape")
    for spec in panel_specs:
        if int(spec["height"]) != panel_height or int(spec["width"]) != panel_width:
            raise ValueError("all panel specs must have the same shape")
        top = int(spec["top"])
        left = int(spec["left"])
        if top < 0 or left < 0 or top + panel_height > height or left + panel_width > width:
            raise ValueError("panel spec exceeds static input shape")

    operation = {
        "A-B": "LEFT_MINUS_RIGHT",
        "B-A": "RIGHT_MINUS_LEFT",
        "UNION": "OR",
        "INTERSECTION": "AND",
    }.get(operation, operation)
    supported = {
        "AND",
        "OR",
        "XOR",
        "LEFT_MINUS_RIGHT",
        "RIGHT_MINUS_LEFT",
        "EQUAL",
        "NOT_EQUAL",
        "MAJORITY",
    }
    if operation not in supported:
        raise ValueError(f"unsupported panel operation: {operation}")
    if operation in {"LEFT_MINUS_RIGHT", "RIGHT_MINUS_LEFT", "EQUAL", "NOT_EQUAL"} and len(panel_specs) != 2:
        raise ValueError(f"{operation} requires exactly two panels")

    active_mask = _make_active_mask(panel_height, panel_width, height, width)
    bool_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    for color in range(num_channels):
        if color != input_false_color:
            bool_weights[0, color, 0, 0] = 1.0

    initializers = [
        numpy_helper.from_array(np.asarray([1, num_channels, height * width], dtype=np.int64), name="FlatShape"),
        numpy_helper.from_array(np.asarray([1, num_channels, height, width], dtype=np.int64), name="GridShape"),
        numpy_helper.from_array(bool_weights, name="BoolW"),
        _bool_mask(active_mask, "ActiveMask"),
        numpy_helper.from_array(_one_hot_color(true_color, num_channels), name="TrueColor"),
        numpy_helper.from_array(_one_hot_color(false_color, num_channels), name="FalseColor"),
    ]
    nodes = [
        _cast_to_float("ActiveMask", "ActiveMaskFloat"),
        helper.make_node("Reshape", ["input", "FlatShape"], ["flat_input"], name="flat_input"),
    ]
    mask_names = []
    base_rows = np.zeros((height, width), dtype=np.int32)
    base_cols = np.zeros((height, width), dtype=np.int32)
    for panel_index, spec in enumerate(panel_specs):
        rows = base_rows.copy()
        cols = base_cols.copy()
        top = int(spec["top"])
        left = int(spec["left"])
        for row in range(panel_height):
            for col in range(panel_width):
                rows[row, col] = top + row
                cols[row, col] = left + col
        flat_indices = (rows * width + cols).reshape(height * width).astype(np.int32)
        index_name = f"Panel{panel_index}Indices"
        flat_name = f"panel{panel_index}_flat"
        panel_name = f"panel{panel_index}"
        mask_name = f"panel{panel_index}_mask"
        initializers.append(numpy_helper.from_array(flat_indices, name=index_name))
        nodes.extend(
            [
                helper.make_node("Gather", ["flat_input", index_name], [flat_name], name=flat_name, axis=2),
                helper.make_node("Reshape", [flat_name, "GridShape"], [panel_name], name=panel_name),
                helper.make_node("Conv", [panel_name, "BoolW"], [mask_name], name=mask_name, kernel_shape=[1, 1]),
            ]
        )
        mask_names.append(mask_name)

    def add_sum(name: str) -> None:
        current = mask_names[0]
        for index, mask_name in enumerate(mask_names[1:], start=1):
            output_name = name if index == len(mask_names) - 1 else f"{name}_{index}"
            nodes.append(helper.make_node("Add", [current, mask_name], [output_name], name=output_name))
            current = output_name
        if current != name:
            nodes.append(helper.make_node("Identity", [current], [name], name=name))

    def ensure_one() -> None:
        if not any(initializer.name == "One" for initializer in initializers):
            initializers.append(numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="One"))

    if operation == "AND":
        current = mask_names[0]
        for index, mask_name in enumerate(mask_names[1:], start=1):
            output_name = "raw_mask" if index == len(mask_names) - 1 else f"and_{index}"
            nodes.append(helper.make_node("Mul", [current, mask_name], [output_name], name=output_name))
            current = output_name
    elif operation == "OR":
        add_sum("or_sum")
        nodes.append(helper.make_node("Clip", ["or_sum"], ["raw_mask"], name="raw_mask", min=0.0, max=1.0))
    elif operation in {"XOR", "NOT_EQUAL"}:
        if len(mask_names) == 2:
            initializers.append(numpy_helper.from_array(np.full((1, 1, 1, 1), 2.0, dtype=np.float32), name="Two"))
            nodes.extend(
                [
                    helper.make_node("Add", mask_names, ["xor_sum"], name="xor_sum"),
                    helper.make_node("Mul", mask_names, ["xor_both"], name="xor_both"),
                    helper.make_node("Mul", ["xor_both", "Two"], ["xor_twice"], name="xor_twice"),
                    helper.make_node("Sub", ["xor_sum", "xor_twice"], ["raw_mask"], name="raw_mask"),
                ]
            )
        else:
            raise ValueError("multi-panel XOR is intentionally unsupported")
    elif operation == "LEFT_MINUS_RIGHT":
        ensure_one()
        nodes.extend(
            [
                helper.make_node("Sub", ["One", mask_names[1]], ["not_right"], name="not_right"),
                helper.make_node("Mul", [mask_names[0], "not_right"], ["raw_mask"], name="raw_mask"),
            ]
        )
    elif operation == "RIGHT_MINUS_LEFT":
        ensure_one()
        nodes.extend(
            [
                helper.make_node("Sub", ["One", mask_names[0]], ["not_left"], name="not_left"),
                helper.make_node("Mul", [mask_names[1], "not_left"], ["raw_mask"], name="raw_mask"),
            ]
        )
    elif operation == "EQUAL":
        ensure_one()
        initializers.append(numpy_helper.from_array(np.full((1, 1, 1, 1), 2.0, dtype=np.float32), name="Two"))
        nodes.extend(
            [
                helper.make_node("Add", mask_names, ["xor_sum"], name="xor_sum"),
                helper.make_node("Mul", mask_names, ["xor_both"], name="xor_both"),
                helper.make_node("Mul", ["xor_both", "Two"], ["xor_twice"], name="xor_twice"),
                helper.make_node("Sub", ["xor_sum", "xor_twice"], ["not_equal"], name="not_equal"),
                helper.make_node("Sub", ["One", "not_equal"], ["raw_mask"], name="raw_mask"),
            ]
        )
    else:
        threshold = len(mask_names) // 2 + 1
        add_sum("panel_sum")
        initializers.append(
            numpy_helper.from_array(np.full((1, 1, 1, 1), float(threshold - 1), dtype=np.float32), name="MajorityMinusOne")
        )
        nodes.extend(
            [
                helper.make_node("Sub", ["panel_sum", "MajorityMinusOne"], ["majority_raw"], name="majority_raw"),
                helper.make_node("Clip", ["majority_raw"], ["raw_mask"], name="raw_mask", min=0.0, max=1.0),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Mul", ["raw_mask", "ActiveMaskFloat"], ["true_mask"], name="true_mask"),
            helper.make_node("Sub", ["ActiveMaskFloat", "true_mask"], ["false_mask"], name="false_mask"),
            helper.make_node("Mul", ["TrueColor", "true_mask"], ["true_part"], name="true_part"),
            helper.make_node("Mul", ["FalseColor", "false_mask"], ["false_part"], name="false_part"),
            helper.make_node("Add", ["true_part", "false_part"], ["output"], name="output"),
        ]
    )
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "generalized_panel_op",
        num_channels,
        height,
        width,
    )


def build_single_color_translation_model(
    target_color: int,
    background_color: int,
    dy: int,
    dx: int,
    active_height: int,
    active_width: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build object translation for one color while preserving other cells."""
    if target_color == background_color:
        raise ValueError("target_color and background_color must differ")
    if active_height <= 0 or active_height > height or active_width <= 0 or active_width > width:
        raise ValueError("active shape must fit static shape")

    rows = np.zeros((height, width), dtype=np.int32)
    cols = np.zeros((height, width), dtype=np.int32)
    valid = np.zeros((1, 1, height, width), dtype=np.float32)
    active_mask = _make_active_mask(active_height, active_width, height, width)
    for row in range(active_height):
        for col in range(active_width):
            source_row = row - dy
            source_col = col - dx
            if 0 <= source_row < active_height and 0 <= source_col < active_width:
                rows[row, col] = source_row
                cols[row, col] = source_col
                valid[0, 0, row, col] = 1.0

    flat_indices = (rows * width + cols).reshape(height * width).astype(np.int32)
    target_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    target_weights[0, target_color, 0, 0] = 1.0

    initializers = [
        numpy_helper.from_array(np.asarray([1, 1, height * width], dtype=np.int64), name="MaskFlatShape"),
        numpy_helper.from_array(np.asarray([1, 1, height, width], dtype=np.int64), name="MaskGridShape"),
        numpy_helper.from_array(flat_indices, name="FlatIndices"),
        numpy_helper.from_array(target_weights, name="TargetW"),
        numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="One"),
        _bool_mask(valid, "ValidMask"),
        _bool_mask(active_mask, "ActiveMask"),
        numpy_helper.from_array(_one_hot_color(target_color, num_channels), name="TargetColor"),
        numpy_helper.from_array(_one_hot_color(background_color, num_channels), name="BackgroundColor"),
    ]
    nodes = [
        _cast_to_float("ValidMask", "ValidMaskFloat"),
        _cast_to_float("ActiveMask", "ActiveMaskFloat"),
        helper.make_node("Conv", ["input", "TargetW"], ["target_mask"], name="target_mask", kernel_shape=[1, 1]),
        helper.make_node("Sub", ["One", "target_mask"], ["not_source_target"], name="not_source_target"),
        helper.make_node("Mul", ["input", "not_source_target"], ["source_removed"], name="source_removed"),
        helper.make_node("Mul", ["BackgroundColor", "target_mask"], ["source_background"], name="source_background"),
        helper.make_node("Add", ["source_removed", "source_background"], ["base_input"], name="base_input"),
        helper.make_node("Reshape", ["target_mask", "MaskFlatShape"], ["flat_mask"], name="flat_mask"),
        helper.make_node("Gather", ["flat_mask", "FlatIndices"], ["flat_shifted"], name="flat_shifted", axis=2),
        helper.make_node("Reshape", ["flat_shifted", "MaskGridShape"], ["shifted_raw"], name="shifted_raw"),
        helper.make_node("Mul", ["shifted_raw", "ValidMaskFloat"], ["shifted_mask"], name="shifted_mask"),
        helper.make_node("Sub", ["One", "shifted_mask"], ["not_dest_target"], name="not_dest_target"),
        helper.make_node("Mul", ["base_input", "not_dest_target"], ["kept_input"], name="kept_input"),
        helper.make_node("Mul", ["TargetColor", "shifted_mask"], ["shifted_target"], name="shifted_target"),
        helper.make_node("Add", ["kept_input", "shifted_target"], ["active_output"], name="active_output"),
        helper.make_node("Mul", ["active_output", "ActiveMaskFloat"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "single_color_translation",
        num_channels,
        height,
        width,
    )


def build_dynamic_single_color_translation_model(
    target_color: int,
    background_color: int,
    dy: int,
    dx: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build single-color translation with the active rectangle inferred from padding."""
    if target_color == background_color:
        raise ValueError("target_color and background_color must differ")
    if dy == 0 and dx == 0:
        raise ValueError("dy/dx cannot both be zero")
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError("dy/dx must leave at least one source row or column in bounds")

    rows: list[int] = []
    row_valid: list[bool] = []
    for row in range(height):
        source_row = row - dy
        valid = 0 <= source_row < height
        rows.append(source_row if valid else 0)
        row_valid.append(valid)
    cols: list[int] = []
    col_valid: list[bool] = []
    for col in range(width):
        source_col = col - dx
        valid = 0 <= source_col < width
        cols.append(source_col if valid else 0)
        col_valid.append(valid)

    static_valid = np.zeros((1, 1, height, width), dtype=np.bool_)
    for row in range(height):
        for col in range(width):
            static_valid[0, 0, row, col] = row_valid[row] and col_valid[col]

    target_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    target_weights[0, target_color, 0, 0] = 1.0
    initializers = [
        numpy_helper.from_array(np.ones((1, num_channels, 1, 1), dtype=np.float32), name="AnyColorW"),
        numpy_helper.from_array(target_weights, name="TargetW"),
        numpy_helper.from_array(np.asarray(rows, dtype=np.int32), name="RowIndices"),
        numpy_helper.from_array(np.asarray(cols, dtype=np.int32), name="ColIndices"),
        numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="One"),
        _bool_mask(static_valid, "StaticValidMask"),
        numpy_helper.from_array(_one_hot_color(target_color, num_channels), name="TargetColor"),
        numpy_helper.from_array(_one_hot_color(background_color, num_channels), name="BackgroundColor"),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "AnyColorW"], ["active_raw"], name="active_raw", kernel_shape=[1, 1]),
        helper.make_node("Clip", ["active_raw"], ["active_mask"], name="active_mask", min=0.0, max=1.0),
        helper.make_node("Conv", ["input", "TargetW"], ["target_mask"], name="target_mask", kernel_shape=[1, 1]),
        helper.make_node("Sub", ["One", "target_mask"], ["not_source_target"], name="not_source_target"),
        helper.make_node("Mul", ["input", "not_source_target"], ["source_removed"], name="source_removed"),
        helper.make_node("Mul", ["BackgroundColor", "target_mask"], ["source_background"], name="source_background"),
        helper.make_node("Add", ["source_removed", "source_background"], ["base_input"], name="base_input"),
        helper.make_node("Gather", ["target_mask", "RowIndices"], ["target_rows"], name="target_rows", axis=2),
        helper.make_node("Gather", ["target_rows", "ColIndices"], ["shifted_raw"], name="shifted_raw", axis=3),
        _cast_to_float("StaticValidMask", "StaticValidMaskFloat"),
        helper.make_node("Mul", ["shifted_raw", "StaticValidMaskFloat"], ["shifted_in_bounds"], name="shifted_in_bounds"),
        helper.make_node("Mul", ["shifted_in_bounds", "active_mask"], ["shifted_mask"], name="shifted_mask"),
        helper.make_node("Sub", ["One", "shifted_mask"], ["not_dest_target"], name="not_dest_target"),
        helper.make_node("Mul", ["base_input", "not_dest_target"], ["kept_input"], name="kept_input"),
        helper.make_node("Mul", ["TargetColor", "shifted_mask"], ["shifted_target"], name="shifted_target"),
        helper.make_node("Add", ["kept_input", "shifted_target"], ["active_output"], name="active_output"),
        helper.make_node("Mul", ["active_output", "active_mask"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "dynamic_single_color_translation",
        num_channels,
        height,
        width,
    )


def build_zero_fill_translation_remap_model(
    dy: int,
    dx: int,
    active_height: int,
    active_width: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a zero-fill whole-grid translation with the generic remap builder."""
    if active_height <= 0 or active_height > height or active_width <= 0 or active_width > width:
        raise ValueError("active shape must fit static shape")
    rows = np.zeros((height, width), dtype=np.int32)
    cols = np.zeros((height, width), dtype=np.int32)
    valid_mask = np.zeros((1, 1, height, width), dtype=np.float32)
    active_mask = _make_active_mask(active_height, active_width, height, width)
    for row in range(active_height):
        for col in range(active_width):
            source_row = row - dy
            source_col = col - dx
            if 0 <= source_row < active_height and 0 <= source_col < active_width:
                rows[row, col] = source_row
                cols[row, col] = source_col
                valid_mask[0, 0, row, col] = 1.0
    flat_indices = (rows * width + cols).reshape(height * width).astype(np.int32)
    initializers = [
        numpy_helper.from_array(np.asarray([1, num_channels, height * width], dtype=np.int64), name="FlatShape"),
        numpy_helper.from_array(flat_indices, name="FlatIndices"),
        numpy_helper.from_array(np.asarray([1, num_channels, height, width], dtype=np.int64), name="GridShape"),
        _bool_mask(valid_mask, "ValidMask"),
        _bool_mask(active_mask, "ActiveMask"),
        numpy_helper.from_array(_one_hot_color(0, num_channels), name="ZeroColor"),
    ]
    nodes = [
        _cast_to_float("ValidMask", "ValidMaskFloat"),
        _cast_to_float("ActiveMask", "ActiveMaskFloat"),
        helper.make_node("Reshape", ["input", "FlatShape"], ["flat_input"], name="flat_input"),
        helper.make_node("Gather", ["flat_input", "FlatIndices"], ["flat_remapped"], name="flat_remapped", axis=2),
        helper.make_node("Reshape", ["flat_remapped", "GridShape"], ["remapped"], name="remapped"),
        helper.make_node("Mul", ["remapped", "ValidMaskFloat"], ["shifted"], name="shifted"),
        helper.make_node("Sub", ["ActiveMaskFloat", "ValidMaskFloat"], ["fill_mask"], name="fill_mask"),
        helper.make_node("Mul", ["ZeroColor", "fill_mask"], ["zero_fill"], name="zero_fill"),
        helper.make_node("Add", ["shifted", "zero_fill"], ["active_output"], name="active_output"),
        helper.make_node("Mul", ["active_output", "ActiveMaskFloat"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "zero_fill_translation_remap",
        num_channels,
        height,
        width,
    )


def build_dynamic_fill_translation_model(
    dy: int,
    dx: int,
    fill_color: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a same-size translation that infers the active rectangle from input padding."""
    if dy == 0 and dx == 0:
        raise ValueError("dy/dx cannot both be zero")
    if abs(dy) >= height or abs(dx) >= width:
        raise ValueError("dy/dx must leave at least one source row or column in bounds")
    if fill_color < 0 or fill_color >= num_channels:
        raise ValueError(f"fill_color must be within 0..{num_channels - 1}")

    rows: list[int] = []
    row_valid: list[bool] = []
    for row in range(height):
        source_row = row - dy
        valid = 0 <= source_row < height
        rows.append(source_row if valid else 0)
        row_valid.append(valid)
    cols: list[int] = []
    col_valid: list[bool] = []
    for col in range(width):
        source_col = col - dx
        valid = 0 <= source_col < width
        cols.append(source_col if valid else 0)
        col_valid.append(valid)

    static_valid = np.zeros((1, 1, height, width), dtype=np.bool_)
    for row in range(height):
        for col in range(width):
            static_valid[0, 0, row, col] = row_valid[row] and col_valid[col]

    initializers = [
        numpy_helper.from_array(np.ones((1, num_channels, 1, 1), dtype=np.float32), name="AnyColorW"),
        numpy_helper.from_array(np.asarray(rows, dtype=np.int32), name="RowIndices"),
        numpy_helper.from_array(np.asarray(cols, dtype=np.int32), name="ColIndices"),
        _bool_mask(static_valid, "StaticValidMask"),
        numpy_helper.from_array(_one_hot_color(fill_color, num_channels), name="FillColor"),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "AnyColorW"], ["active_raw"], name="active_raw", kernel_shape=[1, 1]),
        helper.make_node("Clip", ["active_raw"], ["active_mask"], name="active_mask", min=0.0, max=1.0),
        helper.make_node("Gather", ["input", "RowIndices"], ["gather_rows"], name="gather_rows", axis=2),
        helper.make_node("Gather", ["gather_rows", "ColIndices"], ["remapped"], name="remapped", axis=3),
        helper.make_node("Gather", ["active_mask", "RowIndices"], ["source_active_rows"], name="source_active_rows", axis=2),
        helper.make_node("Gather", ["source_active_rows", "ColIndices"], ["source_active_raw"], name="source_active_raw", axis=3),
        _cast_to_float("StaticValidMask", "StaticValidMaskFloat"),
        helper.make_node("Mul", ["source_active_raw", "StaticValidMaskFloat"], ["source_active"], name="source_active"),
        helper.make_node("Mul", ["source_active", "active_mask"], ["transfer_mask"], name="transfer_mask"),
        helper.make_node("Mul", ["remapped", "transfer_mask"], ["shifted"], name="shifted"),
        helper.make_node("Sub", ["active_mask", "transfer_mask"], ["fill_mask"], name="fill_mask"),
        helper.make_node("Mul", ["FillColor", "fill_mask"], ["edge_fill"], name="edge_fill"),
        helper.make_node("Add", ["shifted", "edge_fill"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "dynamic_fill_translation",
        num_channels,
        height,
        width,
    )


def build_active_rectangle_model(
    mode: str,
    draw_color: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a dynamic rectangle fill/frame over the top-left active input area."""
    if mode not in {"fill", "frame"}:
        raise ValueError("mode must be fill or frame")
    active_color = _one_hot_color(draw_color, num_channels)
    initializers = [
        numpy_helper.from_array(np.ones((1, num_channels, 1, 1), dtype=np.float32), name="AnyColorW"),
        numpy_helper.from_array(active_color, name="DrawColor"),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "AnyColorW"], ["active_raw"], name="active_raw", kernel_shape=[1, 1]),
        helper.make_node("Clip", ["active_raw"], ["active_mask"], name="active_mask", min=0.0, max=1.0),
    ]
    if mode == "fill":
        nodes.append(helper.make_node("Mul", ["DrawColor", "active_mask"], ["output"], name="output"))
    else:
        cardinal_weights = np.zeros((1, 1, 3, 3), dtype=np.float32)
        cardinal_weights[0, 0, 0, 1] = 1.0
        cardinal_weights[0, 0, 1, 0] = 1.0
        cardinal_weights[0, 0, 1, 2] = 1.0
        cardinal_weights[0, 0, 2, 1] = 1.0
        initializers.extend(
            [
                numpy_helper.from_array(cardinal_weights, name="CardinalW"),
                numpy_helper.from_array(np.full((1, 1, 1, 1), 3.0, dtype=np.float32), name="Three"),
                numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="One"),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    ["active_mask", "CardinalW"],
                    ["cardinal_count"],
                    name="cardinal_count",
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                ),
                helper.make_node("Sub", ["cardinal_count", "Three"], ["interior_raw"], name="interior_raw"),
                helper.make_node("Clip", ["interior_raw"], ["interior_mask"], name="interior_mask", min=0.0, max=1.0),
                helper.make_node("Sub", ["One", "interior_mask"], ["not_interior"], name="not_interior"),
                helper.make_node("Mul", ["active_mask", "not_interior"], ["frame_mask"], name="frame_mask"),
                helper.make_node("Sub", ["active_mask", "frame_mask"], ["keep_mask"], name="keep_mask"),
                helper.make_node("Mul", ["input", "keep_mask"], ["kept_input"], name="kept_input"),
                helper.make_node("Mul", ["DrawColor", "frame_mask"], ["frame"], name="frame"),
                helper.make_node("Add", ["kept_input", "frame"], ["output"], name="output"),
            ]
        )
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "active_rectangle",
        num_channels,
        height,
        width,
    )


def build_static_overlay_model(
    draw_mask: list[list[bool]],
    draw_color: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build output = input with a static boolean mask painted in draw_color."""
    if not draw_mask or not draw_mask[0]:
        raise ValueError("draw_mask must be a non-empty 2D list")
    mask_height = len(draw_mask)
    mask_width = len(draw_mask[0])
    if mask_height > height or mask_width > width:
        raise ValueError("draw_mask exceeds model spatial dimensions")
    if any(len(row) != mask_width for row in draw_mask):
        raise ValueError("draw_mask must be rectangular")

    mask = np.zeros((1, 1, height, width), dtype=np.bool_)
    for row_index, row in enumerate(draw_mask):
        for col_index, value in enumerate(row):
            mask[0, 0, row_index, col_index] = bool(value)

    initializers = [
        _bool_mask(mask, "DrawMask"),
        numpy_helper.from_array(_one_hot_color(draw_color, num_channels), name="DrawColor"),
        numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="One"),
    ]
    nodes = [
        _cast_to_float("DrawMask", "DrawMaskFloat"),
        helper.make_node("Sub", ["One", "DrawMaskFloat"], ["KeepMask"], name="KeepMask"),
        helper.make_node("Mul", ["input", "KeepMask"], ["KeptInput"], name="KeptInput"),
        helper.make_node("Mul", ["DrawColor", "DrawMaskFloat"], ["Drawn"], name="Drawn"),
        helper.make_node("Add", ["KeptInput", "Drawn"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "static_overlay",
        num_channels,
        height,
        width,
    )


def build_line_extension_model(
    direction: str,
    draw_color: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build dynamic horizontal/vertical line extension over active cells."""
    if direction not in {"horizontal", "vertical"}:
        raise ValueError("direction must be horizontal or vertical")
    color_selector = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    color_selector[0, draw_color, 0, 0] = 1.0
    reduce_axes = [3] if direction == "horizontal" else [2]
    initializers = [
        numpy_helper.from_array(np.ones((1, num_channels, 1, 1), dtype=np.float32), name="AnyColorW"),
        numpy_helper.from_array(color_selector, name="ColorW"),
        numpy_helper.from_array(_one_hot_color(draw_color, num_channels), name="DrawColor"),
        numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="One"),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "AnyColorW"], ["active_raw"], name="active_raw", kernel_shape=[1, 1]),
        helper.make_node("Clip", ["active_raw"], ["active_mask"], name="active_mask", min=0.0, max=1.0),
        helper.make_node("Conv", ["input", "ColorW"], ["color_mask"], name="color_mask", kernel_shape=[1, 1]),
        helper.make_node(
            "ReduceMax",
            ["color_mask"],
            ["line_has_color"],
            name="line_has_color",
            axes=reduce_axes,
            keepdims=1,
        ),
        helper.make_node("Mul", ["line_has_color", "active_mask"], ["line_mask"], name="line_mask"),
        helper.make_node("Sub", ["One", "line_mask"], ["keep_mask"], name="keep_mask"),
        helper.make_node("Mul", ["input", "keep_mask"], ["kept_input"], name="kept_input"),
        helper.make_node("Mul", ["DrawColor", "line_mask"], ["line"], name="line"),
        helper.make_node("Add", ["kept_input", "line"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "line_extension",
        num_channels,
        height,
        width,
    )


def build_symmetry_completion_model(
    mode: str,
    background_color: int,
    active_height: int,
    active_width: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build output = input, with background cells filled from a symmetric source."""
    if mode not in {"horizontal", "vertical", "rot180", "diag_main", "diag_anti"}:
        raise ValueError("unsupported symmetry mode")
    if active_height <= 0 or active_height > height:
        raise ValueError("active_height must be within 1..height")
    if active_width <= 0 or active_width > width:
        raise ValueError("active_width must be within 1..width")
    if mode in {"diag_main", "diag_anti"} and active_height != active_width:
        raise ValueError("diagonal completion requires a square active shape")

    if mode == "horizontal":
        rows = list(range(active_height))
        cols = list(range(active_width - 1, -1, -1))
    elif mode == "vertical":
        rows = list(range(active_height - 1, -1, -1))
        cols = list(range(active_width))
    elif mode == "rot180":
        rows = list(range(active_height - 1, -1, -1))
        cols = list(range(active_width - 1, -1, -1))
    else:
        rows = list(range(active_height))
        cols = list(range(active_width))

    pad_row = active_height if active_height < height else 0
    pad_col = active_width if active_width < width else 0
    background_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    background_weights[0, background_color, 0, 0] = 1.0
    non_background_weights = np.ones((1, num_channels, 1, 1), dtype=np.float32)
    non_background_weights[0, background_color, 0, 0] = 0.0
    active_mask = _make_active_mask(active_height, active_width, height, width)
    background_value = _one_hot_color(background_color, num_channels)

    initializers = [
        numpy_helper.from_array(_padded_indices(rows, height, height, pad_row), name="RowIndices"),
        numpy_helper.from_array(_padded_indices(cols, width, width, pad_col), name="ColIndices"),
        numpy_helper.from_array(background_weights, name="BackgroundW"),
        numpy_helper.from_array(non_background_weights, name="NonBackgroundW"),
        _bool_mask(active_mask, "ActiveMask"),
        numpy_helper.from_array(background_value, name="BackgroundColor"),
    ]
    nodes = [_cast_to_float("ActiveMask", "ActiveMaskFloat")]
    if mode == "diag_main":
        nodes.append(helper.make_node("Transpose", ["input"], ["mirror"], name="mirror", perm=[0, 1, 3, 2]))
    elif mode == "diag_anti":
        nodes.extend(
            [
                helper.make_node("Gather", ["input", "RowIndices"], ["rev_rows"], name="rev_rows", axis=2),
                helper.make_node("Gather", ["rev_rows", "ColIndices"], ["rev_both"], name="rev_both", axis=3),
                helper.make_node("Transpose", ["rev_both"], ["mirror"], name="mirror", perm=[0, 1, 3, 2]),
            ]
        )
    else:
        nodes.extend(
            [
                helper.make_node("Gather", ["input", "RowIndices"], ["mirror_rows"], name="mirror_rows", axis=2),
                helper.make_node("Gather", ["mirror_rows", "ColIndices"], ["mirror"], name="mirror", axis=3),
            ]
        )

    nodes.extend(
        [
            helper.make_node("Conv", ["input", "BackgroundW"], ["input_bg"], name="input_bg", kernel_shape=[1, 1]),
            helper.make_node("Conv", ["input", "NonBackgroundW"], ["input_non_bg"], name="input_non_bg", kernel_shape=[1, 1]),
            helper.make_node("Conv", ["mirror", "NonBackgroundW"], ["mirror_non_bg"], name="mirror_non_bg", kernel_shape=[1, 1]),
            helper.make_node("Mul", ["input_bg", "mirror_non_bg"], ["fill_mask_raw"], name="fill_mask_raw"),
            helper.make_node("Mul", ["fill_mask_raw", "ActiveMaskFloat"], ["fill_mask"], name="fill_mask"),
            helper.make_node("Mul", ["input", "input_non_bg"], ["input_foreground"], name="input_foreground"),
            helper.make_node("Mul", ["mirror", "fill_mask"], ["mirror_fill"], name="mirror_fill"),
            helper.make_node("Add", ["input_foreground", "mirror_fill"], ["fg_output"], name="fg_output"),
            helper.make_node("Conv", ["fg_output", "NonBackgroundW"], ["fg_any"], name="fg_any", kernel_shape=[1, 1]),
            helper.make_node("Sub", ["ActiveMaskFloat", "fg_any"], ["background_mask"], name="background_mask"),
            helper.make_node("Mul", ["BackgroundColor", "background_mask"], ["background"], name="background"),
            helper.make_node("Add", ["fg_output", "background"], ["active_output"], name="active_output"),
            helper.make_node("Mul", ["active_output", "ActiveMaskFloat"], ["output"], name="output"),
        ]
    )
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "symmetry_completion",
        num_channels,
        height,
        width,
    )


def build_local_neighborhood_fill_model(
    background_color: int,
    fill_color: int,
    source_colors: list[int],
    offsets: list[tuple[int, int]],
    condition: str,
    threshold: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a local rule that fills background cells selected by 3x3/5x5 counts."""
    if condition not in {"eq", "ge"}:
        raise ValueError("condition must be eq or ge")
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if not offsets:
        raise ValueError("offsets must be non-empty")
    if threshold > len(offsets):
        raise ValueError("threshold cannot exceed the number of offsets")
    max_abs_offset = max(max(abs(row_offset), abs(col_offset)) for row_offset, col_offset in offsets)
    if max_abs_offset not in {1, 2}:
        raise ValueError("offsets must fit a 3x3 or 5x5 kernel")
    kernel_size = max_abs_offset * 2 + 1
    center = max_abs_offset

    count_weights = np.zeros((1, num_channels, kernel_size, kernel_size), dtype=np.float32)
    for color in source_colors:
        if color < 0 or color >= num_channels:
            raise ValueError(f"source color {color} is outside 0..{num_channels - 1}")
        for row_offset, col_offset in offsets:
            count_weights[0, color, row_offset + center, col_offset + center] = 1.0
    background_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    background_weights[0, background_color, 0, 0] = 1.0

    initializers = [
        numpy_helper.from_array(count_weights, name="CountW"),
        numpy_helper.from_array(background_weights, name="BackgroundW"),
        numpy_helper.from_array(np.full((1, 1, 1, 1), float(threshold - 1), dtype=np.float32), name="ThresholdMinusOne"),
        numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="One"),
        numpy_helper.from_array(_one_hot_color(fill_color, num_channels), name="FillColor"),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "CountW"],
            ["neighbor_count"],
            name="neighbor_count",
            kernel_shape=[kernel_size, kernel_size],
            pads=[center, center, center, center],
        ),
        helper.make_node("Sub", ["neighbor_count", "ThresholdMinusOne"], ["ge_raw"], name="ge_raw"),
        helper.make_node("Clip", ["ge_raw"], ["ge_mask"], name="ge_mask", min=0.0, max=1.0),
    ]
    if condition == "eq":
        initializers.append(
            numpy_helper.from_array(np.full((1, 1, 1, 1), float(threshold), dtype=np.float32), name="Threshold")
        )
        nodes.extend(
            [
                helper.make_node("Sub", ["neighbor_count", "Threshold"], ["ge_next_raw"], name="ge_next_raw"),
                helper.make_node("Clip", ["ge_next_raw"], ["ge_next"], name="ge_next", min=0.0, max=1.0),
                helper.make_node("Sub", ["ge_mask", "ge_next"], ["condition_mask"], name="condition_mask"),
            ]
        )
    else:
        nodes.append(helper.make_node("Identity", ["ge_mask"], ["condition_mask"], name="condition_mask"))

    nodes.extend(
        [
            helper.make_node("Conv", ["input", "BackgroundW"], ["background_mask"], name="background_mask", kernel_shape=[1, 1]),
            helper.make_node("Mul", ["condition_mask", "background_mask"], ["fill_mask"], name="fill_mask"),
            helper.make_node("Sub", ["One", "fill_mask"], ["keep_mask"], name="keep_mask"),
            helper.make_node("Mul", ["input", "keep_mask"], ["kept_input"], name="kept_input"),
            helper.make_node("Mul", ["FillColor", "fill_mask"], ["filled_cells"], name="filled_cells"),
            helper.make_node("Add", ["kept_input", "filled_cells"], ["output"], name="output"),
        ]
    )
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "local_neighborhood_fill",
        num_channels,
        height,
        width,
    )


def build_local_neighborhood_rewrite_model(
    target_color: int,
    replacement_color: int,
    source_colors: list[int],
    offsets: list[tuple[int, int]],
    condition: str,
    threshold: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a local rule that rewrites target-color cells selected by counts."""
    if condition not in {"eq", "ge"}:
        raise ValueError("condition must be eq or ge")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if not offsets:
        raise ValueError("offsets must be non-empty")
    max_abs_offset = max(max(abs(row_offset), abs(col_offset)) for row_offset, col_offset in offsets)
    if max_abs_offset not in {1, 2}:
        raise ValueError("offsets must fit a 3x3 or 5x5 kernel")
    kernel_size = max_abs_offset * 2 + 1
    center = max_abs_offset

    count_weights = np.zeros((1, num_channels, kernel_size, kernel_size), dtype=np.float32)
    for color in source_colors:
        if color < 0 or color >= num_channels:
            raise ValueError(f"source color {color} is outside 0..{num_channels - 1}")
        for row_offset, col_offset in offsets:
            count_weights[0, color, row_offset + center, col_offset + center] = 1.0
    target_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    target_weights[0, target_color, 0, 0] = 1.0

    initializers = [
        numpy_helper.from_array(count_weights, name="CountW"),
        numpy_helper.from_array(target_weights, name="TargetW"),
        numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="One"),
        numpy_helper.from_array(_one_hot_color(replacement_color, num_channels), name="ReplacementColor"),
    ]
    nodes = [
        helper.make_node(
            "Conv",
            ["input", "CountW"],
            ["neighbor_count"],
            name="neighbor_count",
            kernel_shape=[kernel_size, kernel_size],
            pads=[center, center, center, center],
        ),
        helper.make_node("Conv", ["input", "TargetW"], ["target_mask"], name="target_mask", kernel_shape=[1, 1]),
    ]

    def add_ge_mask(mask_name: str, value: int) -> None:
        if value <= 0:
            nodes.append(helper.make_node("Identity", ["One"], [mask_name], name=mask_name))
            return
        init_name = f"ThresholdMinusOne_{value}"
        initializers.append(
            numpy_helper.from_array(np.full((1, 1, 1, 1), float(value - 1), dtype=np.float32), name=init_name)
        )
        raw_name = f"{mask_name}_raw"
        nodes.append(helper.make_node("Sub", ["neighbor_count", init_name], [raw_name], name=raw_name))
        nodes.append(helper.make_node("Clip", [raw_name], [mask_name], name=mask_name, min=0.0, max=1.0))

    if condition == "ge":
        add_ge_mask("condition_mask", threshold)
    else:
        add_ge_mask("ge_current", threshold)
        add_ge_mask("ge_next", threshold + 1)
        nodes.append(helper.make_node("Sub", ["ge_current", "ge_next"], ["condition_mask"], name="condition_mask"))

    nodes.extend(
        [
            helper.make_node("Mul", ["target_mask", "condition_mask"], ["rewrite_raw"], name="rewrite_raw"),
            helper.make_node("Sub", ["One", "rewrite_raw"], ["keep_mask"], name="keep_mask"),
            helper.make_node("Mul", ["input", "keep_mask"], ["kept_input"], name="kept_input"),
            helper.make_node("Mul", ["ReplacementColor", "rewrite_raw"], ["replacement"], name="replacement"),
            helper.make_node("Add", ["kept_input", "replacement"], ["output"], name="output"),
        ]
    )
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "local_neighborhood_rewrite",
        num_channels,
        height,
        width,
    )


def build_hole_fill_model(
    background_color: int,
    fill_color: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build a fixed-depth flood-fill model that recolors enclosed background holes."""
    if background_color == fill_color:
        raise ValueError("background_color and fill_color must differ")
    background_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    background_weights[0, background_color, 0, 0] = 1.0
    cardinal_weights = np.zeros((1, 1, 3, 3), dtype=np.float32)
    cardinal_weights[0, 0, 0, 1] = 1.0
    cardinal_weights[0, 0, 1, 0] = 1.0
    cardinal_weights[0, 0, 1, 2] = 1.0
    cardinal_weights[0, 0, 2, 1] = 1.0

    initializers = [
        numpy_helper.from_array(np.ones((1, num_channels, 1, 1), dtype=np.float32), name="AnyColorW"),
        numpy_helper.from_array(background_weights, name="BackgroundW"),
        numpy_helper.from_array(cardinal_weights, name="CardinalW"),
        numpy_helper.from_array(np.full((1, 1, 1, 1), 4.0, dtype=np.float32), name="Four"),
        numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), name="One"),
        numpy_helper.from_array(_one_hot_color(fill_color, num_channels), name="FillColor"),
    ]
    nodes = [
        helper.make_node("Conv", ["input", "AnyColorW"], ["active_raw"], name="active_raw", kernel_shape=[1, 1]),
        helper.make_node("Clip", ["active_raw"], ["active_mask"], name="active_mask", min=0.0, max=1.0),
        helper.make_node("Conv", ["input", "BackgroundW"], ["background_mask"], name="background_mask", kernel_shape=[1, 1]),
        helper.make_node(
            "Conv",
            ["active_mask", "CardinalW"],
            ["active_neighbors"],
            name="active_neighbors",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
        ),
        helper.make_node("Sub", ["Four", "active_neighbors"], ["border_raw"], name="border_raw"),
        helper.make_node("Clip", ["border_raw"], ["border_any"], name="border_any", min=0.0, max=1.0),
        helper.make_node("Mul", ["background_mask", "border_any"], ["reachable_0"], name="reachable_0"),
    ]
    previous = "reachable_0"
    for iteration in range(max(height, width)):
        neighbor_name = f"reachable_neighbors_{iteration}"
        sum_name = f"reachable_sum_{iteration}"
        expanded_name = f"reachable_expanded_{iteration}"
        next_name = f"reachable_{iteration + 1}"
        nodes.extend(
            [
                helper.make_node(
                    "Conv",
                    [previous, "CardinalW"],
                    [neighbor_name],
                    name=neighbor_name,
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                ),
                helper.make_node("Add", [previous, neighbor_name], [sum_name], name=sum_name),
                helper.make_node("Clip", [sum_name], [expanded_name], name=expanded_name, min=0.0, max=1.0),
                helper.make_node("Mul", [expanded_name, "background_mask"], [next_name], name=next_name),
            ]
        )
        previous = next_name
    nodes.extend(
        [
            helper.make_node("Sub", ["background_mask", previous], ["hole_mask"], name="hole_mask"),
            helper.make_node("Sub", ["One", "hole_mask"], ["keep_mask"], name="keep_mask"),
            helper.make_node("Mul", ["input", "keep_mask"], ["kept_input"], name="kept_input"),
            helper.make_node("Mul", ["FillColor", "hole_mask"], ["filled_holes"], name="filled_holes"),
            helper.make_node("Add", ["kept_input", "filled_holes"], ["output"], name="output"),
        ]
    )
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "hole_fill",
        num_channels,
        height,
        width,
    )


def build_scale_repeat_model(
    scale_y: int,
    scale_x: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build nearest-neighbor repeat scaling with fixed integer factors."""
    if scale_y <= 0 or scale_x <= 0:
        raise ValueError("scale_y and scale_x must be positive")
    rows = [min(row // scale_y, height - 1) for row in range(height)]
    cols = [min(col // scale_x, width - 1) for col in range(width)]
    build_spatial_remap_model(rows, cols, output_path, None, None, num_channels, height, width)


def build_self_kron_mask_model(
    input_height: int,
    input_width: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> None:
    """Build output = kron(input != 0, input) for a fixed small input shape."""
    output_height = input_height * input_height
    output_width = input_width * input_width
    if output_height > height or output_width > width:
        raise ValueError("self-kron output exceeds static tensor shape")

    tile_rows = _padded_indices([row % input_height for row in range(output_height)], height, height)
    tile_cols = _padded_indices([col % input_width for col in range(output_width)], width, width)
    block_rows = _padded_indices([row // input_height for row in range(output_height)], height, height)
    block_cols = _padded_indices([col // input_width for col in range(output_width)], width, width)
    nonzero_weights = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    nonzero_weights[0, 1:, 0, 0] = 1.0
    active_mask = np.zeros((1, 1, height, width), dtype=np.float32)
    active_mask[:, :, :output_height, :output_width] = 1.0
    zero_color = np.zeros((1, num_channels, 1, 1), dtype=np.float32)
    zero_color[:, 0, :, :] = 1.0

    initializers = [
        numpy_helper.from_array(tile_rows, name="TileRows"),
        numpy_helper.from_array(tile_cols, name="TileCols"),
        numpy_helper.from_array(block_rows, name="BlockRows"),
        numpy_helper.from_array(block_cols, name="BlockCols"),
        numpy_helper.from_array(nonzero_weights, name="NonZeroW"),
        _bool_mask(active_mask, "ActiveMask"),
        numpy_helper.from_array(zero_color, name="ZeroColor"),
    ]
    nodes = [
        _cast_to_float("ActiveMask", "ActiveMaskFloat"),
        helper.make_node("Gather", ["input", "TileRows"], ["tile_rows"], name="tile_rows", axis=2),
        helper.make_node("Gather", ["tile_rows", "TileCols"], ["tile"], name="tile", axis=3),
        helper.make_node("Gather", ["input", "BlockRows"], ["block_rows"], name="block_rows", axis=2),
        helper.make_node("Gather", ["block_rows", "BlockCols"], ["block"], name="block", axis=3),
        helper.make_node(
            "Conv",
            ["block", "NonZeroW"],
            ["raw_mask"],
            name="raw_mask",
            kernel_shape=[1, 1],
            strides=[1, 1],
        ),
        helper.make_node("Mul", ["raw_mask", "ActiveMaskFloat"], ["mask"], name="mask"),
        helper.make_node("Sub", ["ActiveMaskFloat", "mask"], ["inverse_mask"], name="inverse_mask"),
        helper.make_node("Mul", ["tile", "mask"], ["foreground"], name="foreground"),
        helper.make_node("Mul", ["ZeroColor", "inverse_mask"], ["background"], name="background"),
        helper.make_node("Add", ["foreground", "background"], ["output"], name="output"),
    ]
    _save_checked_model(
        output_path,
        nodes,
        initializers,
        "self_kron_mask",
        num_channels,
        height,
        width,
    )


def build_mirror_model(
    mode: str,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    active_height: int | None = None,
    active_width: int | None = None,
) -> None:
    """Build a horizontal or vertical mirror model within a fixed active shape."""
    if mode not in {"horizontal", "vertical"}:
        raise ValueError("mode must be horizontal or vertical")
    active_height = height if active_height is None else active_height
    active_width = width if active_width is None else active_width
    rows = list(range(active_height))
    cols = list(range(active_width))
    if mode == "horizontal":
        cols = list(reversed(cols))
    else:
        rows = list(reversed(rows))
    padding = _padding_coordinate(active_height, active_width, height, width)
    if padding is None:
        build_spatial_remap_model(
            rows,
            cols,
            output_path,
            active_height,
            active_width,
            num_channels,
            height,
            width,
        )
    else:
        pad_row, pad_col = padding
        build_spatial_remap_model(
            rows,
            cols,
            output_path,
            None,
            None,
            num_channels,
            height,
            width,
            pad_row,
            pad_col,
        )


def build_rotate_model(
    k: int,
    output_path: str,
    num_channels: int = DEFAULT_COLORS,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    input_active_height: int | None = None,
    input_active_width: int | None = None,
    output_active_height: int | None = None,
    output_active_width: int | None = None,
) -> None:
    """Build a 90/180/270 degree rotation model within fixed active shapes."""
    if k not in {1, 2, 3}:
        raise ValueError("k must be 1, 2, or 3")
    input_active_height = height if input_active_height is None else input_active_height
    input_active_width = width if input_active_width is None else input_active_width
    if k == 2:
        expected_output_height = input_active_height
        expected_output_width = input_active_width
    else:
        expected_output_height = input_active_width
        expected_output_width = input_active_height
    output_active_height = expected_output_height if output_active_height is None else output_active_height
    output_active_width = expected_output_width if output_active_width is None else output_active_width
    if (output_active_height, output_active_width) != (expected_output_height, expected_output_width):
        raise ValueError("output active shape does not match rotated input active shape")

    if k == 2:
        padding = _padding_coordinate(input_active_height, input_active_width, height, width)
        if padding is None:
            build_spatial_remap_model(
                list(range(input_active_height - 1, -1, -1)),
                list(range(input_active_width - 1, -1, -1)),
                output_path,
                output_active_height,
                output_active_width,
                num_channels,
                height,
                width,
            )
        else:
            pad_row, pad_col = padding
            build_spatial_remap_model(
                list(range(input_active_height - 1, -1, -1)),
                list(range(input_active_width - 1, -1, -1)),
                output_path,
                None,
                None,
                num_channels,
                height,
                width,
                pad_row,
                pad_col,
            )
        return

    padding = _padding_coordinate(input_active_height, input_active_width, height, width)
    pad_row = 0 if padding is None else padding[0]
    pad_col = 0 if padding is None else padding[1]
    row_indices = _padded_indices(list(range(input_active_height - 1, -1, -1)), height, height, pad_row)
    col_indices = _padded_indices(list(range(input_active_width - 1, -1, -1)), width, width, pad_col)
    initializers = []
    nodes = []
    if k == 1:
        initializers.append(numpy_helper.from_array(row_indices, name="RowIndices"))
        nodes.append(helper.make_node("Gather", ["input", "RowIndices"], ["rev_rows"], name="rev_rows", axis=2))
        nodes.append(helper.make_node("Transpose", ["rev_rows"], ["rotated"], name="rotated", perm=[0, 1, 3, 2]))
    else:
        initializers.append(numpy_helper.from_array(col_indices, name="ColIndices"))
        nodes.append(helper.make_node("Gather", ["input", "ColIndices"], ["rev_cols"], name="rev_cols", axis=3))
        nodes.append(helper.make_node("Transpose", ["rev_cols"], ["rotated"], name="rotated", perm=[0, 1, 3, 2]))
    if padding is None:
        active_mask = np.zeros((1, 1, height, width), dtype=np.float32)
        active_mask[:, :, :output_active_height, :output_active_width] = 1.0
        initializers.append(_bool_mask(active_mask, "ActiveMask"))
        nodes.append(_cast_to_float("ActiveMask", "ActiveMaskFloat"))
        nodes.append(helper.make_node("Mul", ["rotated", "ActiveMaskFloat"], ["output"], name="output"))
    else:
        nodes.append(helper.make_node("Identity", ["rotated"], ["output"], name="output"))

    _save_checked_model(
        output_path,
        nodes,
        initializers,
        f"rotate_{k}",
        num_channels,
        height,
        width,
    )
