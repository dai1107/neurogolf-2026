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
