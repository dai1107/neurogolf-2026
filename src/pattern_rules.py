"""Conservative first-version ARC pattern rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .encoding import DEFAULT_HEIGHT, DEFAULT_WIDTH
from .onnx_builders import (
    build_active_rectangle_model,
    build_auto_periodic_extension_color_map_model,
    build_dynamic_active_mirror_model,
    build_generalized_panel_op_model,
    build_hole_fill_model,
    build_panel_binary_op_model,
    build_color_map_model,
    build_dynamic_bbox_extreme_color_swap_model,
    build_dynamic_color_bbox_crop_model,
    build_dynamic_frame_interior_crop_model,
    build_dynamic_non_background_bbox_crop_model,
    build_dynamic_quadrant_panel_select_model,
    build_dynamic_fill_translation_model,
    build_dynamic_single_color_translation_model,
    build_identity_model,
    build_local_neighborhood_fill_model,
    build_local_neighborhood_rewrite_model,
    build_line_extension_model,
    build_mirror_model,
    build_periodic_extension_color_map_model,
    build_rotate_model,
    build_scale_repeat_model,
    build_self_kron_mask_model,
    build_single_color_translation_model,
    build_small_translation_model,
    build_spatial_remap_model,
    build_static_overlay_model,
    build_symmetry_completion_model,
    build_zero_fill_translation_remap_model,
)


@dataclass(frozen=True)
class RuleResult:
    rule_name: str
    matched: bool
    confidence: str
    reason: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CandidateModel:
    task_id: str
    rule_name: str
    model_path: str
    metadata: dict[str, Any]


def grid_shape(grid: list[list[int]]) -> tuple[int, int]:
    return len(grid), len(grid[0])


def same_grid(a: list[list[int]], b: list[list[int]]) -> bool:
    return a == b


def _train_cases(task: dict) -> list[dict]:
    cases = task.get("train", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("task must contain a non-empty train list")
    return cases


def _input_output_shapes_equal(case: dict) -> bool:
    return grid_shape(case["input"]) == grid_shape(case["output"])


class BaseRule:
    name = "BaseRule"
    priority = 999

    def match(self, task: dict) -> RuleResult:
        raise NotImplementedError

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        raise NotImplementedError


class IdentityRule(BaseRule):
    name = "IdentityRule"
    priority = 0

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "output_size_differs_from_input_size", {})
        if not all(same_grid(case["input"], case["output"]) for case in cases):
            return RuleResult(self.name, False, "REJECT", "input_output_not_identical", {})
        return RuleResult(self.name, True, "MATCH", "all train cases are exact identity", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_identity_model(output_path)
        return CandidateModel(task_id, self.name, output_path, metadata)


class ColorMapRule(BaseRule):
    name = "ColorMapRule"
    priority = 1

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "output_size_differs_from_input_size", {})

        color_map: dict[int, int] = {}
        for case_index, case in enumerate(cases):
            input_grid = case["input"]
            output_grid = case["output"]
            for row_index, row in enumerate(input_grid):
                for col_index, old_color in enumerate(row):
                    new_color = output_grid[row_index][col_index]
                    existing = color_map.get(old_color)
                    if existing is not None and existing != new_color:
                        return RuleResult(
                            self.name,
                            False,
                            "REJECT",
                            (
                                f"inconsistent mapping for color {old_color}: "
                                f"{existing} vs {new_color} at case {case_index}, "
                                f"row {row_index}, col {col_index}"
                            ),
                            {},
                        )
                    color_map[old_color] = new_color
        return RuleResult(
            self.name,
            True,
            "MATCH",
            "all observed input colors map to one output color",
            {"color_map": color_map},
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_color_map_model(metadata["color_map"], output_path)
        return CandidateModel(task_id, self.name, output_path, metadata)


def _translated_grid(grid: list[list[int]], dy: int, dx: int) -> list[list[int]]:
    height, width = grid_shape(grid)
    output = [[0 for _ in range(width)] for _ in range(height)]
    for row in range(height):
        for col in range(width):
            source_row = row - dy
            source_col = col - dx
            if 0 <= source_row < height and 0 <= source_col < width:
                output[row][col] = grid[source_row][source_col]
    return output


class OneStepTranslationRule(BaseRule):
    name = "OneStepTranslationRule"
    priority = 2

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "output_size_differs_from_input_size", {})

        shapes = {grid_shape(case["input"]) for case in cases}
        if len(shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "translation_requires_one_shared_grid_size", {})
        active_height, active_width = next(iter(shapes))

        matching_offsets = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if all(_translated_grid(case["input"], dy, dx) == case["output"] for case in cases):
                    matching_offsets.append((dy, dx))

        if not matching_offsets:
            return RuleResult(self.name, False, "REJECT", "no one-step translation matches all train cases", {})
        dy, dx = matching_offsets[0]
        return RuleResult(
            self.name,
            True,
            "MATCH",
            f"all train cases match one-step translation dy={dy}, dx={dx}",
            {
                "dy": dy,
                "dx": dx,
                "active_height": active_height,
                "active_width": active_width,
            },
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_small_translation_model(
            metadata["dy"],
            metadata["dx"],
            output_path,
            active_height=metadata["active_height"],
            active_width=metadata["active_width"],
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


def _shape_set(cases: list[dict], key: str) -> set[tuple[int, int]]:
    return {grid_shape(case[key]) for case in cases}


def _minimal_axis_period(grid: list[list[int]], axis: str) -> int:
    height, width = grid_shape(grid)
    if axis == "row":
        for period in range(1, height + 1):
            if all(grid[row] == grid[row % period] for row in range(period, height)):
                return period
        return height
    if axis == "col":
        for period in range(1, width + 1):
            valid = True
            for row in range(height):
                for col in range(period, width):
                    if grid[row][col] != grid[row][col % period]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return period
        return width
    raise ValueError(f"unsupported axis: {axis}")


def _padding_coord_for_shape(input_height: int, input_width: int) -> tuple[int, int] | None:
    if input_height < DEFAULT_HEIGHT and input_width < DEFAULT_WIDTH:
        return input_height, input_width
    if input_height < DEFAULT_HEIGHT:
        return input_height, 0
    if input_width < DEFAULT_WIDTH:
        return 0, input_width
    return None


class MirrorRule(BaseRule):
    name = "MirrorRule"
    priority = 3

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "output_size_differs_from_input_size", {})
        shapes = _shape_set(cases, "input")
        if len(shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "mirror_requires_one_shared_grid_size", {})

        modes = []
        if all([list(reversed(row)) for row in case["input"]] == case["output"] for case in cases):
            modes.append("horizontal")
        if all(list(reversed(case["input"])) == case["output"] for case in cases):
            modes.append("vertical")
        if not modes:
            return RuleResult(self.name, False, "REJECT", "no mirror mode matches all train cases", {})
        active_height, active_width = next(iter(shapes))
        return RuleResult(
            self.name,
            True,
            "MATCH",
            f"matched {modes[0]} mirror",
            {
                "mode": modes[0],
                "active_height": active_height,
                "active_width": active_width,
            },
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_mirror_model(
            metadata["mode"],
            output_path,
            active_height=metadata["active_height"],
            active_width=metadata["active_width"],
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


class DynamicActiveMirrorRule(BaseRule):
    name = "DynamicActiveMirrorRule"
    priority = 27

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "dynamic_active_mirror_requires_same_size", {})

        modes = []
        if all([list(reversed(row)) for row in case["input"]] == case["output"] for case in cases):
            modes.append("horizontal")
        if all(list(reversed(case["input"])) == case["output"] for case in cases):
            modes.append("vertical")
        if not modes:
            return RuleResult(self.name, False, "REJECT", "no active mirror mode matches all train cases", {})
        return RuleResult(
            self.name,
            True,
            "MATCH",
            f"matched shape-polymorphic {modes[0]} mirror",
            {"mode": modes[0]},
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_dynamic_active_mirror_model(metadata["mode"], output_path)
        return CandidateModel(task_id, self.name, output_path, metadata)


def _rotate_grid(grid: list[list[int]], k: int) -> list[list[int]]:
    result = grid
    for _ in range(k % 4):
        result = [list(row) for row in zip(*reversed(result))]
    return result


class RotateRule(BaseRule):
    name = "RotateRule"
    priority = 4

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "output_size_differs_from_input_size", {})
        shapes = _shape_set(cases, "input")
        if len(shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "rotate_requires_one_shared_grid_size", {})

        for k in (1, 2, 3):
            if all(_rotate_grid(case["input"], k) == case["output"] for case in cases):
                active_height, active_width = next(iter(shapes))
                return RuleResult(
                    self.name,
                    True,
                    "MATCH",
                    f"matched rotation k={k}",
                    {
                        "k": k,
                        "input_active_height": active_height,
                        "input_active_width": active_width,
                        "output_active_height": active_height,
                        "output_active_width": active_width,
                    },
                )
        return RuleResult(self.name, False, "REJECT", "no rotation matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_rotate_model(
            metadata["k"],
            output_path,
            input_active_height=metadata["input_active_height"],
            input_active_width=metadata["input_active_width"],
            output_active_height=metadata["output_active_height"],
            output_active_width=metadata["output_active_width"],
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


class CropRule(BaseRule):
    name = "CropRule"
    priority = 5

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        input_shapes = _shape_set(cases, "input")
        output_shapes = _shape_set(cases, "output")
        if len(output_shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "crop_requires_one_shared_output_size", {})
        output_height, output_width = next(iter(output_shapes))

        possible_rects: set[tuple[int, int]] | None = None
        for case in cases:
            input_grid = case["input"]
            output_grid = case["output"]
            input_height, input_width = grid_shape(input_grid)
            current: set[tuple[int, int]] = set()
            if output_height <= input_height and output_width <= input_width:
                for top in range(input_height - output_height + 1):
                    for left in range(input_width - output_width + 1):
                        crop = [
                            row[left : left + output_width]
                            for row in input_grid[top : top + output_height]
                        ]
                        if crop == output_grid:
                            current.add((top, left))
            possible_rects = current if possible_rects is None else possible_rects & current
            if not possible_rects:
                return RuleResult(self.name, False, "REJECT", "no fixed crop matches all train cases", {})

        top, left = sorted(possible_rects)[0]
        return RuleResult(
            self.name,
            True,
            "MATCH",
            f"matched fixed crop top={top}, left={left}",
            {
                "top": top,
                "left": left,
                "output_height": output_height,
                "output_width": output_width,
                "input_shape": next(iter(input_shapes)) if len(input_shapes) == 1 else None,
            },
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        rows = list(range(metadata["top"], metadata["top"] + metadata["output_height"]))
        cols = list(range(metadata["left"], metadata["left"] + metadata["output_width"]))
        input_shape = metadata.get("input_shape")
        padding = None if input_shape is None else _padding_coord_for_shape(*input_shape)
        if padding is None:
            build_spatial_remap_model(
                rows,
                cols,
                output_path,
                output_active_height=metadata["output_height"],
                output_active_width=metadata["output_width"],
            )
        else:
            build_spatial_remap_model(
                rows,
                cols,
                output_path,
                pad_row_index=padding[0],
                pad_col_index=padding[1],
            )
        return CandidateModel(task_id, self.name, output_path, metadata)


class ScaleRepeatRule(BaseRule):
    name = "ScaleRepeatRule"
    priority = 6

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        possible_scales: set[tuple[int, int]] | None = None
        for case in cases:
            input_height, input_width = grid_shape(case["input"])
            output_height, output_width = grid_shape(case["output"])
            current: set[tuple[int, int]] = set()
            if output_height % input_height == 0 and output_width % input_width == 0:
                scale_y = output_height // input_height
                scale_x = output_width // input_width
                scaled = [
                    [
                        case["input"][row // scale_y][col // scale_x]
                        for col in range(output_width)
                    ]
                    for row in range(output_height)
                ]
                if scaled == case["output"]:
                    current.add((scale_y, scale_x))
            possible_scales = current if possible_scales is None else possible_scales & current
            if not possible_scales:
                return RuleResult(self.name, False, "REJECT", "no fixed scale repeat matches all train cases", {})

        scale_y, scale_x = sorted(possible_scales)[0]
        if scale_y == 1 and scale_x == 1:
            return RuleResult(self.name, False, "REJECT", "scale factor is identity", {})
        return RuleResult(
            self.name,
            True,
            "MATCH",
            f"matched nearest repeat scale_y={scale_y}, scale_x={scale_x}",
            {"scale_y": scale_y, "scale_x": scale_x},
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_scale_repeat_model(metadata["scale_y"], metadata["scale_x"], output_path)
        return CandidateModel(task_id, self.name, output_path, metadata)


class StridedSubsampleRule(BaseRule):
    name = "StridedSubsampleRule"
    priority = 7

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        input_shapes = _shape_set(cases, "input")
        output_shapes = _shape_set(cases, "output")
        if len(input_shapes) != 1 or len(output_shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "strided_subsample_requires_shared_shapes", {})
        input_height, input_width = next(iter(input_shapes))
        output_height, output_width = next(iter(output_shapes))

        possible: set[tuple[int, int, int, int]] = set()
        for stride_y in range(1, input_height + 1):
            for stride_x in range(1, input_width + 1):
                for offset_y in range(stride_y):
                    for offset_x in range(stride_x):
                        if offset_y + (output_height - 1) * stride_y >= input_height:
                            continue
                        if offset_x + (output_width - 1) * stride_x >= input_width:
                            continue
                        possible.add((offset_y, offset_x, stride_y, stride_x))

        for case in cases:
            current = set()
            for offset_y, offset_x, stride_y, stride_x in possible:
                sampled = [
                    [
                        case["input"][offset_y + row * stride_y][offset_x + col * stride_x]
                        for col in range(output_width)
                    ]
                    for row in range(output_height)
                ]
                if sampled == case["output"]:
                    current.add((offset_y, offset_x, stride_y, stride_x))
            possible &= current
            if not possible:
                return RuleResult(self.name, False, "REJECT", "no fixed strided sample matches all train cases", {})

        offset_y, offset_x, stride_y, stride_x = sorted(possible)[0]
        if stride_y == 1 and stride_x == 1 and offset_y == 0 and offset_x == 0:
            return RuleResult(self.name, False, "REJECT", "strided sample is plain crop", {})
        return RuleResult(
            self.name,
            True,
            "MATCH",
            (
                "matched fixed strided sample "
                f"offset_y={offset_y}, offset_x={offset_x}, "
                f"stride_y={stride_y}, stride_x={stride_x}"
            ),
            {
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width,
                "offset_y": offset_y,
                "offset_x": offset_x,
                "stride_y": stride_y,
                "stride_x": stride_x,
            },
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        rows = [
            metadata["offset_y"] + row * metadata["stride_y"]
            for row in range(metadata["output_height"])
        ]
        cols = [
            metadata["offset_x"] + col * metadata["stride_x"]
            for col in range(metadata["output_width"])
        ]
        padding = _padding_coord_for_shape(metadata["input_height"], metadata["input_width"])
        if padding is None:
            build_spatial_remap_model(
                rows,
                cols,
                output_path,
                output_active_height=metadata["output_height"],
                output_active_width=metadata["output_width"],
            )
        else:
            build_spatial_remap_model(
                rows,
                cols,
                output_path,
                pad_row_index=padding[0],
                pad_col_index=padding[1],
            )
        return CandidateModel(task_id, self.name, output_path, metadata)


class TileRepeatRule(BaseRule):
    name = "TileRepeatRule"
    priority = 8

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        input_shapes = _shape_set(cases, "input")
        output_shapes = _shape_set(cases, "output")
        if len(input_shapes) != 1 or len(output_shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "tile_repeat_requires_shared_shapes", {})
        input_height, input_width = next(iter(input_shapes))
        output_height, output_width = next(iter(output_shapes))
        if output_height % input_height != 0 or output_width % input_width != 0:
            return RuleResult(self.name, False, "REJECT", "output_size_is_not_integer_tile", {})
        tile_y = output_height // input_height
        tile_x = output_width // input_width
        if tile_y == 1 and tile_x == 1:
            return RuleResult(self.name, False, "REJECT", "tile factor is identity", {})

        for case in cases:
            tiled = [
                [case["input"][row % input_height][col % input_width] for col in range(output_width)]
                for row in range(output_height)
            ]
            if tiled != case["output"]:
                return RuleResult(self.name, False, "REJECT", "tile repeat does not match all train cases", {})
        return RuleResult(
            self.name,
            True,
            "MATCH",
            f"matched tile repeat tile_y={tile_y}, tile_x={tile_x}",
            {
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width,
                "tile_y": tile_y,
                "tile_x": tile_x,
            },
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        rows = [row % metadata["input_height"] for row in range(metadata["output_height"])]
        cols = [col % metadata["input_width"] for col in range(metadata["output_width"])]
        padding = _padding_coord_for_shape(metadata["input_height"], metadata["input_width"])
        if padding is None:
            build_spatial_remap_model(
                rows,
                cols,
                output_path,
                output_active_height=metadata["output_height"],
                output_active_width=metadata["output_width"],
            )
        else:
            build_spatial_remap_model(
                rows,
                cols,
                output_path,
                pad_row_index=padding[0],
                pad_col_index=padding[1],
            )
        return CandidateModel(task_id, self.name, output_path, metadata)


class MirrorConcatRule(BaseRule):
    name = "MirrorConcatRule"
    priority = 9

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        input_shapes = _shape_set(cases, "input")
        output_shapes = _shape_set(cases, "output")
        if len(input_shapes) != 1 or len(output_shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "mirror_concat_requires_shared_shapes", {})
        input_height, input_width = next(iter(input_shapes))
        output_height, output_width = next(iter(output_shapes))

        patterns = {
            "h_input_mirror": lambda g: [row + list(reversed(row)) for row in g],
            "h_mirror_input": lambda g: [list(reversed(row)) + row for row in g],
            "v_input_mirror": lambda g: g + list(reversed(g)),
            "v_mirror_input": lambda g: list(reversed(g)) + g,
        }
        for mode, transform in patterns.items():
            if all(transform(case["input"]) == case["output"] for case in cases):
                return RuleResult(
                    self.name,
                    True,
                    "MATCH",
                    f"matched mirror concat {mode}",
                    {
                        "mode": mode,
                        "input_height": input_height,
                        "input_width": input_width,
                        "output_height": output_height,
                        "output_width": output_width,
                    },
                )
        return RuleResult(self.name, False, "REJECT", "no mirror concat pattern matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        input_height = metadata["input_height"]
        input_width = metadata["input_width"]
        mode = metadata["mode"]
        if mode == "h_input_mirror":
            rows = list(range(input_height))
            cols = list(range(input_width)) + list(range(input_width - 1, -1, -1))
        elif mode == "h_mirror_input":
            rows = list(range(input_height))
            cols = list(range(input_width - 1, -1, -1)) + list(range(input_width))
        elif mode == "v_input_mirror":
            rows = list(range(input_height)) + list(range(input_height - 1, -1, -1))
            cols = list(range(input_width))
        else:
            rows = list(range(input_height - 1, -1, -1)) + list(range(input_height))
            cols = list(range(input_width))
        padding = _padding_coord_for_shape(input_height, input_width)
        if padding is None:
            build_spatial_remap_model(
                rows,
                cols,
                output_path,
                output_active_height=metadata["output_height"],
                output_active_width=metadata["output_width"],
            )
        else:
            build_spatial_remap_model(
                rows,
                cols,
                output_path,
                pad_row_index=padding[0],
                pad_col_index=padding[1],
            )
        return CandidateModel(task_id, self.name, output_path, metadata)


class SelfKronMaskRule(BaseRule):
    name = "SelfKronMaskRule"
    priority = 10

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        input_shapes = _shape_set(cases, "input")
        if len(input_shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "self_kron_requires_one_shared_input_shape", {})
        input_height, input_width = next(iter(input_shapes))
        expected_shape = (input_height * input_height, input_width * input_width)
        if any(grid_shape(case["output"]) != expected_shape for case in cases):
            return RuleResult(self.name, False, "REJECT", "output_shape_is_not_self_kron_size", {})

        for case in cases:
            expected = []
            input_grid = case["input"]
            for block_row in range(input_height):
                for local_row in range(input_height):
                    row = []
                    for block_col in range(input_width):
                        if input_grid[block_row][block_col] != 0:
                            row.extend(input_grid[local_row])
                        else:
                            row.extend([0] * input_width)
                    expected.append(row)
            if expected != case["output"]:
                return RuleResult(self.name, False, "REJECT", "self-kron mask does not match all train cases", {})

        return RuleResult(
            self.name,
            True,
            "MATCH",
            "matched output = kron(input != 0, input)",
            {"input_height": input_height, "input_width": input_width},
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_self_kron_mask_model(metadata["input_height"], metadata["input_width"], output_path)
        return CandidateModel(task_id, self.name, output_path, metadata)


PANEL_OPERATIONS = (
    "AND",
    "OR",
    "XOR",
    "LEFT_MINUS_RIGHT",
    "RIGHT_MINUS_LEFT",
    "EQUAL",
    "NOT_EQUAL",
)


def _binary_operation(left: bool, right: bool, operation: str) -> bool:
    if operation == "AND":
        return left and right
    if operation == "OR":
        return left or right
    if operation in {"XOR", "NOT_EQUAL"}:
        return left != right
    if operation == "LEFT_MINUS_RIGHT":
        return left and not right
    if operation == "RIGHT_MINUS_LEFT":
        return right and not left
    if operation == "EQUAL":
        return left == right
    raise ValueError(f"unsupported binary operation: {operation}")


def _panel_layouts(case: dict) -> list[dict[str, int | str]]:
    input_grid = case["input"]
    output_grid = case["output"]
    input_height, input_width = grid_shape(input_grid)
    output_height, output_width = grid_shape(output_grid)
    layouts: list[dict[str, int | str]] = []

    if input_width % 2 == 1:
        panel_width = (input_width - 1) // 2
        separator_col = panel_width
        if output_height == input_height and output_width == panel_width:
            separator_color = input_grid[0][separator_col]
            if all(row[separator_col] == separator_color for row in input_grid):
                layouts.append(
                    {
                        "orientation": "vertical",
                        "panel_height": input_height,
                        "panel_width": panel_width,
                    }
                )

    if input_height % 2 == 1:
        panel_height = (input_height - 1) // 2
        separator_row = panel_height
        if output_height == panel_height and output_width == input_width:
            separator_color = input_grid[separator_row][0]
            if all(color == separator_color for color in input_grid[separator_row]):
                layouts.append(
                    {
                        "orientation": "horizontal",
                        "panel_height": panel_height,
                        "panel_width": input_width,
                    }
                )
    return layouts


def _extract_panels(
    case: dict,
    orientation: str,
    panel_height: int,
    panel_width: int,
) -> tuple[list[list[int]], list[list[int]]] | None:
    input_grid = case["input"]
    output_height, output_width = grid_shape(case["output"])
    if (output_height, output_width) != (panel_height, panel_width):
        return None

    if orientation == "vertical":
        expected_shape = (panel_height, panel_width * 2 + 1)
        if grid_shape(input_grid) != expected_shape:
            return None
        separator_col = panel_width
        separator_color = input_grid[0][separator_col]
        if not all(row[separator_col] == separator_color for row in input_grid):
            return None
        left = [row[:panel_width] for row in input_grid]
        right = [row[panel_width + 1 :] for row in input_grid]
        return left, right

    expected_shape = (panel_height * 2 + 1, panel_width)
    if grid_shape(input_grid) != expected_shape:
        return None
    separator_row = panel_height
    separator_color = input_grid[separator_row][0]
    if not all(color == separator_color for color in input_grid[separator_row]):
        return None
    top = input_grid[:panel_height]
    bottom = input_grid[panel_height + 1 :]
    return top, bottom


class PanelSeparatorBinaryOpRule(BaseRule):
    name = "PanelSeparatorBinaryOpRule"
    priority = 11

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        first_layouts = _panel_layouts(cases[0])
        if not first_layouts:
            return RuleResult(self.name, False, "REJECT", "no complete equal-size panel separator found", {})

        for layout in first_layouts:
            orientation = str(layout["orientation"])
            panel_height = int(layout["panel_height"])
            panel_width = int(layout["panel_width"])
            if panel_width <= 0 or panel_height <= 0:
                continue
            panels = [
                _extract_panels(case, orientation, panel_height, panel_width)
                for case in cases
            ]
            if any(panel is None for panel in panels):
                continue

            for input_false_color in range(10):
                for operation in PANEL_OPERATIONS:
                    color_by_bool: dict[bool, int] = {}
                    valid = True
                    for case, panel_pair in zip(cases, panels):
                        assert panel_pair is not None
                        left, right = panel_pair
                        output_grid = case["output"]
                        for row in range(panel_height):
                            for col in range(panel_width):
                                left_bool = left[row][col] != input_false_color
                                right_bool = right[row][col] != input_false_color
                                value = _binary_operation(left_bool, right_bool, operation)
                                output_color = output_grid[row][col]
                                existing = color_by_bool.get(value)
                                if existing is not None and existing != output_color:
                                    valid = False
                                    break
                                color_by_bool[value] = output_color
                            if not valid:
                                break
                        if not valid:
                            break
                    if (
                        valid
                        and True in color_by_bool
                        and False in color_by_bool
                        and color_by_bool[True] != color_by_bool[False]
                    ):
                        return RuleResult(
                            self.name,
                            True,
                            "MATCH",
                            (
                                f"matched {orientation} panels with {operation} "
                                f"and false input color {input_false_color}"
                            ),
                            {
                                "orientation": orientation,
                                "operation": operation,
                                "panel_height": panel_height,
                                "panel_width": panel_width,
                                "input_false_color": input_false_color,
                                "true_color": color_by_bool[True],
                                "false_color": color_by_bool[False],
                            },
                        )

        return RuleResult(self.name, False, "REJECT", "no binary panel operation matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_panel_binary_op_model(
            metadata["orientation"],
            metadata["operation"],
            metadata["panel_height"],
            metadata["panel_width"],
            metadata["input_false_color"],
            metadata["true_color"],
            metadata["false_color"],
            output_path,
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


GENERALIZED_PANEL_OPERATIONS = (
    "AND",
    "OR",
    "XOR",
    "A-B",
    "B-A",
    "EQUAL",
    "NOT_EQUAL",
    "UNION",
    "INTERSECTION",
    "MAJORITY",
)


def _all_same_color(cells: list[int]) -> int | None:
    if not cells:
        return None
    first = cells[0]
    if all(cell == first for cell in cells):
        return first
    return None


def _is_separator_col(grid: list[list[int]], col: int, sep_width: int, color: int) -> bool:
    height, width = grid_shape(grid)
    if col < 0 or col + sep_width > width:
        return False
    return all(grid[row][current_col] == color for row in range(height) for current_col in range(col, col + sep_width))


def _is_separator_row(grid: list[list[int]], row: int, sep_height: int, color: int) -> bool:
    height, width = grid_shape(grid)
    if row < 0 or row + sep_height > height:
        return False
    return all(grid[current_row][col] == color for current_row in range(row, row + sep_height) for col in range(width))


def _panel_rect(grid: list[list[int]], top: int, left: int, height: int, width: int) -> list[list[int]]:
    return [row[left : left + width] for row in grid[top : top + height]]


def _enumerate_panel_layouts_for_grid(grid: list[list[int]], output_shape: tuple[int, int]) -> list[dict[str, Any]]:
    input_height, input_width = grid_shape(grid)
    output_height, output_width = output_shape
    layouts: list[dict[str, Any]] = []

    for sep_color in range(10):
        for panel_count in range(2, 6):
            for sep_width in range(1, input_width + 1):
                numerator = input_width - (panel_count - 1) * sep_width
                if numerator <= 0 or numerator % panel_count != 0:
                    continue
                panel_width = numerator // panel_count
                if panel_width <= 0:
                    continue
                sep_cols = [panel_width + index * (panel_width + sep_width) for index in range(panel_count - 1)]
                if all(_is_separator_col(grid, col, sep_width, sep_color) for col in sep_cols):
                    specs = [
                        {"top": 0, "left": index * (panel_width + sep_width), "height": input_height, "width": panel_width}
                        for index in range(panel_count)
                    ]
                    if (output_height, output_width) == (input_height, panel_width):
                        layouts.append(
                            {
                                "layout": "vertical",
                                "separator_color": sep_color,
                                "separator_width": sep_width,
                                "panel_specs": specs,
                                "panel_height": input_height,
                                "panel_width": panel_width,
                            }
                        )

        for panel_count in range(2, 6):
            for sep_height in range(1, input_height + 1):
                numerator = input_height - (panel_count - 1) * sep_height
                if numerator <= 0 or numerator % panel_count != 0:
                    continue
                panel_height = numerator // panel_count
                if panel_height <= 0:
                    continue
                sep_rows = [panel_height + index * (panel_height + sep_height) for index in range(panel_count - 1)]
                if all(_is_separator_row(grid, row, sep_height, sep_color) for row in sep_rows):
                    specs = [
                        {"top": index * (panel_height + sep_height), "left": 0, "height": panel_height, "width": input_width}
                        for index in range(panel_count)
                    ]
                    if (output_height, output_width) == (panel_height, input_width):
                        layouts.append(
                            {
                                "layout": "horizontal",
                                "separator_color": sep_color,
                                "separator_height": sep_height,
                                "panel_specs": specs,
                                "panel_height": panel_height,
                                "panel_width": input_width,
                            }
                        )

        for sep_height in range(1, input_height + 1):
            remaining_height = input_height - sep_height
            if remaining_height <= 0 or remaining_height % 2 != 0:
                continue
            panel_height = remaining_height // 2
            sep_row = panel_height
            if not _is_separator_row(grid, sep_row, sep_height, sep_color):
                continue
            for sep_width in range(1, input_width + 1):
                remaining_width = input_width - sep_width
                if remaining_width <= 0 or remaining_width % 2 != 0:
                    continue
                panel_width = remaining_width // 2
                sep_col = panel_width
                if not _is_separator_col(grid, sep_col, sep_width, sep_color):
                    continue
                if (output_height, output_width) != (panel_height, panel_width):
                    continue
                layouts.append(
                    {
                        "layout": "grid_2x2",
                        "separator_color": sep_color,
                        "separator_height": sep_height,
                        "separator_width": sep_width,
                        "panel_specs": [
                            {"top": 0, "left": 0, "height": panel_height, "width": panel_width},
                            {"top": 0, "left": panel_width + sep_width, "height": panel_height, "width": panel_width},
                            {"top": panel_height + sep_height, "left": 0, "height": panel_height, "width": panel_width},
                            {
                                "top": panel_height + sep_height,
                                "left": panel_width + sep_width,
                                "height": panel_height,
                                "width": panel_width,
                            },
                        ],
                        "panel_height": panel_height,
                        "panel_width": panel_width,
                    }
                )
    return layouts


def _same_panel_layout(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return (
        a.get("layout") == b.get("layout")
        and a.get("separator_color") == b.get("separator_color")
        and a.get("separator_height") == b.get("separator_height")
        and a.get("separator_width") == b.get("separator_width")
        and a.get("panel_height") == b.get("panel_height")
        and a.get("panel_width") == b.get("panel_width")
        and a.get("panel_specs") == b.get("panel_specs")
    )


def _panel_bool_value(values: list[bool], operation: str) -> bool:
    if operation == "AND":
        return values[0] and values[1]
    if operation == "OR":
        return values[0] or values[1]
    if operation == "XOR":
        return values[0] != values[1]
    if operation == "A-B":
        return values[0] and not values[1]
    if operation == "B-A":
        return values[1] and not values[0]
    if operation == "EQUAL":
        return values[0] == values[1]
    if operation == "NOT_EQUAL":
        return values[0] != values[1]
    if operation == "UNION":
        return any(values)
    if operation == "INTERSECTION":
        return all(values)
    if operation == "MAJORITY":
        return sum(values) >= (len(values) // 2 + 1)
    raise ValueError(f"unsupported panel operation: {operation}")


class GeneralizedPanelRule(BaseRule):
    name = "GeneralizedPanelRule"
    priority = 10

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        output_shapes = _shape_set(cases, "output")
        if len(output_shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "generalized_panel_requires_shared_output_shape", {})
        output_shape = next(iter(output_shapes))
        first_layouts = _enumerate_panel_layouts_for_grid(cases[0]["input"], output_shape)
        if not first_layouts:
            return RuleResult(self.name, False, "REJECT", "no generalized panel layout found", {})

        for layout in first_layouts:
            per_case_panels: list[list[list[list[int]]]] = []
            valid_layout = True
            for case in cases:
                layouts = _enumerate_panel_layouts_for_grid(case["input"], output_shape)
                if not any(_same_panel_layout(layout, candidate) for candidate in layouts):
                    valid_layout = False
                    break
                per_case_panels.append(
                    [
                        _panel_rect(
                            case["input"],
                            spec["top"],
                            spec["left"],
                            spec["height"],
                            spec["width"],
                        )
                        for spec in layout["panel_specs"]
                    ]
                )
            if not valid_layout:
                continue

            panel_count = len(layout["panel_specs"])
            for panel_index in range(panel_count):
                color_map: dict[int, int] = {}
                valid = True
                for panels, case in zip(per_case_panels, cases):
                    case_map = _infer_color_map_from_pairs(panels[panel_index], case["output"])
                    if case_map is None:
                        valid = False
                        break
                    for old_color, new_color in case_map.items():
                        existing = color_map.get(old_color)
                        if existing is not None and existing != new_color:
                            valid = False
                            break
                        color_map[old_color] = new_color
                    if not valid:
                        break
                if valid:
                    metadata = dict(layout)
                    metadata.update({"mode": "panel", "panel_index": panel_index, "color_map": color_map})
                    return RuleResult(
                        self.name,
                        True,
                        "MATCH",
                        f"matched output panel {panel_index} from {layout['layout']} layout",
                        metadata,
                    )

            for input_false_color in range(10):
                for operation in GENERALIZED_PANEL_OPERATIONS:
                    if operation in {"AND", "OR", "XOR", "A-B", "B-A", "EQUAL", "NOT_EQUAL"} and panel_count < 2:
                        continue
                    if operation in {"XOR", "A-B", "B-A", "EQUAL", "NOT_EQUAL"} and panel_count != 2:
                        continue
                    if operation in {"UNION", "INTERSECTION", "MAJORITY"} and panel_count < 2:
                        continue
                    color_by_bool: dict[bool, int] = {}
                    valid = True
                    for panels, case in zip(per_case_panels, cases):
                        for row in range(layout["panel_height"]):
                            for col in range(layout["panel_width"]):
                                values = [panel[row][col] != input_false_color for panel in panels]
                                bool_value = _panel_bool_value(values, operation)
                                output_color = case["output"][row][col]
                                existing = color_by_bool.get(bool_value)
                                if existing is not None and existing != output_color:
                                    valid = False
                                    break
                                color_by_bool[bool_value] = output_color
                            if not valid:
                                break
                        if not valid:
                            break
                    if valid and True in color_by_bool and False in color_by_bool and color_by_bool[True] != color_by_bool[False]:
                        metadata = dict(layout)
                        metadata.update(
                            {
                                "mode": "bool_op",
                                "operation": operation,
                                "input_false_color": input_false_color,
                                "true_color": color_by_bool[True],
                                "false_color": color_by_bool[False],
                            }
                        )
                        return RuleResult(
                            self.name,
                            True,
                            "MATCH",
                            f"matched generalized panel {operation} with false input color {input_false_color}",
                            metadata,
                        )

        return RuleResult(self.name, False, "REJECT", "no generalized panel extraction or operation matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        if metadata["mode"] == "panel":
            spec = metadata["panel_specs"][metadata["panel_index"]]
            rows = list(range(spec["top"], spec["top"] + spec["height"]))
            cols = list(range(spec["left"], spec["left"] + spec["width"]))
            build_spatial_remap_model(
                rows,
                cols,
                output_path,
                output_active_height=spec["height"],
                output_active_width=spec["width"],
                color_map=metadata["color_map"],
            )
        else:
            build_generalized_panel_op_model(
                metadata["panel_specs"],
                metadata["operation"],
                metadata["input_false_color"],
                metadata["true_color"],
                metadata["false_color"],
                output_path,
            )
        return CandidateModel(task_id, self.name, output_path, metadata)


def _panel_non_background_count(panel: list[list[int]], background_color: int) -> int:
    return sum(color != background_color for row in panel for color in row)


def _unique_panel_colors(panels: list[list[list[int]]], panel_index: int) -> set[int]:
    selected_colors = {color for row in panels[panel_index] for color in row}
    other_colors = {
        color
        for index, panel in enumerate(panels)
        if index != panel_index
        for row in panel
        for color in row
    }
    return selected_colors - other_colors


def _mirror_grid(grid: list[list[int]], mode: str) -> list[list[int]]:
    if mode == "horizontal":
        return [list(reversed(row)) for row in grid]
    if mode == "vertical":
        return list(reversed([row[:] for row in grid]))
    raise ValueError(f"unsupported mirror mode: {mode}")


def _rotate_grid(grid: list[list[int]], k: int) -> list[list[int]]:
    if k == 0:
        return [row[:] for row in grid]
    if k == 1:
        return [list(row) for row in zip(*reversed(grid))]
    if k == 2:
        return [list(reversed(row)) for row in reversed(grid)]
    if k == 3:
        return [list(row) for row in reversed(list(zip(*grid)))]
    raise ValueError(f"unsupported rotation k: {k}")


def _trim_background(grid: list[list[int]], background_color: int) -> list[list[int]] | None:
    colors = {color for row in grid for color in row if color != background_color}
    if not colors:
        return None
    bbox = _bbox_for_colors(grid, colors)
    if bbox is None:
        return None
    top, left, height, width = bbox
    return _crop_grid(grid, top, left, height, width)


def _panel_layouts_any_shape(grid: list[list[int]]) -> list[dict[str, Any]]:
    input_height, input_width = grid_shape(grid)
    output_shapes: set[tuple[int, int]] = set()
    for panel_count in range(2, 6):
        for sep_width in range(1, input_width + 1):
            numerator = input_width - (panel_count - 1) * sep_width
            if numerator > 0 and numerator % panel_count == 0:
                output_shapes.add((input_height, numerator // panel_count))
        for sep_height in range(1, input_height + 1):
            numerator = input_height - (panel_count - 1) * sep_height
            if numerator > 0 and numerator % panel_count == 0:
                output_shapes.add((numerator // panel_count, input_width))
    for sep_height in range(1, input_height + 1):
        remaining_height = input_height - sep_height
        if remaining_height <= 0 or remaining_height % 2 != 0:
            continue
        for sep_width in range(1, input_width + 1):
            remaining_width = input_width - sep_width
            if remaining_width > 0 and remaining_width % 2 == 0:
                output_shapes.add((remaining_height // 2, remaining_width // 2))

    layouts: list[dict[str, Any]] = []
    for output_shape in sorted(output_shapes):
        layouts.extend(_enumerate_panel_layouts_for_grid(grid, output_shape))
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for layout in layouts:
        key = repr(layout)
        if key not in seen:
            seen.add(key)
            unique.append(layout)
    return unique


def _transformed_panel_candidates(
    panel: list[list[int]],
    output_grid: list[list[int]],
    separator_color: int,
) -> list[dict[str, Any]]:
    transforms: list[tuple[str, list[list[int]]]] = [
        ("identity", panel),
        ("crop_background", _trim_background(panel, separator_color) or []),
        ("mirror_horizontal", _mirror_grid(panel, "horizontal")),
        ("mirror_vertical", _mirror_grid(panel, "vertical")),
        ("rotate_90", _rotate_grid(panel, 1)),
        ("rotate_180", _rotate_grid(panel, 2)),
        ("rotate_270", _rotate_grid(panel, 3)),
    ]
    matches: list[dict[str, Any]] = []
    for transform_name, transformed in transforms:
        if not transformed:
            continue
        color_map = _infer_color_map_from_pairs(transformed, output_grid)
        if color_map is None:
            continue
        matches.append(
            {
                "transform": transform_name,
                "source": transformed,
                "color_map": color_map,
            }
        )
    return matches


class PanelSemanticRule(BaseRule):
    """Probe variable-layout panel selection semantics.

    This rule is intentionally probe-first. Dynamic panel selection can require
    runtime layout/index inference, so formal candidates are emitted only by
    older conservative panel rules.
    """

    name = "PanelSemanticRule"
    priority = 21

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        candidates: dict[tuple[str, str], dict[str, Any]] | None = None
        for case in cases:
            case_candidates: dict[tuple[str, str], dict[str, Any]] = {}
            for layout in _panel_layouts_any_shape(case["input"]):
                panels = [
                    _panel_rect(
                        case["input"],
                        int(spec["top"]),
                        int(spec["left"]),
                        int(spec["height"]),
                        int(spec["width"]),
                    )
                    for spec in layout["panel_specs"]
                ]
                background_color = int(layout["separator_color"])
                non_background_counts = [
                    _panel_non_background_count(panel, background_color)
                    for panel in panels
                ]
                panel_shapes = [grid_shape(panel) for panel in panels]
                for panel_index, panel in enumerate(panels):
                    selectors: list[str] = []
                    for unique_color in sorted(_unique_panel_colors(panels, panel_index)):
                        selectors.append(f"select_panel_by_unique_color_{unique_color}")
                        selectors.append(f"select_panel_by_color_absent_from_others_{unique_color}")
                    selected_count = non_background_counts[panel_index]
                    if non_background_counts.count(selected_count) == 1:
                        if selected_count == max(non_background_counts):
                            selectors.append("select_panel_by_most_non_background")
                        if selected_count == min(non_background_counts):
                            selectors.append("select_panel_by_least_non_background")
                    if panel_shapes.count(panel_shapes[panel_index]) == 1:
                        selectors.append("select_panel_by_different_shape")
                    selectors.append("select_panel_by_matching_output_after_colormap")
                    for transformed in _transformed_panel_candidates(panel, case["output"], background_color):
                        selector_names = selectors[:]
                        if transformed["transform"] == "crop_background":
                            selector_names.append("select_panel_by_matching_output_after_crop")
                        if transformed["transform"].startswith(("rotate", "mirror")):
                            selector_names.append("select_panel_by_matching_output_after_rotate_or_mirror")
                        for selector in selector_names:
                            key = (selector, transformed["transform"])
                            case_candidates[key] = {
                                "selector": selector,
                                "transform": transformed["transform"],
                                "layout": layout["layout"],
                                "panel_index": panel_index,
                                "color_map": transformed["color_map"],
                                "builder_available": False,
                                "blocked_reason": "builder_missing_dynamic_panel_select",
                            }
            if candidates is None:
                candidates = case_candidates
            else:
                merged: dict[tuple[str, str], dict[str, Any]] = {}
                for key, candidate in candidates.items():
                    current = case_candidates.get(key)
                    if current is None:
                        continue
                    color_map = dict(candidate["color_map"])
                    compatible = True
                    for old_color, new_color in current["color_map"].items():
                        existing = color_map.get(old_color)
                        if existing is not None and existing != new_color:
                            compatible = False
                            break
                        color_map[old_color] = new_color
                    if compatible:
                        merged[key] = {**candidate, "color_map": color_map}
                candidates = merged
            if not candidates:
                return RuleResult(self.name, False, "REJECT", "no panel semantic selector matches all train cases", {})

        best = sorted(candidates.values(), key=lambda item: (item["selector"], item["transform"]))[0]
        return RuleResult(
            self.name,
            True,
            "MATCH",
            f"matched panel semantic selector={best['selector']} transform={best['transform']}",
            best,
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        raise NotImplementedError("PanelSemanticRule is probe-only until dynamic panel selection is compiled")


def _center_cross_quadrants(grid: list[list[int]]) -> list[list[list[int]]] | None:
    height, width = grid_shape(grid)
    if height != width or height < 3 or height % 2 == 0:
        return None
    panel_size = height // 2
    center = panel_size
    separator_color = grid[center][center]
    if any(grid[center][col] != separator_color for col in range(width)):
        return None
    if any(grid[row][center] != separator_color for row in range(height)):
        return None
    return [
        _crop_grid(grid, 0, 0, panel_size, panel_size),
        _crop_grid(grid, 0, center + 1, panel_size, panel_size),
        _crop_grid(grid, center + 1, 0, panel_size, panel_size),
        _crop_grid(grid, center + 1, center + 1, panel_size, panel_size),
    ]


def _panel_difference_score(panel: list[list[int]], other: list[list[int]]) -> int:
    height, width = grid_shape(panel)
    if grid_shape(other) != (height, width):
        raise ValueError("panel shapes must match")
    return sum(
        1
        for row in range(height)
        for col in range(width)
        if panel[row][col] != other[row][col]
    )


def _unique_max_difference_panel(panels: list[list[list[int]]]) -> int | None:
    scores = [
        sum(_panel_difference_score(panel, other) for other_index, other in enumerate(panels) if other_index != panel_index)
        for panel_index, panel in enumerate(panels)
    ]
    max_score = max(scores)
    if max_score <= 0 or scores.count(max_score) != 1:
        return None
    return scores.index(max_score)


class DynamicQuadrantPanelSelectRule(BaseRule):
    """Select the unique-pattern quadrant from a 2x2 center-cross panel grid."""

    name = "DynamicQuadrantPanelSelectRule"
    priority = 27

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        color_map: dict[int, int] = {}
        for case in cases:
            panels = _center_cross_quadrants(case["input"])
            if panels is None:
                return RuleResult(self.name, False, "REJECT", "input is not an odd square 2x2 center-cross panel grid", {})
            panel_size, _ = grid_shape(panels[0])
            if grid_shape(case["output"]) != (panel_size, panel_size):
                return RuleResult(self.name, False, "REJECT", "output shape does not match quadrant shape", {})
            selected_index = _unique_max_difference_panel(panels)
            if selected_index is None:
                return RuleResult(self.name, False, "REJECT", "no unique-pattern quadrant", {})
            current_map = _infer_color_map_from_pairs(panels[selected_index], case["output"])
            if current_map is None:
                return RuleResult(self.name, False, "REJECT", "selected quadrant does not match output under a color map", {})
            for old_color, new_color in current_map.items():
                existing = color_map.get(old_color)
                if existing is not None and existing != new_color:
                    return RuleResult(self.name, False, "REJECT", "inconsistent color map across train cases", {})
                color_map[old_color] = new_color

        return RuleResult(
            self.name,
            True,
            "MATCH",
            "matched dynamic 2x2 center-cross unique-pattern panel selection",
            {
                "layout": "center_cross_2x2",
                "selector": "unique_max_panel_difference",
                "transform": "identity",
                "color_map": color_map,
            },
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_dynamic_quadrant_panel_select_model(metadata["color_map"], output_path)
        return CandidateModel(task_id, self.name, output_path, metadata)


class PanelSelectByColorRule(BaseRule):
    name = "PanelSelectByColorRule"
    priority = 13

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        output_shapes = _shape_set(cases, "output")
        if len(output_shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "panel_select_requires_shared_output_shape", {})
        output_shape = next(iter(output_shapes))

        candidates: list[dict[str, Any]] | None = None
        for case in cases:
            current: list[dict[str, Any]] = []
            for layout in _enumerate_panel_layouts_for_grid(case["input"], output_shape):
                panels = [
                    _panel_rect(
                        case["input"],
                        int(spec["top"]),
                        int(spec["left"]),
                        int(spec["height"]),
                        int(spec["width"]),
                    )
                    for spec in layout["panel_specs"]
                ]
                for panel_index, panel in enumerate(panels):
                    color_map = _infer_color_map_from_pairs(panel, case["output"])
                    if color_map is None:
                        continue
                    background_color = int(layout["separator_color"])
                    non_background_counts = [
                        _panel_non_background_count(candidate_panel, background_color)
                        for candidate_panel in panels
                    ]
                    unique_colors = _unique_panel_colors(panels, panel_index)
                    selectors: list[str] = []
                    if unique_colors:
                        for unique_color in sorted(unique_colors):
                            selectors.append(f"contains_unique_color_{unique_color}")
                    selected_count = non_background_counts[panel_index]
                    if non_background_counts.count(selected_count) == 1:
                        if selected_count == max(non_background_counts):
                            selectors.append("most_non_background")
                        if selected_count == min(non_background_counts):
                            selectors.append("least_non_background")
                    for selector in selectors:
                        current.append(
                            {
                                "layout": layout,
                                "panel_index": panel_index,
                                "selector": selector,
                                "color_map": color_map,
                            }
                        )
            if candidates is None:
                candidates = current
            else:
                merged: list[dict[str, Any]] = []
                for candidate in candidates:
                    for item in current:
                        if (
                            _same_panel_layout(candidate["layout"], item["layout"])
                            and candidate["panel_index"] == item["panel_index"]
                            and candidate["selector"] == item["selector"]
                        ):
                            color_map = dict(candidate["color_map"])
                            compatible = True
                            for old_color, new_color in item["color_map"].items():
                                existing = color_map.get(old_color)
                                if existing is not None and existing != new_color:
                                    compatible = False
                                    break
                                color_map[old_color] = new_color
                            if compatible:
                                merged.append({**candidate, "color_map": color_map})
                candidates = merged
            if not candidates:
                return RuleResult(self.name, False, "REJECT", "no panel selection matches all train cases", {})

        best = sorted(
            candidates or [],
            key=lambda item: (
                str(item["selector"]),
                int(item["panel_index"]),
            ),
        )[0]
        layout = best["layout"]
        spec = layout["panel_specs"][best["panel_index"]]
        return RuleResult(
            self.name,
            True,
            "MATCH",
            (
                f"matched panel selection index={best['panel_index']} "
                f"selector={best['selector']}"
            ),
            {
                "layout": layout["layout"],
                "separator_color": layout["separator_color"],
                "panel_index": best["panel_index"],
                "selector": best["selector"],
                "top": int(spec["top"]),
                "left": int(spec["left"]),
                "output_height": int(spec["height"]),
                "output_width": int(spec["width"]),
                "color_map": best["color_map"],
            },
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        rows = list(range(metadata["top"], metadata["top"] + metadata["output_height"]))
        cols = list(range(metadata["left"], metadata["left"] + metadata["output_width"]))
        build_spatial_remap_model(
            rows,
            cols,
            output_path,
            output_active_height=metadata["output_height"],
            output_active_width=metadata["output_width"],
            color_map=metadata["color_map"],
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


class PeriodicExtensionColorMapRule(BaseRule):
    name = "PeriodicExtensionColorMapRule"
    priority = 12

    def _match_auto_axis(
        self,
        cases: list[dict],
        input_height: int,
        input_width: int,
        output_height: int,
        output_width: int,
        axis: str,
    ) -> RuleResult | None:
        if axis == "row":
            if output_height <= input_height or output_width != input_width:
                return None
        elif axis == "col":
            if output_width <= input_width or output_height != input_height:
                return None
        else:
            raise ValueError(f"unsupported axis: {axis}")

        color_map: dict[int, int] = {}
        periods: list[int] = []
        for case_index, case in enumerate(cases):
            input_grid = case["input"]
            output_grid = case["output"]
            period = _minimal_axis_period(input_grid, axis)
            periods.append(period)
            for row in range(output_height):
                for col in range(output_width):
                    if axis == "row":
                        old_color = input_grid[row % period][col]
                    else:
                        old_color = input_grid[row][col % period]
                    new_color = output_grid[row][col]
                    existing = color_map.get(old_color)
                    if existing is not None and existing != new_color:
                        return None
                    color_map[old_color] = new_color

        if len(set(periods)) <= 1:
            return None
        return RuleResult(
            self.name,
            True,
            "MATCH",
            f"matched auto {axis} periodic extension periods={periods}",
            {
                "mode": f"auto_{axis}_period",
                "axis": axis,
                "input_height": input_height,
                "input_width": input_width,
                "output_height": output_height,
                "output_width": output_width,
                "periods": periods,
                "color_map": color_map,
            },
        )

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        input_shapes = _shape_set(cases, "input")
        output_shapes = _shape_set(cases, "output")
        if len(input_shapes) != 1 or len(output_shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "periodic_extension_requires_shared_shapes", {})
        input_height, input_width = next(iter(input_shapes))
        output_height, output_width = next(iter(output_shapes))
        if output_height < input_height or output_width < input_width:
            return RuleResult(self.name, False, "REJECT", "periodic_extension_requires_non_shrinking_output", {})
        if output_height == input_height and output_width == input_width:
            return RuleResult(self.name, False, "REJECT", "periodic_extension_output_size_is_identity", {})

        for period_y in range(1, input_height + 1):
            for period_x in range(1, input_width + 1):
                color_map: dict[int, int] = {}
                valid = True
                for case in cases:
                    input_grid = case["input"]
                    output_grid = case["output"]
                    for row in range(output_height):
                        for col in range(output_width):
                            old_color = input_grid[row % period_y][col % period_x]
                            new_color = output_grid[row][col]
                            existing = color_map.get(old_color)
                            if existing is not None and existing != new_color:
                                valid = False
                                break
                            color_map[old_color] = new_color
                        if not valid:
                            break
                    if not valid:
                        break
                if valid:
                    return RuleResult(
                        self.name,
                        True,
                        "MATCH",
                        f"matched periodic extension period_y={period_y}, period_x={period_x}",
                        {
                            "period_y": period_y,
                            "period_x": period_x,
                            "input_height": input_height,
                            "input_width": input_width,
                            "output_height": output_height,
                            "output_width": output_width,
                            "color_map": color_map,
                        },
                    )

        for axis in ("row", "col"):
            auto_result = self._match_auto_axis(
                cases,
                input_height,
                input_width,
                output_height,
                output_width,
                axis,
            )
            if auto_result is not None:
                return auto_result

        return RuleResult(self.name, False, "REJECT", "no periodic extension color map matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        if metadata.get("mode") in {"auto_row_period", "auto_col_period"}:
            build_auto_periodic_extension_color_map_model(
                metadata["axis"],
                metadata["input_height"],
                metadata["input_width"],
                metadata["output_height"],
                metadata["output_width"],
                metadata["color_map"],
                output_path,
            )
        else:
            build_periodic_extension_color_map_model(
                metadata["period_y"],
                metadata["period_x"],
                metadata["input_height"],
                metadata["input_width"],
                metadata["output_height"],
                metadata["output_width"],
                metadata["color_map"],
                output_path,
            )
        return CandidateModel(task_id, self.name, output_path, metadata)


LOCAL_OFFSET_SETS: dict[str, list[tuple[int, int]]] = {
    "all8": [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ],
    "cardinal4": [(-1, 0), (0, -1), (0, 1), (1, 0)],
    "diagonal4": [(-1, -1), (-1, 1), (1, -1), (1, 1)],
    "horizontal2": [(0, -1), (0, 1)],
    "vertical2": [(-1, 0), (1, 0)],
    "diag_main2": [(-1, -1), (1, 1)],
    "diag_anti2": [(-1, 1), (1, -1)],
    "all24": [
        (row_offset, col_offset)
        for row_offset in range(-2, 3)
        for col_offset in range(-2, 3)
        if (row_offset, col_offset) != (0, 0)
    ],
    "cardinal8": [
        (-2, 0),
        (-1, 0),
        (0, -2),
        (0, -1),
        (0, 1),
        (0, 2),
        (1, 0),
        (2, 0),
    ],
    "diagonal8": [
        (-2, -2),
        (-1, -1),
        (-2, 2),
        (-1, 1),
        (1, -1),
        (2, -2),
        (1, 1),
        (2, 2),
    ],
}


def _local_source_colors(source_mode: str, background_color: int) -> list[int]:
    if source_mode == "non_background":
        return [color for color in range(10) if color != background_color]
    if source_mode.startswith("color_"):
        return [int(source_mode.removeprefix("color_"))]
    raise ValueError(f"unsupported source mode: {source_mode}")


def _local_fill_transform(
    grid: list[list[int]],
    background_color: int,
    fill_color: int,
    source_colors: list[int],
    offsets: list[tuple[int, int]],
    condition: str,
    threshold: int,
) -> list[list[int]]:
    height, width = grid_shape(grid)
    source_color_set = set(source_colors)
    output = [row[:] for row in grid]
    for row in range(height):
        for col in range(width):
            if grid[row][col] != background_color:
                continue
            count = 0
            for row_offset, col_offset in offsets:
                neighbor_row = row + row_offset
                neighbor_col = col + col_offset
                if 0 <= neighbor_row < height and 0 <= neighbor_col < width:
                    if grid[neighbor_row][neighbor_col] in source_color_set:
                        count += 1
            if condition == "eq":
                should_fill = count == threshold
            elif condition == "ge":
                should_fill = count >= threshold
            else:
                raise ValueError(f"unsupported condition: {condition}")
            if should_fill:
                output[row][col] = fill_color
    return output


def _hole_fill_transform(
    grid: list[list[int]],
    background_color: int,
    fill_color: int,
) -> list[list[int]]:
    height, width = grid_shape(grid)
    reachable: set[tuple[int, int]] = set()
    stack: list[tuple[int, int]] = []
    for row in range(height):
        for col in range(width):
            if row not in {0, height - 1} and col not in {0, width - 1}:
                continue
            if grid[row][col] == background_color:
                reachable.add((row, col))
                stack.append((row, col))
    while stack:
        row, col = stack.pop()
        for row_delta, col_delta in ((-1, 0), (0, -1), (0, 1), (1, 0)):
            next_row = row + row_delta
            next_col = col + col_delta
            if not (0 <= next_row < height and 0 <= next_col < width):
                continue
            coord = (next_row, next_col)
            if coord in reachable or grid[next_row][next_col] != background_color:
                continue
            reachable.add(coord)
            stack.append(coord)

    output = [row[:] for row in grid]
    for row in range(height):
        for col in range(width):
            if grid[row][col] == background_color and (row, col) not in reachable:
                output[row][col] = fill_color
    return output


class HoleFillRule(BaseRule):
    name = "HoleFillRule"
    priority = 14

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "hole_fill_requires_same_input_output_size", {})

        for background_color in range(10):
            for fill_color in range(10):
                if fill_color == background_color:
                    continue
                changed_total = 0
                valid_recolor = True
                for case in cases:
                    for row_index, row in enumerate(case["input"]):
                        for col_index, old_color in enumerate(row):
                            new_color = case["output"][row_index][col_index]
                            if old_color == new_color:
                                continue
                            if old_color == background_color and new_color == fill_color:
                                changed_total += 1
                                continue
                            valid_recolor = False
                            break
                        if not valid_recolor:
                            break
                    if not valid_recolor:
                        break
                if not valid_recolor or changed_total == 0:
                    continue
                if all(
                    _hole_fill_transform(case["input"], background_color, fill_color) == case["output"]
                    for case in cases
                ):
                    return RuleResult(
                        self.name,
                        True,
                        "MATCH",
                        f"matched hole fill background={background_color}, fill={fill_color}",
                        {
                            "background_color": background_color,
                            "fill_color": fill_color,
                        },
                    )
        return RuleResult(self.name, False, "REJECT", "no hole fill matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_hole_fill_model(
            metadata["background_color"],
            metadata["fill_color"],
            output_path,
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


class LocalNeighborhoodFillRule(BaseRule):
    name = "LocalNeighborhoodFillRule"
    priority = 13

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "local_fill_requires_same_input_output_size", {})

        for background_color in range(10):
            for fill_color in range(10):
                if fill_color == background_color:
                    continue
                changed_total = 0
                valid_recolor = True
                for case in cases:
                    input_grid = case["input"]
                    output_grid = case["output"]
                    case_changed = 0
                    for row_index, row in enumerate(input_grid):
                        for col_index, old_color in enumerate(row):
                            new_color = output_grid[row_index][col_index]
                            if old_color == new_color:
                                continue
                            if old_color == background_color and new_color == fill_color:
                                case_changed += 1
                                continue
                            valid_recolor = False
                            break
                        if not valid_recolor:
                            break
                    if not valid_recolor:
                        break
                    changed_total += case_changed
                if not valid_recolor or changed_total == 0:
                    continue

                source_modes = ["non_background"] + [f"color_{color}" for color in range(10)]
                for source_mode in source_modes:
                    source_colors = _local_source_colors(source_mode, background_color)
                    for offset_name, offsets in LOCAL_OFFSET_SETS.items():
                        for condition in ("eq", "ge"):
                            for threshold in range(1, len(offsets) + 1):
                                if all(
                                    _local_fill_transform(
                                        case["input"],
                                        background_color,
                                        fill_color,
                                        source_colors,
                                        offsets,
                                        condition,
                                        threshold,
                                    )
                                    == case["output"]
                                    for case in cases
                                ):
                                    return RuleResult(
                                        self.name,
                                        True,
                                        "MATCH",
                                        (
                                            "matched local background fill "
                                            f"bg={background_color}, fill={fill_color}, "
                                            f"source={source_mode}, offsets={offset_name}, "
                                            f"{condition}{threshold}"
                                        ),
                                        {
                                            "background_color": background_color,
                                            "fill_color": fill_color,
                                            "source_mode": source_mode,
                                            "source_colors": source_colors,
                                            "offset_name": offset_name,
                                            "offsets": offsets,
                                            "condition": condition,
                                            "threshold": threshold,
                                        },
                                    )

        return RuleResult(self.name, False, "REJECT", "no local neighborhood fill matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_local_neighborhood_fill_model(
            metadata["background_color"],
            metadata["fill_color"],
            metadata["source_colors"],
            metadata["offsets"],
            metadata["condition"],
            metadata["threshold"],
            output_path,
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


def _apply_color_map(grid: list[list[int]], color_map: dict[int, int]) -> list[list[int]]:
    return [[int(color_map.get(color, color)) for color in row] for row in grid]


def _infer_color_map_from_pairs(
    source_grid: list[list[int]],
    output_grid: list[list[int]],
) -> dict[int, int] | None:
    if grid_shape(source_grid) != grid_shape(output_grid):
        return None
    color_map: dict[int, int] = {}
    for row_index, row in enumerate(source_grid):
        for col_index, old_color in enumerate(row):
            new_color = output_grid[row_index][col_index]
            existing = color_map.get(old_color)
            if existing is not None and existing != new_color:
                return None
            color_map[old_color] = new_color
    return color_map


def _crop_grid(grid: list[list[int]], top: int, left: int, height: int, width: int) -> list[list[int]]:
    return [row[left : left + width] for row in grid[top : top + height]]


def _is_identity_observed_color_map(color_map: dict[int, int]) -> bool:
    return all(old_color == new_color for old_color, new_color in color_map.items())


def _component_bboxes(grid: list[list[int]], background_color: int) -> list[tuple[int, int, int, int, int]]:
    height, width = grid_shape(grid)
    seen: set[tuple[int, int]] = set()
    components: list[tuple[int, int, int, int, int]] = []
    for row in range(height):
        for col in range(width):
            color = grid[row][col]
            if color == background_color or (row, col) in seen:
                continue
            stack = [(row, col)]
            seen.add((row, col))
            coords: list[tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                coords.append((current_row, current_col))
                for row_delta, col_delta in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                    next_row = current_row + row_delta
                    next_col = current_col + col_delta
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in seen or grid[next_row][next_col] != color:
                        continue
                    seen.add((next_row, next_col))
                    stack.append((next_row, next_col))
            min_row = min(coord[0] for coord in coords)
            max_row = max(coord[0] for coord in coords)
            min_col = min(coord[1] for coord in coords)
            max_col = max(coord[1] for coord in coords)
            components.append((min_row, min_col, max_row - min_row + 1, max_col - min_col + 1, color))
    return components


def _substructure_windows_for_case(
    case: dict,
    output_height: int,
    output_width: int,
) -> list[dict[str, Any]]:
    grid = case["input"]
    input_height, input_width = grid_shape(grid)
    windows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int, int, int]] = set()

    def add(kind: str, top: int, left: int, height: int, width: int, color: int = -1) -> None:
        if height != output_height or width != output_width:
            return
        if top < 0 or left < 0 or top + height > input_height or left + width > input_width:
            return
        key = (kind, top, left, height, width, color)
        if key in seen:
            return
        seen.add(key)
        windows.append({"kind": kind, "top": top, "left": left, "height": height, "width": width, "color": color})

    if output_height <= input_height and output_width <= input_width:
        for top in range(input_height - output_height + 1):
            for left in range(input_width - output_width + 1):
                add("input_crop", top, left, output_height, output_width)

    input_colors = {cell for row in grid for cell in row}
    for background_color in range(10):
        colors = {color for color in input_colors if color != background_color}
        bbox = _bbox_for_colors(grid, colors)
        if bbox is not None:
            add("non_background_bbox", bbox[0], bbox[1], bbox[2], bbox[3], background_color)
        for component_bbox in _component_bboxes(grid, background_color):
            add("component_bbox", component_bbox[0], component_bbox[1], component_bbox[2], component_bbox[3], component_bbox[4])

    for color in range(10):
        bbox = _bbox_for_colors(grid, {color})
        if bbox is not None:
            add("color_bbox", bbox[0], bbox[1], bbox[2], bbox[3], color)

    for layout in _enumerate_panel_layouts_for_grid(grid, (output_height, output_width)):
        for panel_index, spec in enumerate(layout["panel_specs"]):
            add(f"panel_{layout['layout']}_{panel_index}", spec["top"], spec["left"], spec["height"], spec["width"])

    return windows


class SubstructureExtractRule(BaseRule):
    name = "SubstructureExtractRule"
    priority = 14

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        output_shapes = _shape_set(cases, "output")
        if len(output_shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "substructure_extract_requires_shared_output_size", {})
        output_height, output_width = next(iter(output_shapes))

        possible_windows: set[tuple[str, int, int, int, int, int]] | None = None
        for case in cases:
            current = {
                (
                    str(window["kind"]),
                    int(window["top"]),
                    int(window["left"]),
                    int(window["height"]),
                    int(window["width"]),
                    int(window["color"]),
                )
                for window in _substructure_windows_for_case(case, output_height, output_width)
            }
            possible_windows = current if possible_windows is None else possible_windows & current
        if not possible_windows:
            return RuleResult(self.name, False, "REJECT", "no shared substructure window can cover all train cases", {})

        for kind, top, left, height, width, color in sorted(possible_windows):
            color_map: dict[int, int] = {}
            valid = True
            for case in cases:
                crop = _crop_grid(case["input"], top, left, height, width)
                case_map = _infer_color_map_from_pairs(crop, case["output"])
                if case_map is None:
                    valid = False
                    break
                for old_color, new_color in case_map.items():
                    existing = color_map.get(old_color)
                    if existing is not None and existing != new_color:
                        valid = False
                        break
                    color_map[old_color] = new_color
                if not valid:
                    break
            if valid:
                if top == 0 and left == 0 and _is_identity_observed_color_map(color_map):
                    continue
                return RuleResult(
                    self.name,
                    True,
                    "MATCH",
                    f"matched {kind} top={top}, left={left} with optional color map",
                    {
                        "kind": kind,
                        "top": top,
                        "left": left,
                        "output_height": height,
                        "output_width": width,
                        "color": color,
                        "color_map": color_map,
                    },
                )

        return RuleResult(self.name, False, "REJECT", "no substructure color map matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        rows = list(range(metadata["top"], metadata["top"] + metadata["output_height"]))
        cols = list(range(metadata["left"], metadata["left"] + metadata["output_width"]))
        build_spatial_remap_model(
            rows,
            cols,
            output_path,
            output_active_height=metadata["output_height"],
            output_active_width=metadata["output_width"],
            color_map=metadata["color_map"],
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


def _bbox_for_colors(grid: list[list[int]], colors: set[int]) -> tuple[int, int, int, int] | None:
    coords = [
        (row_index, col_index)
        for row_index, row in enumerate(grid)
        for col_index, color in enumerate(row)
        if color in colors
    ]
    if not coords:
        return None
    min_row = min(row for row, _ in coords)
    max_row = max(row for row, _ in coords)
    min_col = min(col for _, col in coords)
    max_col = max(col for _, col in coords)
    return min_row, min_col, max_row - min_row + 1, max_col - min_col + 1


def _tile_from_crop(
    grid: list[list[int]],
    top: int,
    left: int,
    tile_height: int,
    tile_width: int,
    output_height: int,
    output_width: int,
) -> list[list[int]]:
    return [
        [
            grid[top + row % tile_height][left + col % tile_width]
            for col in range(output_width)
        ]
        for row in range(output_height)
    ]


class TileFromBBoxRepeatRule(BaseRule):
    name = "TileFromBBoxRepeatRule"
    priority = 15

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        output_shapes = _shape_set(cases, "output")
        if len(output_shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "tile_from_bbox_requires_shared_output_size", {})
        output_height, output_width = next(iter(output_shapes))

        candidate_specs: list[tuple[str, int]] = []
        for color in range(10):
            candidate_specs.append(("color", color))
            candidate_specs.append(("non_background", color))

        for kind, color in candidate_specs:
            shared_bbox: tuple[int, int, int, int] | None = None
            valid_bbox = True
            for case in cases:
                input_colors = {cell for row in case["input"] for cell in row}
                if kind == "color" and color not in input_colors:
                    valid_bbox = False
                    break
                colors = {color} if kind == "color" else {cell for cell in input_colors if cell != color}
                bbox = _bbox_for_colors(case["input"], colors)
                if bbox is None:
                    valid_bbox = False
                    break
                shared_bbox = bbox if shared_bbox is None else shared_bbox
                if bbox != shared_bbox:
                    valid_bbox = False
                    break
            if not valid_bbox or shared_bbox is None:
                continue

            top, left, tile_height, tile_width = shared_bbox
            if tile_height <= 0 or tile_width <= 0:
                continue
            color_map: dict[int, int] = {}
            valid = True
            for case in cases:
                tiled = _tile_from_crop(
                    case["input"],
                    top,
                    left,
                    tile_height,
                    tile_width,
                    output_height,
                    output_width,
                )
                case_map = _infer_color_map_from_pairs(tiled, case["output"])
                if case_map is None:
                    valid = False
                    break
                for old_color, new_color in case_map.items():
                    existing = color_map.get(old_color)
                    if existing is not None and existing != new_color:
                        valid = False
                        break
                    color_map[old_color] = new_color
                if not valid:
                    break
            if valid:
                return RuleResult(
                    self.name,
                    True,
                    "MATCH",
                    f"matched tiled bbox kind={kind}, color={color}, bbox={shared_bbox}",
                    {
                        "kind": kind,
                        "color": color,
                        "top": top,
                        "left": left,
                        "tile_height": tile_height,
                        "tile_width": tile_width,
                        "output_height": output_height,
                        "output_width": output_width,
                        "color_map": color_map,
                    },
                )

        return RuleResult(self.name, False, "REJECT", "no shared bbox tile repeat matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        rows = [
            metadata["top"] + row % metadata["tile_height"]
            for row in range(metadata["output_height"])
        ]
        cols = [
            metadata["left"] + col % metadata["tile_width"]
            for col in range(metadata["output_width"])
        ]
        build_spatial_remap_model(
            rows,
            cols,
            output_path,
            output_active_height=metadata["output_height"],
            output_active_width=metadata["output_width"],
            color_map=metadata["color_map"],
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


def _translated_grid_with_fill(
    grid: list[list[int]],
    dy: int,
    dx: int,
    fill_color: int,
) -> list[list[int]]:
    height, width = grid_shape(grid)
    output = [[fill_color for _ in range(width)] for _ in range(height)]
    for row in range(height):
        for col in range(width):
            source_row = row - dy
            source_col = col - dx
            if 0 <= source_row < height and 0 <= source_col < width:
                output[row][col] = grid[source_row][source_col]
    return output


def _single_color_translation_grid(
    grid: list[list[int]],
    target_color: int,
    background_color: int,
    dy: int,
    dx: int,
) -> list[list[int]]:
    height, width = grid_shape(grid)
    output = [row[:] for row in grid]
    for row in range(height):
        for col in range(width):
            if grid[row][col] == target_color:
                output[row][col] = background_color
    for row in range(height):
        for col in range(width):
            if grid[row][col] != target_color:
                continue
            dest_row = row + dy
            dest_col = col + dx
            if 0 <= dest_row < height and 0 <= dest_col < width:
                output[dest_row][dest_col] = target_color
    return output


def _candidate_shifts_for_color(
    input_grid: list[list[int]],
    output_grid: list[list[int]],
    target_color: int,
    max_offset: int,
) -> set[tuple[int, int]]:
    input_cells = [
        (row, col)
        for row, line in enumerate(input_grid)
        for col, color in enumerate(line)
        if color == target_color
    ]
    output_cells = [
        (row, col)
        for row, line in enumerate(output_grid)
        for col, color in enumerate(line)
        if color == target_color
    ]
    if not input_cells:
        return set()
    candidates: set[tuple[int, int]] = set()
    for source_row, source_col in input_cells:
        for dest_row, dest_col in output_cells:
            dy = dest_row - source_row
            dx = dest_col - source_col
            if (dy != 0 or dx != 0) and abs(dy) <= max_offset and abs(dx) <= max_offset:
                candidates.add((dy, dx))
    if not output_cells:
        height, width = grid_shape(input_grid)
        for dy in range(-max_offset, max_offset + 1):
            for dx in range(-max_offset, max_offset + 1):
                if dy == 0 and dx == 0:
                    continue
                if all(not (0 <= row + dy < height and 0 <= col + dx < width) for row, col in input_cells):
                    candidates.add((dy, dx))
    return candidates


class MultiStepTranslationRule(BaseRule):
    name = "MultiStepTranslationRule"
    priority = 16

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "multi_step_translation_requires_same_size", {})
        shapes = _shape_set(cases, "input")

        for fill_color in range(10):
            for dy in range(-15, 16):
                for dx in range(-15, 16):
                    if dy == 0 and dx == 0:
                        continue
                    if dy in {-1, 0, 1} and dx in {-1, 0, 1}:
                        continue
                    if any(abs(dy) >= height or abs(dx) >= width for height, width in shapes):
                        continue
                    if all(
                        _translated_grid_with_fill(case["input"], dy, dx, fill_color) == case["output"]
                        for case in cases
                    ):
                        return RuleResult(
                            self.name,
                            True,
                            "MATCH",
                            f"matched translation dy={dy}, dx={dx}, fill={fill_color}",
                            {
                                "dy": dy,
                                "dx": dx,
                                "fill_color": fill_color,
                                "dynamic_active": len(shapes) != 1,
                                "active_height": next(iter(shapes))[0] if len(shapes) == 1 else None,
                                "active_width": next(iter(shapes))[1] if len(shapes) == 1 else None,
                            },
                        )

        max_offset = 15
        input_colors = {cell for case in cases for row in case["input"] for cell in row}
        output_colors = {cell for case in cases for row in case["output"] for cell in row}
        candidate_backgrounds = sorted(input_colors | output_colors | {0})
        for target_color in sorted(input_colors):
            first_case_shifts = _candidate_shifts_for_color(
                cases[0]["input"],
                cases[0]["output"],
                target_color,
                max_offset,
            )
            if not first_case_shifts:
                continue
            for background_color in candidate_backgrounds:
                if target_color == background_color:
                    continue
                for dy, dx in sorted(first_case_shifts):
                    if all(
                        _single_color_translation_grid(
                            case["input"],
                            target_color,
                            background_color,
                            dy,
                            dx,
                        )
                        == case["output"]
                        for case in cases
                    ):
                        return RuleResult(
                            self.name,
                            True,
                            "MATCH",
                            (
                                f"matched single-color translation color={target_color}, "
                                f"bg={background_color}, dy={dy}, dx={dx}"
                            ),
                            {
                                "mode": "single_color",
                                "target_color": target_color,
                                "background_color": background_color,
                                "dy": dy,
                                "dx": dx,
                                "dynamic_active": len(shapes) != 1,
                                "active_height": next(iter(shapes))[0] if len(shapes) == 1 else None,
                                "active_width": next(iter(shapes))[1] if len(shapes) == 1 else None,
                            },
                        )

        if len(shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "no shape-polymorphic multi-step translation matches all train cases", {})
        return RuleResult(self.name, False, "REJECT", "no multi-step translation matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        if metadata.get("mode") == "single_color":
            if metadata.get("dynamic_active"):
                build_dynamic_single_color_translation_model(
                    metadata["target_color"],
                    metadata["background_color"],
                    metadata["dy"],
                    metadata["dx"],
                    output_path,
                )
            else:
                build_single_color_translation_model(
                    metadata["target_color"],
                    metadata["background_color"],
                    metadata["dy"],
                    metadata["dx"],
                    metadata["active_height"],
                    metadata["active_width"],
                    output_path,
                )
        elif metadata.get("dynamic_active"):
            build_dynamic_fill_translation_model(
                metadata["dy"],
                metadata["dx"],
                metadata["fill_color"],
                output_path,
            )
        elif metadata.get("fill_color", 0) == 0:
            build_zero_fill_translation_remap_model(
                metadata["dy"],
                metadata["dx"],
                metadata["active_height"],
                metadata["active_width"],
                output_path,
            )
        else:
            build_small_translation_model(
                metadata["dy"],
                metadata["dx"],
                output_path,
                active_height=metadata["active_height"],
                active_width=metadata["active_width"],
                fill_color=metadata["fill_color"],
            )
        return CandidateModel(task_id, self.name, output_path, metadata)


def _symmetry_completion_grid(
    grid: list[list[int]],
    mode: str,
    background_color: int,
) -> list[list[int]] | None:
    height, width = grid_shape(grid)
    if mode in {"diag_main", "diag_anti"} and height != width:
        return None
    output = [row[:] for row in grid]
    for row in range(height):
        for col in range(width):
            if grid[row][col] != background_color:
                continue
            if mode == "horizontal":
                source_row, source_col = row, width - 1 - col
            elif mode == "vertical":
                source_row, source_col = height - 1 - row, col
            elif mode == "rot180":
                source_row, source_col = height - 1 - row, width - 1 - col
            elif mode == "diag_main":
                source_row, source_col = col, row
            elif mode == "diag_anti":
                source_row, source_col = width - 1 - col, height - 1 - row
            else:
                raise ValueError(f"unsupported symmetry mode: {mode}")
            source_color = grid[source_row][source_col]
            if source_color != background_color:
                output[row][col] = source_color
    return output


class SymmetryCompletionRule(BaseRule):
    name = "SymmetryCompletionRule"
    priority = 17

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "symmetry_completion_requires_same_size", {})
        shapes = _shape_set(cases, "input")
        if len(shapes) != 1:
            return RuleResult(self.name, False, "REJECT", "symmetry_completion_requires_shared_shape", {})
        active_height, active_width = next(iter(shapes))

        for background_color in range(10):
            for mode in ("horizontal", "vertical", "rot180", "diag_main", "diag_anti"):
                transformed = [
                    _symmetry_completion_grid(case["input"], mode, background_color)
                    for case in cases
                ]
                if any(grid is None for grid in transformed):
                    continue
                if all(grid == case["output"] for grid, case in zip(transformed, cases)):
                    if all(case["input"] == case["output"] for case in cases):
                        continue
                    return RuleResult(
                        self.name,
                        True,
                        "MATCH",
                        f"matched {mode} completion with background color {background_color}",
                        {
                            "mode": mode,
                            "background_color": background_color,
                            "active_height": active_height,
                            "active_width": active_width,
                        },
                    )

        return RuleResult(self.name, False, "REJECT", "no symmetry completion matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_symmetry_completion_model(
            metadata["mode"],
            metadata["background_color"],
            metadata["active_height"],
            metadata["active_width"],
            output_path,
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


def _draw_bbox_grid(
    grid: list[list[int]],
    bbox: tuple[int, int, int, int],
    draw_color: int,
    mode: str,
) -> list[list[int]]:
    top, left, height, width = bbox
    output = [row[:] for row in grid]
    for row in range(top, top + height):
        for col in range(left, left + width):
            if mode == "fill" or row in {top, top + height - 1} or col in {left, left + width - 1}:
                output[row][col] = draw_color
    return output


def _line_points(
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    row0, col0 = start
    row1, col1 = end
    dy = row1 - row0
    dx = col1 - col0
    if dy == 0:
        step_row, step_col, steps = 0, 1 if dx > 0 else -1, abs(dx)
    elif dx == 0:
        step_row, step_col, steps = 1 if dy > 0 else -1, 0, abs(dy)
    elif abs(dy) == abs(dx):
        step_row = 1 if dy > 0 else -1
        step_col = 1 if dx > 0 else -1
        steps = abs(dy)
    else:
        return None
    return [(row0 + step_row * index, col0 + step_col * index) for index in range(steps + 1)]


def _draw_line_grid(
    grid: list[list[int]],
    points: list[tuple[int, int]],
    color: int,
) -> list[list[int]]:
    output = [row[:] for row in grid]
    for row, col in points:
        output[row][col] = color
    return output


def _shared_static_draw_mask(
    cases: list[dict],
    draw_color: int,
) -> list[list[bool]] | None:
    first_height, first_width = grid_shape(cases[0]["input"])
    shared_mask: list[list[bool]] | None = None
    for case in cases:
        if grid_shape(case["input"]) != (first_height, first_width):
            return None
        current_mask = [[False for _ in range(first_width)] for _ in range(first_height)]
        for row in range(first_height):
            for col in range(first_width):
                old_color = case["input"][row][col]
                new_color = case["output"][row][col]
                if old_color == new_color:
                    continue
                if new_color != draw_color:
                    return None
                current_mask[row][col] = True
        if shared_mask is None:
            shared_mask = current_mask
        elif current_mask != shared_mask:
            return None
    return shared_mask


def _extend_line_points(
    grid: list[list[int]],
    color: int,
    direction: tuple[int, int],
) -> list[tuple[int, int]] | None:
    height, width = grid_shape(grid)
    cells = [(row, col) for row in range(height) for col in range(width) if grid[row][col] == color]
    if len(cells) < 2:
        return None
    dr, dc = direction
    values = {row * dc - col * dr for row, col in cells}
    if len(values) != 1:
        return None
    projections = [row * dr + col * dc for row, col in cells]
    if len(set(projections)) != len(projections):
        return None
    row, col = min(cells, key=lambda item: item[0] * dr + item[1] * dc)
    while 0 <= row - dr < height and 0 <= col - dc < width:
        row -= dr
        col -= dc
    points = []
    while 0 <= row < height and 0 <= col < width:
        points.append((row, col))
        row += dr
        col += dc
    return points


class RectangleAndLineRule(BaseRule):
    name = "RectangleAndLineRule"
    priority = 19

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "rectangle_line_requires_same_size", {})

        for draw_color in range(10):
            for mode in ("fill", "frame"):
                if all(
                    _draw_bbox_grid(
                        case["input"],
                        (0, 0, *grid_shape(case["input"])),
                        draw_color,
                        mode,
                    )
                    == case["output"]
                    for case in cases
                ):
                    if all(case["input"] == case["output"] for case in cases):
                        continue
                    return RuleResult(
                        self.name,
                        True,
                        "MATCH",
                        f"matched active rectangle {mode} draw={draw_color}",
                        {"mode": f"active_{mode}", "draw_color": draw_color},
                    )

        for color in range(10):
            valid = True
            for case in cases:
                cells = [
                    (row, col)
                    for row, line in enumerate(case["input"])
                    for col, cell in enumerate(line)
                    if cell == color
                ]
                if len(cells) != 2:
                    valid = False
                    break
                points = _line_points(cells[0], cells[1])
                if points is None or _draw_line_grid(case["input"], points, color) != case["output"]:
                    valid = False
                    break
            if valid:
                if cells[0][0] == cells[1][0]:
                    mode = "connect_two_points_horizontal"
                elif cells[0][1] == cells[1][1]:
                    mode = "connect_two_points_vertical"
                else:
                    mode = "connect_two_points_diagonal"
                static_draw_mask = _shared_static_draw_mask(cases, color)
                metadata = {"mode": mode, "draw_color": color}
                if static_draw_mask is not None:
                    metadata["static_draw_mask"] = static_draw_mask
                return RuleResult(
                    self.name,
                    True,
                    "MATCH",
                    f"matched straight line connection color={color}",
                    metadata,
                )

        for bbox_kind in ("non_background", "color"):
            for bbox_color in range(10):
                for draw_color in range(10):
                    for mode in ("fill", "frame"):
                        valid = True
                        for case in cases:
                            input_colors = {cell for row in case["input"] for cell in row}
                            colors = (
                                {color for color in input_colors if color != bbox_color}
                                if bbox_kind == "non_background"
                                else {bbox_color}
                            )
                            bbox = _bbox_for_colors(case["input"], colors)
                            if bbox is None:
                                valid = False
                                break
                            if _draw_bbox_grid(case["input"], bbox, draw_color, mode) != case["output"]:
                                valid = False
                                break
                        if valid:
                            static_draw_mask = _shared_static_draw_mask(cases, draw_color)
                            metadata = {
                                "mode": f"bbox_{mode}",
                                "bbox_kind": bbox_kind,
                                "bbox_color": bbox_color,
                                "draw_color": draw_color,
                            }
                            if static_draw_mask is not None:
                                metadata["static_draw_mask"] = static_draw_mask
                            return RuleResult(
                                self.name,
                                True,
                                "MATCH",
                                f"matched bbox {mode} kind={bbox_kind} color={bbox_color} draw={draw_color}",
                                metadata,
                            )

        directions = {
            "horizontal": (0, 1),
            "vertical": (1, 0),
            "diag_down": (1, 1),
            "diag_up": (1, -1),
        }
        for color in range(10):
            for direction_name, direction in directions.items():
                valid = True
                for case in cases:
                    points = _extend_line_points(case["input"], color, direction)
                    if points is None or _draw_line_grid(case["input"], points, color) != case["output"]:
                        valid = False
                        break
                if valid:
                    return RuleResult(
                        self.name,
                        True,
                        "MATCH",
                        f"matched line extension direction={direction_name} color={color}",
                        {"mode": "extend_line", "direction": direction_name, "draw_color": color},
                    )

        return RuleResult(self.name, False, "REJECT", "no rectangle or line transform matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        if metadata["mode"] in {"active_fill", "active_frame"}:
            build_active_rectangle_model(
                metadata["mode"].removeprefix("active_"),
                metadata["draw_color"],
                output_path,
            )
            return CandidateModel(task_id, self.name, output_path, metadata)
        if metadata["mode"] in {"bbox_fill", "bbox_frame"} and "static_draw_mask" in metadata:
            build_static_overlay_model(
                metadata["static_draw_mask"],
                metadata["draw_color"],
                output_path,
            )
            return CandidateModel(task_id, self.name, output_path, metadata)
        if metadata["mode"] in {"connect_two_points_horizontal", "connect_two_points_vertical"}:
            if "static_draw_mask" not in metadata:
                raise NotImplementedError(
                    f"RectangleAndLineRule mode requires static_draw_mask: {metadata['mode']}"
                )
            build_static_overlay_model(
                metadata["static_draw_mask"],
                metadata["draw_color"],
                output_path,
            )
            return CandidateModel(task_id, self.name, output_path, metadata)
        if metadata["mode"] == "extend_line" and metadata.get("direction") in {"horizontal", "vertical"}:
            build_line_extension_model(
                metadata["direction"],
                metadata["draw_color"],
                output_path,
            )
            return CandidateModel(task_id, self.name, output_path, metadata)
        raise NotImplementedError(f"RectangleAndLineRule mode is probe-only: {metadata['mode']}")


def _component_bboxes_for_colors(
    grid: list[list[int]],
    colors: set[int],
) -> list[dict[str, Any]]:
    height, width = grid_shape(grid)
    seen: set[tuple[int, int]] = set()
    components: list[dict[str, Any]] = []
    for row in range(height):
        for col in range(width):
            if (row, col) in seen or grid[row][col] not in colors:
                continue
            stack = [(row, col)]
            seen.add((row, col))
            coords: list[tuple[int, int]] = []
            while stack:
                current_row, current_col = stack.pop()
                coords.append((current_row, current_col))
                for row_delta, col_delta in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                    next_row = current_row + row_delta
                    next_col = current_col + col_delta
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in seen or grid[next_row][next_col] not in colors:
                        continue
                    seen.add((next_row, next_col))
                    stack.append((next_row, next_col))
            min_row = min(coord[0] for coord in coords)
            max_row = max(coord[0] for coord in coords)
            min_col = min(coord[1] for coord in coords)
            max_col = max(coord[1] for coord in coords)
            components.append(
                {
                    "top": min_row,
                    "left": min_col,
                    "height": max_row - min_row + 1,
                    "width": max_col - min_col + 1,
                    "area": len(coords),
                }
            )
    return components


def _select_unique_object(objects: list[dict[str, Any]], selector: str) -> dict[str, Any] | None:
    if not objects:
        return None
    if selector == "only":
        return objects[0] if len(objects) == 1 else None
    key_functions = {
        "largest_area": lambda obj: obj["area"],
        "smallest_area": lambda obj: -obj["area"],
        "topmost": lambda obj: -obj["top"],
        "bottommost": lambda obj: obj["top"] + obj["height"] - 1,
        "leftmost": lambda obj: -obj["left"],
        "rightmost": lambda obj: obj["left"] + obj["width"] - 1,
    }
    if selector not in key_functions:
        raise ValueError(f"unsupported object selector: {selector}")
    key_fn = key_functions[selector]
    scores = [key_fn(obj) for obj in objects]
    best_score = max(scores)
    if scores.count(best_score) != 1:
        return None
    return objects[scores.index(best_score)]


def _selected_object_for_case(case: dict, kind: str, color: int, selector: str) -> dict[str, Any] | None:
    grid = case["input"]
    input_colors = {cell for row in grid for cell in row}
    objects: list[dict[str, Any]]
    if kind == "color_bbox":
        if color not in input_colors:
            return None
        bbox = _bbox_for_colors(grid, {color})
        if bbox is None:
            return None
        top, left, height, width = bbox
        objects = [{"top": top, "left": left, "height": height, "width": width, "area": sum(cell == color for row in grid for cell in row)}]
    elif kind == "color_component":
        if color not in input_colors:
            return None
        objects = _component_bboxes_for_colors(grid, {color})
    elif kind == "non_background_bbox":
        colors = {input_color for input_color in input_colors if input_color != color}
        bbox = _bbox_for_colors(grid, colors)
        if bbox is None:
            return None
        top, left, height, width = bbox
        objects = [{"top": top, "left": left, "height": height, "width": width, "area": sum(cell in colors for row in grid for cell in row)}]
    elif kind == "non_background_component":
        colors = {input_color for input_color in input_colors if input_color != color}
        objects = _component_bboxes_for_colors(grid, colors)
    else:
        raise ValueError(f"unsupported object kind: {kind}")
    selected = _select_unique_object(objects, selector)
    if selected is None:
        return None
    return {
        **selected,
        "grid": _crop_grid(grid, selected["top"], selected["left"], selected["height"], selected["width"]),
    }


def _object_candidates_for_case(case: dict, kind: str, color: int, selector: str) -> list[list[int]] | None:
    selected = _selected_object_for_case(case, kind, color, selector)
    if selected is None:
        return None
    return selected["grid"]


class ObjectSelectionRule(BaseRule):
    name = "ObjectSelectionRule"
    priority = 20

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        kinds = (
            "color_bbox",
            "color_component",
            "non_background_bbox",
            "non_background_component",
        )
        selectors = ("only", "largest_area", "smallest_area", "topmost", "bottommost", "leftmost", "rightmost")
        for kind in kinds:
            for color in range(10):
                for selector in selectors:
                    if kind.endswith("_bbox") and selector != "only":
                        continue
                    color_map: dict[int, int] = {}
                    valid = True
                    saw_non_identity = False
                    selected_rects: list[tuple[int, int, int, int]] = []
                    for case in cases:
                        selected = _selected_object_for_case(case, kind, color, selector)
                        if selected is None:
                            valid = False
                            break
                        source = selected["grid"]
                        case_map = _infer_color_map_from_pairs(source, case["output"])
                        if case_map is None:
                            valid = False
                            break
                        if source != case["input"] or not _is_identity_observed_color_map(case_map):
                            saw_non_identity = True
                        selected_rects.append(
                            (
                                int(selected["top"]),
                                int(selected["left"]),
                                int(selected["height"]),
                                int(selected["width"]),
                            )
                        )
                        for old_color, new_color in case_map.items():
                            existing = color_map.get(old_color)
                            if existing is not None and existing != new_color:
                                valid = False
                                break
                            color_map[old_color] = new_color
                        if not valid:
                            break
                    if valid and saw_non_identity:
                        static_rects = set(selected_rects)
                        if len(static_rects) != 1:
                            continue
                        top, left, output_height, output_width = next(iter(static_rects))
                        return RuleResult(
                            self.name,
                            True,
                            "MATCH",
                            (
                                f"matched static object selection kind={kind}, "
                                f"color={color}, selector={selector}"
                            ),
                            {
                                "kind": kind,
                                "color": color,
                                "selector": selector,
                                "top": top,
                                "left": left,
                                "output_height": output_height,
                                "output_width": output_width,
                                "color_map": color_map,
                            },
                        )
        return RuleResult(self.name, False, "REJECT", "no object selection matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        rows = list(range(metadata["top"], metadata["top"] + metadata["output_height"]))
        cols = list(range(metadata["left"], metadata["left"] + metadata["output_width"]))
        build_spatial_remap_model(
            rows,
            cols,
            output_path,
            output_active_height=metadata["output_height"],
            output_active_width=metadata["output_width"],
            color_map=metadata["color_map"],
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


LOCAL_REWRITE_OFFSET_SETS: dict[str, list[tuple[int, int]]] = {
    **LOCAL_OFFSET_SETS,
    "all24": [
        (row_offset, col_offset)
        for row_offset in range(-2, 3)
        for col_offset in range(-2, 3)
        if (row_offset, col_offset) != (0, 0)
    ],
    "cardinal8": [
        (-2, 0),
        (-1, 0),
        (0, -2),
        (0, -1),
        (0, 1),
        (0, 2),
        (1, 0),
        (2, 0),
    ],
    "diagonal8": [
        (-2, -2),
        (-1, -1),
        (-2, 2),
        (-1, 1),
        (1, -1),
        (2, -2),
        (1, 1),
        (2, 2),
    ],
}


def _local_rewrite_transform(
    grid: list[list[int]],
    target_color: int,
    replacement_color: int,
    source_colors: list[int],
    offsets: list[tuple[int, int]],
    condition: str,
    threshold: int,
) -> list[list[int]]:
    height, width = grid_shape(grid)
    source_color_set = set(source_colors)
    output = [row[:] for row in grid]
    for row in range(height):
        for col in range(width):
            if grid[row][col] != target_color:
                continue
            count = 0
            for row_offset, col_offset in offsets:
                neighbor_row = row + row_offset
                neighbor_col = col + col_offset
                if 0 <= neighbor_row < height and 0 <= neighbor_col < width:
                    if grid[neighbor_row][neighbor_col] in source_color_set:
                        count += 1
            if condition == "eq":
                should_rewrite = count == threshold
            elif condition == "ge":
                should_rewrite = count >= threshold
            else:
                raise ValueError(f"unsupported condition: {condition}")
            if should_rewrite:
                output[row][col] = replacement_color
    return output


class LocalNeighborhoodRewriteRule(BaseRule):
    name = "LocalNeighborhoodRewriteRule"
    priority = 18

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "local_rewrite_requires_same_size", {})

        for target_color in range(10):
            for replacement_color in range(10):
                if replacement_color == target_color:
                    continue
                changed_total = 0
                valid_rewrite = True
                for case in cases:
                    for row_index, row in enumerate(case["input"]):
                        for col_index, old_color in enumerate(row):
                            new_color = case["output"][row_index][col_index]
                            if old_color == new_color:
                                continue
                            if old_color == target_color and new_color == replacement_color:
                                changed_total += 1
                                continue
                            valid_rewrite = False
                            break
                        if not valid_rewrite:
                            break
                    if not valid_rewrite:
                        break
                if not valid_rewrite or changed_total == 0:
                    continue

                source_modes = [f"color_{color}" for color in range(10)]
                source_modes.extend(["non_target", "non_replacement"])
                for source_mode in source_modes:
                    if source_mode == "non_target":
                        source_colors = [color for color in range(10) if color != target_color]
                    elif source_mode == "non_replacement":
                        source_colors = [color for color in range(10) if color != replacement_color]
                    else:
                        source_colors = [int(source_mode.removeprefix("color_"))]
                    for offset_name, offsets in LOCAL_REWRITE_OFFSET_SETS.items():
                        for condition in ("eq", "ge"):
                            thresholds = range(0, len(offsets) + 1) if condition == "eq" else range(1, len(offsets) + 1)
                            for threshold in thresholds:
                                if all(
                                    _local_rewrite_transform(
                                        case["input"],
                                        target_color,
                                        replacement_color,
                                        source_colors,
                                        offsets,
                                        condition,
                                        threshold,
                                    )
                                    == case["output"]
                                    for case in cases
                                ):
                                    return RuleResult(
                                        self.name,
                                        True,
                                        "MATCH",
                                        (
                                            "matched local rewrite "
                                            f"target={target_color}, replacement={replacement_color}, "
                                            f"source={source_mode}, offsets={offset_name}, {condition}{threshold}"
                                        ),
                                        {
                                            "target_color": target_color,
                                            "replacement_color": replacement_color,
                                            "source_mode": source_mode,
                                            "source_colors": source_colors,
                                            "offset_name": offset_name,
                                            "offsets": offsets,
                                            "condition": condition,
                                            "threshold": threshold,
                                        },
                                    )

        return RuleResult(self.name, False, "REJECT", "no local neighborhood rewrite matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_local_neighborhood_rewrite_model(
            metadata["target_color"],
            metadata["replacement_color"],
            metadata["source_colors"],
            metadata["offsets"],
            metadata["condition"],
            metadata["threshold"],
            output_path,
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


def _bbox_crop_candidates_for_case(case: dict) -> list[dict[str, Any]]:
    grid = case["input"]
    input_colors = {color for row in grid for color in row}
    candidates: list[dict[str, Any]] = []

    def add(kind: str, bbox: tuple[int, int, int, int] | None, color: int = -1) -> None:
        if bbox is None:
            return
        top, left, height, width = bbox
        crop = _crop_grid(grid, top, left, height, width)
        candidates.append(
            {
                "kind": kind,
                "color": color,
                "top": top,
                "left": left,
                "height": height,
                "width": width,
                "grid": crop,
            }
        )

    for background_color in range(10):
        non_background = {color for color in input_colors if color != background_color}
        add("bbox_of_all_non_background", _bbox_for_colors(grid, non_background), background_color)
    for color in range(10):
        add("bbox_of_color", _bbox_for_colors(grid, {color}), color)
        components = _component_bboxes_for_colors(grid, {color})
        if components:
            largest = max(components, key=lambda item: item["area"])
            smallest = min(components, key=lambda item: item["area"])
            add("bbox_of_largest_component", (largest["top"], largest["left"], largest["height"], largest["width"]), color)
            add("bbox_of_smallest_component", (smallest["top"], smallest["left"], smallest["height"], smallest["width"]), color)
            if len(components) == 1:
                only = components[0]
                add("bbox_of_unique_color_component", (only["top"], only["left"], only["height"], only["width"]), color)

    input_height, input_width = grid_shape(grid)
    for background_color in range(10):
        non_background = {color for color in input_colors if color != background_color}
        for component in _component_bboxes_for_colors(grid, non_background):
            top = component["top"]
            left = component["left"]
            height = component["height"]
            width = component["width"]
            touches_border = top == 0 or left == 0 or top + height == input_height or left + width == input_width
            if not touches_border:
                add("bbox_of_component_not_touching_border", (top, left, height, width), background_color)
    return candidates


def _merge_color_maps(left: dict[int, int], right: dict[int, int]) -> dict[int, int] | None:
    merged = dict(left)
    for old_color, new_color in right.items():
        existing = merged.get(old_color)
        if existing is not None and existing != new_color:
            return None
        merged[old_color] = new_color
    return merged


BUILDABLE_BBOX_KINDS = {"bbox_of_all_non_background", "bbox_of_color", "bbox_of_unique_color_component"}


class DynamicBBoxCropRule(BaseRule):
    """Probe dynamic bbox/object crop and normalization semantics."""

    name = "DynamicBBoxCropRule"
    priority = 22

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        candidates: dict[str, dict[str, Any]] | None = None
        for case in cases:
            current: dict[str, dict[str, Any]] = {}
            for candidate in _bbox_crop_candidates_for_case(case):
                transforms = {
                    "identity": candidate["grid"],
                    "mirror_horizontal": _mirror_grid(candidate["grid"], "horizontal"),
                    "mirror_vertical": _mirror_grid(candidate["grid"], "vertical"),
                }
                for transform, transformed in transforms.items():
                    color_map = _infer_color_map_from_pairs(transformed, case["output"])
                    if color_map is None:
                        continue
                    buildable = candidate["kind"] in BUILDABLE_BBOX_KINDS
                    key = f"{candidate['kind']}:{candidate['color']}:{transform}:crop_colormap"
                    current[key] = {
                        "kind": candidate["kind"],
                        "color": candidate["color"],
                        "transform": transform,
                        "mode": "bbox_crop_colormap",
                        "color_map": color_map,
                        "builder_available": buildable,
                        "blocked_reason": "" if buildable else "builder_missing_dynamic_bbox",
                    }
            if candidates is None:
                candidates = current
            else:
                merged: dict[str, dict[str, Any]] = {}
                for key, previous in candidates.items():
                    item = current.get(key)
                    if item is None:
                        continue
                    color_map = _merge_color_maps(previous["color_map"], item["color_map"])
                    if color_map is not None:
                        merged[key] = {**previous, "color_map": color_map}
                candidates = merged
            if not candidates:
                return RuleResult(self.name, False, "REJECT", "no dynamic bbox crop matches all train cases", {})
        buildable = [candidate for candidate in candidates.values() if candidate.get("builder_available")]
        best_pool = buildable if buildable else list(candidates.values())
        best = sorted(best_pool, key=lambda item: (item["kind"], item["color"], item["transform"]))[0]
        return RuleResult(
            self.name,
            True,
            "MATCH",
            f"matched dynamic bbox crop kind={best['kind']} color={best['color']} transform={best['transform']}",
            best,
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        if metadata["kind"] == "bbox_of_all_non_background":
            build_dynamic_non_background_bbox_crop_model(
                int(metadata["color"]),
                metadata["color_map"],
                output_path,
                transform=metadata.get("transform", "identity"),
            )
        elif metadata["kind"] in {"bbox_of_color", "bbox_of_unique_color_component"}:
            build_dynamic_color_bbox_crop_model(
                int(metadata["color"]),
                metadata["color_map"],
                output_path,
                transform=metadata.get("transform", "identity"),
            )
        else:
            raise NotImplementedError(f"DynamicBBoxCropRule cannot build kind={metadata['kind']}")
        return CandidateModel(task_id, self.name, output_path, metadata)


class DynamicNonBackgroundBBoxCropRule(BaseRule):
    """Buildable dynamic crop of the bbox around all non-background cells."""

    name = "DynamicNonBackgroundBBoxCropRule"
    priority = 26

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        candidates: dict[tuple[int, str], dict[str, Any]] | None = None
        for case in cases:
            grid = case["input"]
            input_colors = {color for row in grid for color in row}
            current: dict[tuple[int, str], dict[str, Any]] = {}
            for background_color in range(10):
                non_background = {color for color in input_colors if color != background_color}
                bbox = _bbox_for_colors(grid, non_background)
                if bbox is None:
                    continue
                top, left, height, width = bbox
                crop = _crop_grid(grid, top, left, height, width)
                transforms = {
                    "identity": crop,
                    "mirror_horizontal": _mirror_grid(crop, "horizontal"),
                    "mirror_vertical": _mirror_grid(crop, "vertical"),
                }
                for transform, transformed in transforms.items():
                    color_map = _infer_color_map_from_pairs(transformed, case["output"])
                    if color_map is None:
                        continue
                    current[(background_color, transform)] = {
                        "background_color": background_color,
                        "transform": transform,
                        "color_map": color_map,
                        "mode": "dynamic_non_background_bbox_crop",
                    }
            if candidates is None:
                candidates = current
            else:
                merged: dict[tuple[int, str], dict[str, Any]] = {}
                for key, previous in candidates.items():
                    item = current.get(key)
                    if item is None:
                        continue
                    color_map = dict(previous["color_map"])
                    compatible = True
                    for old_color, new_color in item["color_map"].items():
                        existing = color_map.get(old_color)
                        if existing is not None and existing != new_color:
                            compatible = False
                            break
                        color_map[old_color] = new_color
                    if compatible:
                        merged[key] = {**previous, "color_map": color_map}
                candidates = merged
            if not candidates:
                return RuleResult(
                    self.name,
                    False,
                    "REJECT",
                    "no dynamic non-background bbox crop matches all train cases",
                    {},
                )

        best = candidates[sorted(candidates)[0]]
        return RuleResult(
            self.name,
            True,
            "MATCH",
            (
                "matched dynamic non-background bbox crop "
                f"background={best['background_color']} transform={best['transform']}"
            ),
            best,
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_dynamic_non_background_bbox_crop_model(
            metadata["background_color"],
            metadata["color_map"],
            output_path,
            transform=metadata.get("transform", "identity"),
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


def _swap_extreme_colors(grid: list[list[int]], background_color: int) -> list[list[int]] | None:
    colors = [color for row in grid for color in row]
    if any(color == background_color for color in colors):
        return None
    counts = {color: colors.count(color) for color in sorted(set(colors))}
    if len(counts) != 2:
        return None
    least_color = min(counts, key=lambda color: (counts[color], color))
    most_color = max(counts, key=lambda color: (counts[color], -color))
    if least_color == most_color:
        return None
    return [
        [
            least_color if color == most_color else most_color if color == least_color else color
            for color in row
        ]
        for row in grid
    ]


class DynamicBBoxExtremeColorSwapRule(BaseRule):
    """Crop the non-background bbox and swap its most/least frequent colors."""

    name = "DynamicBBoxExtremeColorSwapRule"
    priority = 28

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        possible_backgrounds: set[int] | None = None
        for case in cases:
            grid = case["input"]
            input_colors = {color for row in grid for color in row}
            current: set[int] = set()
            for background_color in range(10):
                non_background = {color for color in input_colors if color != background_color}
                bbox = _bbox_for_colors(grid, non_background)
                if bbox is None:
                    continue
                top, left, height, width = bbox
                crop = _crop_grid(grid, top, left, height, width)
                swapped = _swap_extreme_colors(crop, background_color)
                if swapped == case["output"]:
                    current.add(background_color)
            possible_backgrounds = current if possible_backgrounds is None else possible_backgrounds & current
            if not possible_backgrounds:
                return RuleResult(
                    self.name,
                    False,
                    "REJECT",
                    "no dynamic bbox extreme color swap matches all train cases",
                    {},
                )
        background_color = sorted(possible_backgrounds)[0]
        return RuleResult(
            self.name,
            True,
            "MATCH",
            f"matched dynamic bbox extreme color swap background={background_color}",
            {"background_color": background_color},
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        build_dynamic_bbox_extreme_color_swap_model(metadata["background_color"], output_path)
        return CandidateModel(task_id, self.name, output_path, metadata)


def _frame_bboxes(grid: list[list[int]], color: int) -> list[tuple[int, int, int, int]]:
    height, width = grid_shape(grid)
    bboxes: list[tuple[int, int, int, int]] = []
    for top in range(height):
        for bottom in range(top + 2, height):
            for left in range(width):
                for right in range(left + 2, width):
                    if all(grid[top][col] == color and grid[bottom][col] == color for col in range(left, right + 1)) and all(
                        grid[row][left] == color and grid[row][right] == color for row in range(top, bottom + 1)
                    ):
                        bboxes.append((top, left, bottom - top + 1, right - left + 1))
    return bboxes


class FrameInteriorRule(BaseRule):
    """Probe frame interior/extract/fill/remove transformations."""

    name = "FrameInteriorRule"
    priority = 23

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        candidates: dict[str, dict[str, Any]] | None = None
        for case in cases:
            current: dict[str, dict[str, Any]] = {}
            for color in range(10):
                for bbox in _frame_bboxes(case["input"], color):
                    top, left, height, width = bbox
                    if height <= 2 or width <= 2:
                        continue
                    interior = _crop_grid(case["input"], top + 1, left + 1, height - 2, width - 2)
                    color_map = _infer_color_map_from_pairs(interior, case["output"])
                    if color_map is not None:
                        buildable = _bbox_for_colors(case["input"], {color}) == bbox
                        current[f"frame_interior_crop:{color}"] = {
                            "mode": "frame_interior_crop",
                            "frame_color": color,
                            "color_map": color_map,
                            "builder_available": buildable,
                            "blocked_reason": "" if buildable else "frame_color_bbox_contains_extra_cells",
                        }
                    for fill_color in range(10):
                        filled = [row[:] for row in case["input"]]
                        for row in range(top + 1, top + height - 1):
                            for col in range(left + 1, left + width - 1):
                                filled[row][col] = fill_color
                        if filled == case["output"]:
                            current[f"frame_fill:{color}:{fill_color}"] = {
                                "mode": "frame_fill",
                                "frame_color": color,
                                "fill_color": fill_color,
                                "builder_available": False,
                                "blocked_reason": "builder_missing_dynamic_bbox",
                            }
            if candidates is None:
                candidates = current
            else:
                merged: dict[str, dict[str, Any]] = {}
                for key, previous in candidates.items():
                    item = current.get(key)
                    if item is None:
                        continue
                    if previous["mode"] == "frame_interior_crop":
                        color_map = _merge_color_maps(previous["color_map"], item["color_map"])
                        if color_map is None:
                            continue
                        merged[key] = {**previous, "color_map": color_map}
                    else:
                        merged[key] = previous
                candidates = merged
            if not candidates:
                return RuleResult(self.name, False, "REJECT", "no frame interior transform matches all train cases", {})
        buildable = [candidate for candidate in candidates.values() if candidate.get("builder_available")]
        best_pool = buildable if buildable else list(candidates.values())
        best = sorted(best_pool, key=lambda item: (item["mode"], item.get("frame_color", -1)))[0]
        return RuleResult(
            self.name,
            True,
            "MATCH",
            f"matched frame transform mode={best['mode']} color={best.get('frame_color')}",
            best,
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        if metadata.get("mode") != "frame_interior_crop":
            raise NotImplementedError("FrameInteriorRule currently builds only frame_interior_crop candidates")
        build_dynamic_frame_interior_crop_model(
            int(metadata["frame_color"]),
            metadata["color_map"],
            output_path,
        )
        return CandidateModel(task_id, self.name, output_path, metadata)


def _neighbor_count_same_color(grid: list[list[int]], row: int, col: int, color: int) -> int:
    height, width = grid_shape(grid)
    total = 0
    for row_delta in (-1, 0, 1):
        for col_delta in (-1, 0, 1):
            if row_delta == 0 and col_delta == 0:
                continue
            next_row = row + row_delta
            next_col = col + col_delta
            if 0 <= next_row < height and 0 <= next_col < width and grid[next_row][next_col] == color:
                total += 1
    return total


class ObjectEditRule(BaseRule):
    """Probe same-size object edit rules such as outline and noise removal."""

    name = "ObjectEditRule"
    priority = 24

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        if not all(_input_output_shapes_equal(case) for case in cases):
            return RuleResult(self.name, False, "REJECT", "object_edit_requires_same_size", {})
        for noise_color in range(10):
            for background_color in range(10):
                if noise_color == background_color:
                    continue
                valid = True
                changed = 0
                for case in cases:
                    output = [row[:] for row in case["input"]]
                    for row_index, row in enumerate(case["input"]):
                        for col_index, color in enumerate(row):
                            if color == noise_color and _neighbor_count_same_color(case["input"], row_index, col_index, noise_color) == 0:
                                output[row_index][col_index] = background_color
                                changed += 1
                    if output != case["output"]:
                        valid = False
                        break
                if valid and changed:
                    return RuleResult(
                        self.name,
                        True,
                        "MATCH",
                        f"matched isolated noise removal color={noise_color} background={background_color}",
                        {
                            "mode": "remove_isolated_noise",
                            "noise_color": noise_color,
                            "background_color": background_color,
                            "builder_available": False,
                            "blocked_reason": "builder_missing_object_edit",
                        },
                    )
        for object_color in range(10):
            for outline_color in range(10):
                if object_color == outline_color:
                    continue
                valid = True
                changed = 0
                for case in cases:
                    output = [row[:] for row in case["input"]]
                    height, width = grid_shape(case["input"])
                    for row in range(height):
                        for col in range(width):
                            if case["input"][row][col] == object_color:
                                continue
                            touches = False
                            for row_delta, col_delta in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                                next_row = row + row_delta
                                next_col = col + col_delta
                                if 0 <= next_row < height and 0 <= next_col < width and case["input"][next_row][next_col] == object_color:
                                    touches = True
                                    break
                            if touches:
                                output[row][col] = outline_color
                                changed += 1
                    if output != case["output"]:
                        valid = False
                        break
                if valid and changed:
                    return RuleResult(
                        self.name,
                        True,
                        "MATCH",
                        f"matched object outline color={object_color} outline={outline_color}",
                        {
                            "mode": "object_outline",
                            "object_color": object_color,
                            "outline_color": outline_color,
                            "builder_available": False,
                            "blocked_reason": "builder_missing_object_edit",
                        },
                    )
        return RuleResult(self.name, False, "REJECT", "no object edit transform matches all train cases", {})

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        raise NotImplementedError("ObjectEditRule is probe-only until object edit builders are compiled")


class ComposedRuleSearch(BaseRule):
    """Probe simple two-step extract/finish compositions."""

    name = "ComposedRuleSearch"
    priority = 25

    def match(self, task: dict) -> RuleResult:
        cases = _train_cases(task)
        candidates: dict[str, dict[str, Any]] | None = None
        for case in cases:
            current: dict[str, dict[str, Any]] = {}
            extractors = _bbox_crop_candidates_for_case(case)
            for layout in _panel_layouts_any_shape(case["input"]):
                for panel_index, spec in enumerate(layout["panel_specs"]):
                    extractors.append(
                        {
                            "kind": f"panel_extract_{layout['layout']}",
                            "color": panel_index,
                            "grid": _panel_rect(case["input"], spec["top"], spec["left"], spec["height"], spec["width"]),
                        }
                    )
            for extractor in extractors:
                mids = [
                    ("identity", extractor["grid"]),
                    ("mirror_horizontal", _mirror_grid(extractor["grid"], "horizontal")),
                    ("mirror_vertical", _mirror_grid(extractor["grid"], "vertical")),
                    ("rotate_90", _rotate_grid(extractor["grid"], 1)),
                    ("rotate_180", _rotate_grid(extractor["grid"], 2)),
                    ("rotate_270", _rotate_grid(extractor["grid"], 3)),
                ]
                for finisher, mid in mids:
                    color_map = _infer_color_map_from_pairs(mid, case["output"])
                    if color_map is None:
                        continue
                    key = f"{extractor['kind']}:{extractor.get('color', -1)}:{finisher}:color_map"
                    buildable = extractor["kind"] in BUILDABLE_BBOX_KINDS and finisher in {
                        "identity",
                        "mirror_horizontal",
                        "mirror_vertical",
                    }
                    current[key] = {
                        "extractor": extractor["kind"],
                        "extractor_color": extractor.get("color", -1),
                        "finisher": finisher,
                        "color_map": color_map,
                        "builder_available": buildable,
                        "blocked_reason": "" if buildable else "requires_composed_rule",
                    }
            if candidates is None:
                candidates = current
            else:
                merged: dict[str, dict[str, Any]] = {}
                for key, previous in candidates.items():
                    item = current.get(key)
                    if item is None:
                        continue
                    color_map = _merge_color_maps(previous["color_map"], item["color_map"])
                    if color_map is not None:
                        merged[key] = {**previous, "color_map": color_map}
                candidates = merged
            if not candidates:
                return RuleResult(self.name, False, "REJECT", "no two-step composition matches all train cases", {})
        buildable = [candidate for candidate in candidates.values() if candidate.get("builder_available")]
        best_pool = buildable if buildable else list(candidates.values())
        best = sorted(best_pool, key=lambda item: (item["extractor"], item["finisher"]))[0]
        return RuleResult(
            self.name,
            True,
            "MATCH",
            f"matched composed rule {best['extractor']} -> {best['finisher']}",
            best,
        )

    def build(
        self,
        task_id: str,
        task: dict,
        output_path: str,
        metadata: dict[str, Any],
    ) -> CandidateModel:
        transform = metadata.get("finisher", "identity")
        if transform not in {"identity", "mirror_horizontal", "mirror_vertical"}:
            raise NotImplementedError(f"ComposedRuleSearch cannot build finisher={transform}")
        if metadata["extractor"] == "bbox_of_all_non_background":
            build_dynamic_non_background_bbox_crop_model(
                int(metadata["extractor_color"]),
                metadata["color_map"],
                output_path,
                transform=transform,
            )
        elif metadata["extractor"] in {"bbox_of_color", "bbox_of_unique_color_component"}:
            build_dynamic_color_bbox_crop_model(
                int(metadata["extractor_color"]),
                metadata["color_map"],
                output_path,
                transform=transform,
            )
        else:
            raise NotImplementedError(f"ComposedRuleSearch cannot build extractor={metadata['extractor']}")
        return CandidateModel(task_id, self.name, output_path, metadata)


def second_round_probe_rules() -> list[BaseRule]:
    """Return the new pure-Python probe rules requested for the second round."""
    return [
        GeneralizedPanelRule(),
        PanelSeparatorBinaryOpRule(),
        PanelSelectByColorRule(),
        PeriodicExtensionColorMapRule(),
        HoleFillRule(),
        LocalNeighborhoodFillRule(),
    ]


def third_round_probe_rules() -> list[BaseRule]:
    """Return all second- and third-round probe rules."""
    return [
        GeneralizedPanelRule(),
        PanelSeparatorBinaryOpRule(),
        PanelSelectByColorRule(),
        PeriodicExtensionColorMapRule(),
        HoleFillRule(),
        LocalNeighborhoodFillRule(),
        SubstructureExtractRule(),
        TileFromBBoxRepeatRule(),
        MultiStepTranslationRule(),
        SymmetryCompletionRule(),
        RectangleAndLineRule(),
        ObjectSelectionRule(),
        LocalNeighborhoodRewriteRule(),
        PanelSemanticRule(),
        DynamicQuadrantPanelSelectRule(),
        DynamicBBoxCropRule(),
        FrameInteriorRule(),
        ObjectEditRule(),
        ComposedRuleSearch(),
    ]


def first_version_rules() -> list[BaseRule]:
    """Return rules in deterministic priority order."""
    return [
        IdentityRule(),
        ColorMapRule(),
        OneStepTranslationRule(),
        MirrorRule(),
        DynamicActiveMirrorRule(),
        RotateRule(),
        CropRule(),
        ScaleRepeatRule(),
        StridedSubsampleRule(),
        TileRepeatRule(),
        MirrorConcatRule(),
        SelfKronMaskRule(),
        GeneralizedPanelRule(),
        PanelSeparatorBinaryOpRule(),
        PanelSelectByColorRule(),
        PeriodicExtensionColorMapRule(),
        HoleFillRule(),
        LocalNeighborhoodFillRule(),
        SubstructureExtractRule(),
        TileFromBBoxRepeatRule(),
        MultiStepTranslationRule(),
        SymmetryCompletionRule(),
        RectangleAndLineRule(),
        ObjectSelectionRule(),
        LocalNeighborhoodRewriteRule(),
        DynamicQuadrantPanelSelectRule(),
        DynamicBBoxCropRule(),
        DynamicNonBackgroundBBoxCropRule(),
        FrameInteriorRule(),
        ComposedRuleSearch(),
        DynamicBBoxExtremeColorSwapRule(),
    ]
