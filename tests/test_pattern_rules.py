from __future__ import annotations

from src.pattern_rules import (
    DynamicBBoxCropRule,
    DynamicBBoxExtremeColorSwapRule,
    DynamicActiveMirrorRule,
    DynamicNonBackgroundBBoxCropRule,
    DynamicQuadrantPanelSelectRule,
    FrameInteriorRule,
    ColorMapRule,
    ComposedRuleSearch,
    CropRule,
    GeneralizedPanelRule,
    HoleFillRule,
    IdentityRule,
    LocalNeighborhoodFillRule,
    LocalNeighborhoodRewriteRule,
    MirrorConcatRule,
    ObjectEditRule,
    MultiStepTranslationRule,
    ObjectSelectionRule,
    PanelSemanticRule,
    PanelSelectByColorRule,
    OneStepTranslationRule,
    PanelSeparatorBinaryOpRule,
    PeriodicExtensionColorMapRule,
    RectangleAndLineRule,
    ScaleRepeatRule,
    SelfKronMaskRule,
    StridedSubsampleRule,
    SubstructureExtractRule,
    SymmetryCompletionRule,
    TileFromBBoxRepeatRule,
)
from src.candidate_discovery_report import build_candidate_discovery_report
from src.validate_onnx_model import validate_task


def _assert_rule_builds_and_validates(rule, task, tmp_path) -> None:
    result = rule.match(task)
    assert result.matched is True
    model_path = tmp_path / f"{rule.name}.onnx"
    rule.build("task999", task, str(model_path), result.metadata)
    validation = validate_task(str(model_path), task)
    assert validation["passed"] is True


def test_identity_rule_matches_identity_task() -> None:
    task = {"train": [{"input": [[1, 0], [2, 3]], "output": [[1, 0], [2, 3]]}]}

    result = IdentityRule().match(task)

    assert result.matched is True
    assert result.confidence == "MATCH"


def test_identity_rule_rejects_non_identity_task() -> None:
    task = {"train": [{"input": [[1]], "output": [[2]]}]}

    result = IdentityRule().match(task)

    assert result.matched is False


def test_color_map_rule_infers_global_mapping() -> None:
    task = {
        "train": [
            {"input": [[1, 0], [1, 0]], "output": [[2, 0], [2, 0]]},
            {"input": [[3, 1]], "output": [[4, 2]]},
        ]
    }

    result = ColorMapRule().match(task)

    assert result.matched is True
    assert result.metadata["color_map"] == {1: 2, 0: 0, 3: 4}


def test_color_map_rule_rejects_inconsistent_mapping() -> None:
    task = {"train": [{"input": [[1, 1]], "output": [[2, 3]]}]}

    result = ColorMapRule().match(task)

    assert result.matched is False
    assert "inconsistent mapping" in result.reason


def test_one_step_translation_rule_detects_shared_shift() -> None:
    task = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[0, 1], [0, 3]],
            },
            {
                "input": [[5, 6], [7, 8]],
                "output": [[0, 5], [0, 7]],
            },
        ]
    }

    result = OneStepTranslationRule().match(task)

    assert result.matched is True
    assert result.metadata["dy"] == 0
    assert result.metadata["dx"] == 1


def test_one_step_translation_rule_rejects_different_shifts() -> None:
    task = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [[0, 1], [0, 3]],
            },
            {
                "input": [[5, 6], [7, 8]],
                "output": [[0, 0], [5, 6]],
            },
        ]
    }

    result = OneStepTranslationRule().match(task)

    assert result.matched is False


def test_dynamic_active_mirror_rule_builds_valid_variable_shapes(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [6, 6, 6, 2],
                    [6, 1, 6, 2],
                    [7, 2, 7, 2],
                    [1, 7, 2, 2],
                ],
                "output": [
                    [2, 6, 6, 6],
                    [2, 6, 1, 6],
                    [2, 7, 2, 7],
                    [2, 2, 7, 1],
                ],
            },
            {
                "input": [
                    [8, 1, 2],
                    [4, 4, 8],
                    [3, 7, 2],
                ],
                "output": [
                    [2, 1, 8],
                    [8, 4, 4],
                    [2, 7, 3],
                ],
            },
        ]
    }

    rule = DynamicActiveMirrorRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["mode"] == "horizontal"
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_crop_rule_detects_fixed_crop() -> None:
    task = {
        "train": [
            {"input": [[1, 2, 3], [4, 5, 6]], "output": [[2, 3], [5, 6]]},
            {"input": [[7, 8, 9], [1, 2, 3]], "output": [[8, 9], [2, 3]]},
        ]
    }

    result = CropRule().match(task)

    assert result.matched is True
    assert result.metadata["top"] == 0
    assert result.metadata["left"] == 1


def test_scale_repeat_rule_detects_nearest_repeat() -> None:
    task = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [
                    [1, 1, 2, 2],
                    [1, 1, 2, 2],
                    [3, 3, 4, 4],
                    [3, 3, 4, 4],
                ],
            }
        ]
    }

    result = ScaleRepeatRule().match(task)

    assert result.matched is True
    assert result.metadata["scale_y"] == 2
    assert result.metadata["scale_x"] == 2


def test_strided_subsample_rule_detects_fixed_stride() -> None:
    task = {
        "train": [
            {
                "input": [
                    [1, 9, 2, 9],
                    [9, 9, 9, 9],
                    [3, 9, 4, 9],
                    [9, 9, 9, 9],
                ],
                "output": [[1, 2], [3, 4]],
            }
        ]
    }

    result = StridedSubsampleRule().match(task)

    assert result.matched is True
    assert result.metadata["stride_y"] == 2
    assert result.metadata["stride_x"] == 2


def test_mirror_concat_rule_detects_horizontal_append() -> None:
    task = {"train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2, 2, 1], [3, 4, 4, 3]]}]}

    result = MirrorConcatRule().match(task)

    assert result.matched is True
    assert result.metadata["mode"] == "h_input_mirror"


def test_self_kron_mask_rule_detects_template_expansion() -> None:
    task = {
        "train": [
            {
                "input": [[1, 0], [2, 3]],
                "output": [
                    [1, 0, 0, 0],
                    [2, 3, 0, 0],
                    [1, 0, 1, 0],
                    [2, 3, 2, 3],
                ],
            }
        ]
    }

    result = SelfKronMaskRule().match(task)

    assert result.matched is True


def test_panel_separator_binary_op_rule_builds_valid_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [1, 0, 9, 0, 1],
                    [1, 0, 9, 1, 0],
                ],
                "output": [
                    [0, 0],
                    [2, 0],
                ],
            }
        ]
    }

    rule = PanelSeparatorBinaryOpRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["operation"] == "AND"
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_generalized_panel_rule_supports_multi_col_separator(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [1, 0, 9, 9, 0, 1],
                    [1, 0, 9, 9, 1, 0],
                ],
                "output": [
                    [0, 0],
                    [2, 0],
                ],
            }
        ]
    }

    rule = GeneralizedPanelRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["operation"] == "AND"
    assert result.metadata["separator_width"] == 2
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_generalized_panel_rule_supports_2x2_majority(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [1, 0, 9, 1, 0],
                    [0, 0, 9, 1, 1],
                    [9, 9, 9, 9, 9],
                    [1, 1, 9, 0, 0],
                    [0, 1, 9, 0, 1],
                ],
                "output": [
                    [2, 0],
                    [0, 2],
                ],
            }
        ]
    }

    rule = GeneralizedPanelRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["layout"] == "grid_2x2"
    assert result.metadata["operation"] == "MAJORITY"
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_panel_semantic_rule_detects_variable_panel_selection() -> None:
    task = {
        "train": [
            {
                "input": [
                    [1, 0, 9, 0, 0],
                    [1, 0, 9, 2, 2],
                ],
                "output": [
                    [1, 0],
                    [1, 0],
                ],
            },
            {
                "input": [
                    [3, 0, 8, 0, 0],
                    [3, 0, 8, 4, 4],
                ],
                "output": [
                    [3, 0],
                    [3, 0],
                ],
            },
        ]
    }

    result = PanelSemanticRule().match(task)

    assert result.matched is True
    assert result.metadata["builder_available"] is False
    assert result.metadata["blocked_reason"] == "builder_missing_dynamic_panel_select"


def test_dynamic_quadrant_panel_select_rule_builds_unique_color_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [8, 8, 3, 8, 8],
                    [8, 8, 3, 8, 8],
                    [3, 3, 3, 3, 3],
                    [8, 8, 3, 8, 8],
                    [4, 8, 3, 8, 8],
                ],
                "output": [
                    [8, 8],
                    [4, 8],
                ],
            },
            {
                "input": [
                    [4, 4, 4, 2, 4, 4, 4],
                    [4, 4, 4, 2, 4, 1, 4],
                    [4, 4, 4, 2, 4, 4, 4],
                    [2, 2, 2, 2, 2, 2, 2],
                    [4, 4, 4, 2, 4, 4, 4],
                    [4, 4, 4, 2, 4, 4, 4],
                    [4, 4, 4, 2, 4, 4, 4],
                ],
                "output": [
                    [4, 4, 4],
                    [4, 1, 4],
                    [4, 4, 4],
                ],
            },
        ]
    }

    rule = DynamicQuadrantPanelSelectRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["selector"] == "unique_max_panel_difference"
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_dynamic_quadrant_panel_select_rule_builds_unique_pattern_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 2, 0, 0, 2],
                    [2, 2, 0, 2, 2],
                    [0, 0, 0, 0, 0],
                    [0, 2, 0, 2, 2],
                    [2, 2, 0, 2, 0],
                ],
                "output": [
                    [2, 2],
                    [2, 0],
                ],
            },
            {
                "input": [
                    [1, 0, 0, 1, 0],
                    [0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 0],
                    [1, 1, 0, 0, 1],
                ],
                "output": [
                    [1, 0],
                    [1, 1],
                ],
            },
        ]
    }

    _assert_rule_builds_and_validates(DynamicQuadrantPanelSelectRule(), task, tmp_path)


def test_dynamic_bbox_crop_rule_detects_object_crop() -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 5, 5, 0],
                    [0, 5, 5, 0],
                    [0, 0, 0, 0],
                ],
                "output": [[5, 5], [5, 5]],
            },
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 0, 7, 7, 0],
                    [0, 0, 7, 7, 0],
                    [0, 0, 0, 0, 0],
                ],
                "output": [[7, 7], [7, 7]],
            },
        ]
    }

    result = DynamicBBoxCropRule().match(task)

    assert result.matched is True
    assert result.metadata["builder_available"] is True


def test_dynamic_bbox_crop_rule_builds_color_specific_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 5, 5, 0, 2],
                    [0, 5, 0, 0, 2],
                    [0, 0, 0, 0, 0],
                ],
                "output": [[8, 8], [8, 0]],
            },
            {
                "input": [
                    [0, 3, 0, 0],
                    [0, 0, 5, 5],
                    [0, 0, 0, 5],
                    [0, 0, 0, 0],
                ],
                "output": [[8, 8], [0, 8]],
            },
        ]
    }

    rule = DynamicBBoxCropRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["kind"] in {"bbox_of_color", "bbox_of_unique_color_component"}
    assert result.metadata["color"] == 5
    assert result.metadata["builder_available"] is True
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_dynamic_non_background_bbox_crop_rule_builds_valid_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    [1, 1, 2, 1],
                    [1, 1, 1, 1],
                ],
                "output": [[2, 2], [0, 2]],
            },
            {
                "input": [
                    [1, 1, 1, 1, 1],
                    [1, 1, 3, 1, 1],
                    [1, 1, 3, 4, 1],
                    [1, 1, 1, 1, 1],
                ],
                "output": [[3, 0], [3, 4]],
            },
        ]
    }

    rule = DynamicNonBackgroundBBoxCropRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["background_color"] == 1
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_dynamic_non_background_bbox_crop_rule_builds_mirrored_crop(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0],
                    [0, 2, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                "output": [[1, 1], [1, 2]],
            },
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 0, 5, 5, 0],
                    [0, 0, 6, 5, 0],
                    [0, 0, 0, 0, 0],
                ],
                "output": [[5, 5], [5, 6]],
            },
        ]
    }

    rule = DynamicNonBackgroundBBoxCropRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["transform"] == "mirror_horizontal"
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_dynamic_bbox_extreme_color_swap_rule_builds_valid_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 2, 2, 2, 0],
                    [0, 2, 4, 2, 0],
                    [0, 2, 2, 2, 0],
                    [0, 0, 0, 0, 0],
                ],
                "output": [
                    [4, 4, 4],
                    [4, 2, 4],
                    [4, 4, 4],
                ],
            },
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 7, 7, 0],
                    [0, 7, 3, 0],
                    [0, 0, 0, 0],
                ],
                "output": [
                    [3, 3],
                    [3, 7],
                ],
            },
        ]
    }

    rule = DynamicBBoxExtremeColorSwapRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["background_color"] == 0
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_frame_interior_rule_detects_interior_crop() -> None:
    task = {
        "train": [
            {
                "input": [
                    [6, 6, 6, 6],
                    [6, 1, 2, 6],
                    [6, 3, 4, 6],
                    [6, 6, 6, 6],
                ],
                "output": [[1, 2], [3, 4]],
            }
        ]
    }

    result = FrameInteriorRule().match(task)

    assert result.matched is True
    assert result.metadata["mode"] == "frame_interior_crop"
    assert result.metadata["builder_available"] is True


def test_frame_interior_rule_builds_dynamic_interior_crop(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 6, 6, 6, 0],
                    [0, 6, 1, 6, 0],
                    [0, 6, 2, 6, 0],
                    [0, 6, 6, 6, 0],
                ],
                "output": [[7], [8]],
            },
            {
                "input": [
                    [6, 6, 6, 6],
                    [6, 1, 2, 6],
                    [6, 6, 6, 6],
                ],
                "output": [[7, 8]],
            },
        ]
    }

    rule = FrameInteriorRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["mode"] == "frame_interior_crop"
    assert result.metadata["color_map"] == {1: 7, 2: 8}
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_object_edit_rule_detects_isolated_noise_removal() -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0],
                    [0, 8, 0],
                    [0, 0, 0],
                ],
                "output": [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            }
        ]
    }

    result = ObjectEditRule().match(task)

    assert result.matched is True
    assert result.metadata["mode"] == "remove_isolated_noise"


def test_composed_rule_search_detects_extract_then_mirror(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 1, 2, 0],
                    [0, 0, 0, 0],
                ],
                "output": [[0, 1], [2, 1]],
            }
        ]
    }

    result = ComposedRuleSearch().match(task)

    assert result.matched is True
    assert result.metadata["finisher"] == "mirror_horizontal"
    assert result.metadata["builder_available"] is True
    _assert_rule_builds_and_validates(ComposedRuleSearch(), task, tmp_path)


def test_candidate_discovery_report_records_probe_match(tmp_path) -> None:
    data_dir = tmp_path / "task"
    data_dir.mkdir()
    (data_dir / "task001.json").write_text(
        """
{
  "train": [
    {
      "input": [[0, 0, 0], [0, 8, 0], [0, 0, 0]],
      "output": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    }
  ],
  "test": [{"input": [[0]]}]
}
""".strip(),
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    summary = report_dir / "summary.csv"
    summary.write_text(
        "task_id,status,best_rule,model_path,num_candidates,num_valid_candidates,estimated_cost,estimated_score,file_size_bytes,failure_reasons\n"
        "task001,failed,,,,0,,,,[]\n",
        encoding="utf-8",
    )
    output = report_dir / "candidate_discovery_report.csv"

    rows = build_candidate_discovery_report(
        data_dir=str(data_dir),
        summary_path=str(summary),
        log_dir=str(tmp_path / "logs"),
        report_path=str(output),
    )

    assert output.is_file()
    assert any(row["candidate_rule"] == "ObjectEditRule" for row in rows)


def test_panel_select_by_color_rule_builds_static_panel_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [1, 0, 9, 0, 2],
                    [1, 1, 9, 2, 2],
                ],
                "output": [
                    [0, 2],
                    [2, 2],
                ],
            },
            {
                "input": [
                    [3, 0, 9, 0, 2],
                    [3, 3, 9, 2, 2],
                ],
                "output": [
                    [0, 2],
                    [2, 2],
                ],
            },
        ]
    }

    rule = PanelSelectByColorRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["panel_index"] == 1
    assert result.metadata["selector"] == "contains_unique_color_2"
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_periodic_extension_color_map_rule_builds_valid_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [[1, 2], [3, 4]],
                "output": [
                    [1, 2, 1, 2, 1],
                    [3, 4, 3, 4, 3],
                    [1, 2, 1, 2, 1],
                ],
            }
        ]
    }

    rule = PeriodicExtensionColorMapRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["period_y"] == 2
    assert result.metadata["period_x"] == 2
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_periodic_extension_color_map_rule_infers_row_period_per_case(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 1, 0],
                    [1, 1, 0],
                ],
                "output": [
                    [0, 2, 0],
                    [2, 2, 0],
                    [0, 2, 0],
                    [0, 2, 2],
                    [0, 2, 0],
                    [2, 2, 0],
                    [0, 2, 0],
                    [0, 2, 2],
                    [0, 2, 0],
                ],
            },
            {
                "input": [
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                ],
                "output": [
                    [0, 2, 0],
                    [2, 0, 2],
                    [0, 2, 0],
                    [2, 0, 2],
                    [0, 2, 0],
                    [2, 0, 2],
                    [0, 2, 0],
                    [2, 0, 2],
                    [0, 2, 0],
                ],
            },
            {
                "input": [
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                ],
                "output": [
                    [0, 2, 0],
                    [2, 2, 0],
                    [0, 2, 0],
                    [0, 2, 0],
                    [2, 2, 0],
                    [0, 2, 0],
                    [0, 2, 0],
                    [2, 2, 0],
                    [0, 2, 0],
                ],
            },
        ]
    }

    rule = PeriodicExtensionColorMapRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["mode"] == "auto_row_period"
    assert result.metadata["periods"] == [4, 2, 3]
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_local_neighborhood_fill_rule_builds_valid_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                ],
                "output": [
                    [0, 1, 0],
                    [0, 2, 0],
                    [0, 1, 0],
                ],
            }
        ]
    }

    rule = LocalNeighborhoodFillRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["background_color"] == 0
    assert result.metadata["fill_color"] == 2
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_hole_fill_rule_builds_valid_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 1, 1],
                    [1, 1, 1, 1],
                ],
                "output": [
                    [1, 1, 1, 1],
                    [1, 2, 2, 1],
                    [1, 2, 1, 1],
                    [1, 1, 1, 1],
                ],
            },
            {
                "input": [
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1],
                ],
                "output": [
                    [1, 1, 1, 1, 1],
                    [1, 2, 2, 2, 1],
                    [1, 2, 1, 2, 1],
                    [1, 1, 1, 1, 1],
                ],
            },
        ]
    }

    rule = HoleFillRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["background_color"] == 0
    assert result.metadata["fill_color"] == 2
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_local_neighborhood_fill_rule_supports_variable_case_shapes(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 1, 0],
                    [0, 0, 0],
                    [0, 1, 0],
                ],
                "output": [
                    [0, 1, 0],
                    [0, 2, 0],
                    [0, 1, 0],
                ],
            },
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 2, 0, 0],
                    [0, 1, 0, 0],
                ],
            },
        ]
    }

    rule = LocalNeighborhoodFillRule()
    result = rule.match(task)

    assert result.matched is True
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_substructure_extract_rule_supports_color_map(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 2, 0],
                    [0, 3, 4, 0],
                ],
                "output": [[5, 6], [7, 8]],
            }
        ]
    }

    rule = SubstructureExtractRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["top"] == 1
    assert result.metadata["left"] == 1
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_substructure_extract_rule_supports_color_bbox(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 5, 5, 0],
                    [0, 5, 0, 0],
                ],
                "output": [[5, 5], [5, 0]],
            }
        ]
    }

    rule = SubstructureExtractRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["kind"] == "color_bbox"
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_tile_from_bbox_repeat_rule_builds_valid_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 2, 0],
                    [0, 3, 4, 0],
                ],
                "output": [
                    [1, 2, 1, 2, 1],
                    [3, 4, 3, 4, 3],
                    [1, 2, 1, 2, 1],
                ],
            }
        ]
    }

    rule = TileFromBBoxRepeatRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["tile_height"] == 2
    assert result.metadata["tile_width"] == 2
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_tile_from_bbox_repeat_rule_supports_truncated_last_tile(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 2, 3],
                    [0, 4, 5, 6],
                ],
                "output": [
                    [1, 2, 3, 1],
                    [4, 5, 6, 4],
                    [1, 2, 3, 1],
                ],
            }
        ]
    }

    rule = TileFromBBoxRepeatRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["tile_height"] == 2
    assert result.metadata["tile_width"] == 3
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_multi_step_translation_rule_builds_valid_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                ],
                "output": [
                    [0, 0, 1, 2],
                    [0, 0, 5, 6],
                ],
            }
        ]
    }

    rule = MultiStepTranslationRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["dy"] == 0
    assert result.metadata["dx"] == 2
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_multi_step_translation_rule_supports_single_color_object(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 2, 0, 0],
                    [0, 2, 0, 0],
                    [3, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 2],
                    [0, 0, 0, 2],
                    [3, 0, 0, 0],
                ],
            }
        ]
    }

    rule = MultiStepTranslationRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["mode"] == "single_color"
    assert result.metadata["dx"] == 2
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_multi_step_translation_rule_supports_variable_shape_single_color_object(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 2, 0, 0],
                    [0, 2, 0, 0],
                    [3, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 2],
                    [0, 0, 0, 2],
                    [3, 0, 0, 0],
                ],
            },
            {
                "input": [
                    [0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [3, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0],
                    [3, 0, 0, 0, 0],
                    [0, 0, 0, 2, 0],
                ],
            },
        ]
    }

    rule = MultiStepTranslationRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["mode"] == "single_color"
    assert result.metadata["dynamic_active"] is True
    assert result.metadata["dx"] == 2
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_multi_step_translation_rule_supports_variable_case_shapes(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
                "output": [
                    [0, 0, 1],
                    [0, 0, 4],
                ],
            },
            {
                "input": [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 1, 2, 3],
                ],
                "output": [
                    [0, 0, 1, 2],
                    [0, 0, 5, 6],
                    [0, 0, 9, 1],
                ],
            },
        ]
    }

    rule = MultiStepTranslationRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["dynamic_active"] is True
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_symmetry_completion_rule_builds_valid_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [1, 0, 0],
                    [2, 0, 0],
                    [3, 0, 0],
                ],
                "output": [
                    [1, 0, 1],
                    [2, 0, 2],
                    [3, 0, 3],
                ],
            }
        ]
    }

    rule = SymmetryCompletionRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["mode"] == "horizontal"
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_local_neighborhood_fill_rule_supports_5x5_kernel(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                ],
                "output": [
                    [1, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 1],
                ],
            }
        ]
    }

    rule = LocalNeighborhoodFillRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["offset_name"] in {"all24", "diagonal8"}
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_local_neighborhood_rewrite_rule_deletes_isolated_cell(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                "output": [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            }
        ]
    }

    rule = LocalNeighborhoodRewriteRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["target_color"] == 1
    assert result.metadata["replacement_color"] == 0
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_local_neighborhood_rewrite_rule_supports_variable_case_shapes(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0],
                ],
                "output": [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
            },
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
            },
        ]
    }

    rule = LocalNeighborhoodRewriteRule()
    result = rule.match(task)

    assert result.matched is True
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_rectangle_and_line_rule_static_bbox_fill_builds_valid_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0],
                    [0, 2, 2, 0],
                    [0, 2, 2, 0],
                    [0, 0, 0, 0],
                ],
            }
        ]
    }

    result = RectangleAndLineRule().match(task)

    assert result.matched is True
    assert result.metadata["mode"] == "bbox_fill"
    assert "static_draw_mask" in result.metadata
    _assert_rule_builds_and_validates(RectangleAndLineRule(), task, tmp_path)


def test_rectangle_and_line_rule_builds_active_frame_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                "output": [
                    [8, 8, 8, 8],
                    [8, 0, 0, 8],
                    [8, 8, 8, 8],
                ],
            },
            {
                "input": [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                "output": [
                    [8, 8, 8],
                    [8, 0, 8],
                    [8, 0, 8],
                    [8, 8, 8],
                ],
            },
        ]
    }

    rule = RectangleAndLineRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["mode"] == "active_frame"
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_rectangle_and_line_rule_static_horizontal_connect_builds_valid_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 3, 0, 3, 0],
                    [0, 0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0, 0],
                    [0, 3, 3, 3, 0],
                    [0, 0, 0, 0, 0],
                ],
            },
            {
                "input": [
                    [0, 0, 0, 0, 0],
                    [0, 3, 0, 3, 0],
                    [0, 0, 1, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0, 0],
                    [0, 3, 3, 3, 0],
                    [0, 0, 1, 0, 0],
                ],
            },
        ]
    }

    rule = RectangleAndLineRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata["mode"] == "connect_two_points_horizontal"
    assert "static_draw_mask" in result.metadata
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_rectangle_and_line_rule_horizontal_extend_builds_valid_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 4, 4, 0],
                    [0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0],
                    [4, 4, 4, 4],
                    [0, 0, 0, 0],
                ],
            },
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 4, 4, 0],
                    [0, 0, 0, 0],
                ],
                "output": [
                    [0, 0, 0, 0],
                    [4, 4, 4, 4],
                    [0, 0, 0, 0],
                ],
            },
        ]
    }

    rule = RectangleAndLineRule()
    result = rule.match(task)

    assert result.matched is True
    assert result.metadata == {"mode": "extend_line", "direction": "horizontal", "draw_color": 4}
    _assert_rule_builds_and_validates(rule, task, tmp_path)


def test_object_selection_rule_probe_detects_largest_component() -> None:
    task = {
        "train": [
            {
                "input": [
                    [2, 0, 0, 1],
                    [2, 2, 0, 2],
                    [0, 0, 0, 2],
                ],
                "output": [
                    [2, 0],
                    [2, 2],
                ],
            },
            {
                "input": [
                    [2, 0, 1, 0],
                    [2, 2, 0, 2],
                    [0, 0, 0, 2],
                ],
                "output": [
                    [2, 0],
                    [2, 2],
                ],
            },
        ]
    }

    result = ObjectSelectionRule().match(task)

    assert result.matched is True
    assert result.metadata["kind"] == "color_component"
    assert result.metadata["selector"] == "largest_area"
    assert result.metadata["top"] == 0
    assert result.metadata["left"] == 0


def test_object_selection_rule_builds_static_component_model(tmp_path) -> None:
    task = {
        "train": [
            {
                "input": [
                    [2, 0, 0, 1],
                    [2, 2, 0, 2],
                    [0, 0, 0, 2],
                ],
                "output": [
                    [2, 0],
                    [2, 2],
                ],
            },
            {
                "input": [
                    [2, 0, 1, 0],
                    [2, 2, 0, 2],
                    [0, 0, 0, 2],
                ],
                "output": [
                    [2, 0],
                    [2, 2],
                ],
            },
        ]
    }

    _assert_rule_builds_and_validates(ObjectSelectionRule(), task, tmp_path)
