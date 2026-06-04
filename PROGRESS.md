# 当前进度

## 2026-06-04 - Medium/high-risk probe-only round after low online gain

- User reported the full-model initializer cleanup improved the online score by
  only about `+0.03`.
- Decision: medium/high-risk work is now justified, but only as isolated
  probe/candidate generation. No unverified semantic replacement was copied to
  `outputs/onnx` or `outputs/submission.zip`.
- Current final submission remained unchanged and usable:
  - `python -m src.inspect_submission --zip outputs\submission.zip`: passed.
  - ONNX count: 400.
- Formal replacement search over strategy targets:
  - targets: `task133`, `task076`, `task157`, `task233`, `task366`,
    `task363`, `task319`
  - searched formal rules: 37
  - report rows: 259
  - replacement count: 0
  - report: `outputs/reports/replacement_search_report_medium_risk_targets.csv`
- Extended `src.high_risk_ablation_probes` with a new probe-only hypothesis:
  - `two_panel_marker_object_transfer`
  - It splits the input into two equal panels, treats the sparse-marker panel
    as the target, and copies source-panel objects whose marker-color layouts
    match the target markers.
  - This is not in `first_version_rules()` and does not build ONNX.
- Probe results:
  - `task366/two_panel_marker_object_transfer`: train 3/3, test 1/1,
    arc-gen 262/262.
  - All other strategy-target probe rows rejected on train.
  - `task363/horizontal_zero_runs_by_marker_length` still has train 0/3
    despite arc-gen 45/261, so it is not usable.
- Reports:
  - `outputs/reports/high_risk_ablation_probe_report_task366_panel_transfer.csv`
  - `outputs/reports/high_risk_ablation_probe_report_strategy_targets.csv`
  - earlier focused rejects:
    `outputs/reports/high_risk_ablation_probe_report_task133_task363.csv`
- Validation:
  - `python -m compileall src`: passed.
  - `python -m src.inspect_submission --zip outputs\submission.zip`: passed.
- No ablation zip was generated in this round because no ONNX builder/candidate
  exists yet for the passing `task366` probe. Next step is an isolated
  `task366` builder attempt under `outputs/candidates/`, followed by strict
  validation and a one-task ablation zip only if the generated ONNX is smaller
  and exact.

## 2026-06-04 - Full-model graph-equivalent initializer cleanup

- Followed `优化策略.md` low-risk path: extended initializer cleanup from the
  previous top-5 pass to all 400 ONNX models.
- Updated `src.deduplicate_initializers`:
  - `--task-ids` now defaults to all `task*.onnx` in the model directory.
  - Removes unreferenced non-input initializers.
  - Still merges byte-identical initializer tensors and rewires node inputs to
    canonical names.
  - Preserves graph inputs/outputs and does not add semantic rules.
- Full scan report:
  - `outputs/reports/deduplicate_initializers_all.csv`
  - output candidates: `outputs/candidates/deduplicated_all/`
  - improved tasks: 24 / 400
  - total estimated cost delta: -173,602
  - total ONNX file-size delta: -248,824 bytes
- Largest promoted cost reductions:
  - `task367`: 295,949 -> 219,324, delta -76,625
  - `task319`: 78,739 -> 26,814, delta -51,925
  - `task153`: 26,811 -> 3,320, delta -23,491
  - `task366`: 266,691 -> 260,211, delta -6,480
  - `task285`: 15,191 -> 10,215, delta -4,976
- Promoted all 24 graph-equivalent improved candidates into `outputs/onnx`.
- Also generated one-task ablation zips for the 24 promoted candidates:
  - directory: `outputs/ablation_submissions/dedup_all/`
  - report: `outputs/reports/ablation_submission_report_dedup_all.csv`
  - valid zips: 24 / 24
- Current final submission after trusted rebuild:
  - Path: `outputs/submission.zip`
  - ONNX count: 400
  - Zip size: 1,312,206 bytes
  - Estimated cost total: 6,661,880
  - ONNX file size total: 10,981,062 bytes
  - `python -m src.inspect_submission --zip outputs\submission.zip`: passed.
- Validation:
  - `python -m pytest -q tests\test_deduplicate_initializers.py`: 2 passed.
  - `python -m pytest -q tests\test_deduplicate_initializers.py tests\test_build_current_model_submission.py tests\test_sync_and_ablation_submissions.py`: 9 passed.
  - `python -m compileall src tests`: passed.
  - `python -m pytest -q`: 80 passed, 2 skipped.
  - `git diff --check`: no whitespace errors; only CRLF warnings.

Remaining top-cost rule targets are still mostly semantic/high-risk:

`task133`, `task076`, `task157`, `task233`, `task366`, `task367`,
`task363`, `task209`, `task396`, `task028`, `task255`.

## 2026-06-04 - Top-5 high-cost initializer dedup optimization

- Current top-5 before this pass:
  - `task133`: 1,406,822
  - `task209`: 1,170,350
  - `task076`: 1,147,810
  - `task157`: 1,023,477
  - `task233`: 668,250
- Added exact graph-preserving optimizer:
  - `src.deduplicate_initializers`
  - Merges byte-identical ONNX initializers and rewires node inputs to the
    canonical initializer.
  - Does not change node topology or tensor values.
- Top-5 dedup results:
  - `task209`: 1,170,350 -> 144,226, delta -1,026,124
  - `task157`: 1,023,477 -> 1,008,489, delta -14,988
  - `task233`: 668,250 -> 661,410, delta -6,840
  - `task133`: no duplicate-initializer gain
  - `task076`: no duplicate-initializer gain
- Extra validation for promoted dedup models:
  - `task209`: train 3/3, test 1/1, arc-gen 262/262 passed.
  - `task157`: train 2/2, test 1/1, arc-gen 262/262 passed.
  - `task233`: train 3/3, test 1/1, arc-gen 262/262 passed.
- Final model bank changes:
  - Promoted dedup candidates into `outputs/onnx/task209.onnx`,
    `outputs/onnx/task157.onnx`, and `outputs/onnx/task233.onnx`.
  - Rebuilt `outputs/submission.zip` with trusted packaging.
- Current final submission:
  - Path: `outputs/submission.zip`
  - ONNX count: 400
  - Zip size: 1,332,788 bytes
  - Estimated cost total: 6,835,482
  - ONNX file size total: 11,229,886 bytes
  - `python -m src.inspect_submission --zip outputs\submission.zip`: passed.
- Reports:
  - `outputs/reports/deduplicate_initializers_top5.csv`
  - `outputs/reports/high_cost_task_diagnosis_current_top5.csv`
  - `outputs/reports/replacement_search_report_current_top5.csv`
  - `outputs/reports/high_cost_task_diagnosis_after_dedup_top10.csv`
- Validation:
  - `python -m pytest -q tests\test_deduplicate_initializers.py`: passed.
  - `python -m pytest -q`: 79 passed, 2 skipped.

Remaining top costs after this pass:

`task133`, `task076`, `task157`, `task233`, `task367`,
`task366`, `task363`, `task209`, `task396`, `task319`.

## 2026-06-04 - Online ablation winners promoted into final submission

- User submitted the six one-task ablation zips online and reported scores:
  - `task025_DynamicLineProjectionRule.zip`: 6031.51
  - `task028_TwoMarkerHorizontalBandRule.zip`: 6028.80
  - `task084_DiagonalBottomFillRule.zip`: 6032.31
  - `task200_BottomMarkerVerticalStripeRule.zip`: 6031.46
  - `task367_DynamicRectangularCavityFillRule.zip`: 6028.06
  - `task396_DynamicLargestFrameRecolorCropRule.zip`: 6027.99
- Decision against 6029 baseline:
  - Promoted: `task025`, `task084`, `task200`.
  - Rejected for final submission: `task028`, `task367`, `task396`.
- Report written:
  - `outputs/reports/online_ablation_results_6029.csv`
- Final model bank changes:
  - Copied promoted candidates into `outputs/onnx/task025.onnx`,
    `outputs/onnx/task084.onnx`, and `outputs/onnx/task200.onnx`.
  - Rebuilt `outputs/submission.zip` with `--validation-mode trusted`.
  - `task028`, `task367`, and `task396` remain baseline models in the final
    model bank.
- Current final submission:
  - Path: `outputs/submission.zip`
  - ONNX count: 400
  - Zip size: 1,365,198 bytes
  - Estimated cost total: 7,883,434
  - ONNX file size total: 12,209,498 bytes
  - `python -m src.inspect_submission --zip outputs\submission.zip`: passed.
- Expected online effect from independent one-task ablations:
  - baseline 6029 + 2.51 + 3.31 + 2.46 = about 6037.28
  - This is an online-ablation estimate, not a guaranteed leaderboard score.
- Validation:
  - `python -m pytest -q`: 78 passed, 2 skipped.

## 2026-06-04 - Code/model bank realigned to 6029 baseline and ablation workflow

- Answered the model-bank mismatch:
  - Before this round, `outputs/submission.zip` had been restored to the 6029
    baseline, but `outputs/onnx` still contained the locally optimized 5909
    variant.
  - Added `src.sync_model_bank_from_submission` and synchronized
    `outputs/onnx` from the current safe `outputs/submission.zip`.
- Added explicit rebuild modes to `src.build_current_model_submission`:
  - `strict`: existing local train/static/padding validation, used for new
    generated candidates.
  - `trusted`: packages a known online-clean baseline model bank with checker,
    forbidden-op, file-size and cost checks, without rejecting baseline models
    that fail local padding/static-shape heuristics.
- Rebuilt current submission with trusted mode:
  - selected tasks: 400 / 400
  - missing or invalid tasks: 0
  - estimated cost total: 10,594,016
  - ONNX file size total: 14,849,987 bytes
  - `python -m src.inspect_submission --zip outputs\submission.zip`: passed,
    400 ONNX models.
- Added `src.build_ablation_submissions` for online-safe optimization:
  - Builds one submission zip per candidate.
  - Each zip replaces exactly one baseline task model and keeps the other 399
    baseline models unchanged.
  - This avoids bundling many unproven local replacements into the final
    submission.
- Baseline 6029 local replacement candidates:
  - `task084`: `DiagonalBottomFillRule`, cost 1,390,970 -> 722.
  - `task200`: `BottomMarkerVerticalStripeRule`, cost 990,050 -> 992.
  - `task025`: `DynamicLineProjectionRule`, cost 332,565 -> 1,289.
  - `task367`: `DynamicRectangularCavityFillRule`, cost 295,949 -> 200,585.
  - `task028`: `TwoMarkerHorizontalBandRule`, cost 63,050 -> 22,600.
  - `task396`: `DynamicLargestFrameRecolorCropRule`, cost 115,080 -> 6,009.
- Generated ablation zips under `outputs/ablation_submissions/`:
  - `task025_DynamicLineProjectionRule.zip`
  - `task028_TwoMarkerHorizontalBandRule.zip`
  - `task084_DiagonalBottomFillRule.zip`
  - `task200_BottomMarkerVerticalStripeRule.zip`
  - `task367_DynamicRectangularCavityFillRule.zip`
  - `task396_DynamicLargestFrameRecolorCropRule.zip`
  - All 6 passed `src.inspect_submission`.
- Reports:
  - `outputs/reports/replacement_search_report_6029_baseline.csv`
  - `outputs/reports/replacement_search_report_6029_followup.csv`
  - `outputs/reports/ablation_submission_report_6029.csv`
- Validation:
  - `python -m compileall src tests`: passed.
  - `python -m pytest -q`: 78 passed, 2 skipped.
  - `python -m src.inspect_submission --zip outputs\submission.zip`: passed.
  - `git diff --check` reports whitespace false positives inside binary ONNX
    diffs; scoped check excluding ONNX passed with only CRLF warnings.

Current safe final submission:

- `outputs/submission.zip` remains the 6029-baseline-aligned artifact.
- Do not promote any ablation zip to `outputs/submission.zip` until its online
  leaderboard score is confirmed to be at least 6029.

## 2026-06-04 - Online regression rollback to 6029 baseline

- User-reported leaderboard scores:
  - current locally optimized submission: 5909
  - original baseline submission: 6029
- Compared `submission（原baseline结果）.zip` against pre-rollback
  `outputs/submission.zip`.
  - Both zips contained 400 ONNX files.
  - Missing/extra tasks: 0.
  - Identical task models: 361.
  - Different task models: 39.
  - Baseline local estimated cost total: 10,594,016.
  - Pre-rollback current local estimated cost total: 7,575,450.
  - Baseline local estimated score total: 7088.218589.
  - Pre-rollback current local estimated score total: 7134.687555.
- Diagnosis:
  - The local optimizer selected lower-cost models, so local estimated score
    increased by about 46.47.
  - The leaderboard dropped by 120 points, which means at least some changed
    models were not online-equivalent to the original baseline despite passing
    local train validation and, for some tasks, extra labelled local splits.
  - Without per-task online ablation, the conservative fix is to remove all 39
    changed models from the final submission and restore the exact known
    6029-scoring baseline zip.
- Restored final submission:
  - `outputs/submission.zip` is now byte-for-byte equivalent at task content
    level to the original baseline zip.
  - Post-restore comparison: 400 identical task models, 0 differing task
    models.
  - Zip size: 1,467,677 bytes.
  - `python -m src.inspect_submission --zip outputs\submission.zip`: passed,
    400 ONNX models.
- Reports written:
  - `outputs/reports/submission_baseline_vs_current_diff.csv`
  - `outputs/reports/submission_baseline_vs_current_diff_after_restore.csv`

Important note:

- `outputs/submission.zip` is the current safe submission artifact for online
  use. The `outputs/onnx` model bank still contains the locally optimized
  experimental replacements from the 5909 submission; rebuilding from that bank
  will recreate the lower-online-score variant unless the differing tasks are
  separately reverted or an ablation-driven whitelist is established.

## 2026-06-03 - high-cost model bank optimization: task025, task367, task028

- Current validated submission: `outputs/submission.zip`
  - `python -m src.inspect_submission --zip outputs\submission.zip`: passed
  - ONNX count: 400
  - Zip size: 1379041 bytes
  - Selected tasks: 400 / 400
  - Missing or invalid tasks: 0
- Current model bank report: `outputs/reports/current_model_bank_report.csv`
  - Estimated cost total: 7575450
  - ONNX file size total: 12203846 bytes

This round's accepted high-cost replacements:

- `task025`: `DynamicLineProjectionRule`
  - old estimated cost: 332565
  - new estimated cost: 1289
  - delta: 331276
  - extra validation: train 3/3, test 1/1, arc-gen 262/262 passed
- `task367`: `DynamicRectangularCavityFillRule`
  - old estimated cost: 295949
  - new estimated cost: 200585
  - delta: 95364
  - extra validation: train 3/3, test 1/1, arc-gen 262/262 passed
- `task028`: `TwoMarkerHorizontalBandRule`
  - old estimated cost: 63050
  - new estimated cost: 22600
  - delta: 40450
  - extra validation: train 2/2, test 1/1, arc-gen 262/262 passed

New formal rules/builders added:

- `DynamicLineProjectionRule` / `build_dynamic_line_projection_model`
  - Detects a full same-color horizontal or vertical line and projects stray
    cells of that color to the adjacent row/column next to the line.
- `DynamicRectangularCavityFillRule` /
  `build_dynamic_rectangular_cavity_fill_model`
  - Fills rectangular color-0 cavities bounded by color-5 top/bottom walls and
    color-5 or grid-boundary side walls, with a side-boundary termination check
    to avoid filling exterior gaps.
- `TwoMarkerHorizontalBandRule` / `build_two_marker_horizontal_bands_model`
  - For shared-shape two-marker tasks, extracts marker colors dynamically and
    draws fixed top/bottom horizontal band frames.

Remaining highest-cost tasks after rebuild:

`task133`, `task209`, `task076`, `task157`, `task233`, `task366`,
`task367`, `task363`, `task319`, `task255`.

Notes:

- Formal replacement search after the new rules found no additional safe
  replacement for `task133`, `task209`, `task076`, `task157`, `task233`,
  `task366`, `task363`, `task319`, or `task028` before `task028` was handled
  separately.
- `task319`, `task076`, `task157`, and `task255` were inspected manually. They
  appear to require more complex object selection, orientation-aware copy, or
  large-region fill semantics, so no unsafe MATCH rule was promoted.
- A first rebuild with 120s per-model timeout excluded `task191` due to a
  transient `evaluation_timeout`; standalone strict validation passed, and the
  final rebuild used 300s per-model timeout and selected all 400 tasks.

Validation commands used:

```powershell
python -m pytest -q tests\test_pattern_rules.py -k line_projection
python -m pytest -q tests\test_pattern_rules.py -k "rectangular_cavity or two_marker_horizontal"
python -m src.search_symbolic_replacements --data-dir task --current-model-dir outputs\onnx --current-report outputs\reports\current_model_bank_report.csv --candidate-dir outputs\candidates\replacements --report outputs\reports\replacement_search_report_task025.csv --task-ids task025 --replace --timeout-seconds 120
python -m src.search_symbolic_replacements --data-dir task --current-model-dir outputs\onnx --current-report outputs\reports\current_model_bank_report.csv --candidate-dir outputs\candidates\replacements --report outputs\reports\replacement_search_report_task367.csv --task-ids task367 --replace --timeout-seconds 120
python -m src.search_symbolic_replacements --data-dir task --current-model-dir outputs\onnx --current-report outputs\reports\current_model_bank_report.csv --candidate-dir outputs\candidates\replacements --report outputs\reports\replacement_search_report_task028.csv --task-ids task028 --replace --timeout-seconds 120
python -m pytest -q tests\test_pattern_rules.py
python -m pytest -q tests\test_high_cost_replacement_search.py
python -m compileall src tests
python -m src.build_current_model_submission --data-dir task --model-dir outputs\onnx --validated-dir outputs\current_model_bank_verified_onnx --report outputs\reports\current_model_bank_report.csv --zip outputs\submission.zip --timeout-seconds 300
python -m src.inspect_submission --zip outputs\submission.zip
python -m pytest -q
git diff --check
```

Validation results:

- `tests\test_pattern_rules.py`: 60 passed.
- `tests\test_high_cost_replacement_search.py`: 2 passed.
- Full pytest: 74 passed, 2 skipped.
- `python -m compileall src tests`: passed.
- `inspect_submission`: passed, 400 ONNX models.
- `git diff --check`: no whitespace errors; only LF-to-CRLF warnings.

## 2026-06-01 官方处理错误回退与 padding 修复候选

- 用户反馈官方处理 `task099`, `task180`, `task266`, `task283`, `task331` 失败。
- 已先按要求从当前主提交 `outputs/submission.zip` 排除这五个任务。
- Current safe submission: `outputs/submission.zip`
  - Submission inspection: passed
  - ONNX count: 383
  - Zip size: 1396158 bytes
  - 已确认不包含 `task099.onnx`, `task180.onnx`, `task266.onnx`, `task283.onnx`, `task331.onnx`
- 根因判断:
  - 第一版 padding repair 把 graph output 名从 `output` 改成了 `masked_output`。
  - 本地 validator 能运行，但官方处理器很可能要求固定 output 名。
- 已修复 repair 逻辑:
  - 保留 graph input 名 `input` 和 graph output 名 `output`。
  - 将原输出 producer 改名为 `output_unmasked`，再追加 `Mul(output_unmasked, mask) -> output`。
  - mask dtype 跟随原 graph output dtype，修复 `float16` 模型的 `Mul(float16, float)` 类型错误。
- 新增动态 active-mask repair:
  - 用 1x1 `Conv` 从 input one-hot 推导真实 active grid 区域。
  - 适用于 input/output 同尺寸但 train shape 可变的 padding 非零任务。

候选优化包:
- Candidate submission: `outputs/submission_candidate_active_static.zip`
- Candidate inspection: passed
- Candidate ONNX count: 393
- Candidate zip size: 1448132 bytes
- Candidate source counts:
  - archive/repaired archive: 377
  - current local optimized models: 16
- Candidate selected estimated cost total: 10494589
- Candidate selected ONNX file size total: 14703185 bytes
- 新增 candidate 修复任务:
  - `task004`, `task098`, `task099`, `task120`, `task122`, `task180`, `task266`, `task283`, `task331`, `task344`

仍未解决的 7 个任务:
- `task042`, `task094`, `task168`, `task184`, `task224`, `task288`: archive 在本地 ORT 子进程中返回 `3221225477`，当前不纳入主提交。
- `task277`: archive 存在动态 shape，当前不纳入主提交。

## 2026-06-01 archive baseline 混合提交

- External baseline source: `archive/` / `6029-09-lb-neurogolf-all-task-onnx-solution.ipynb`
- Blended report: `outputs/reports/archive_blend_report.csv`
- Padding repair report: `outputs/reports/archive_padding_repair_report.csv`
- Blended ONNX dir: `outputs/archive_blended_onnx`
- Current final submission: `outputs/submission.zip`
- Submission inspection: passed
- Submission ONNX count: 388
- Submission zip size: 1420939 bytes
- Selected source counts:
  - archive/repaired archive: 372
  - current local optimized models: 16
- Blended selected estimated cost total: 10474834
- Blended selected ONNX file size total: 14683926 bytes

说明:
- `outputs/reports/summary.csv` 仍表示规则求解器自身的 53 solved 结果。
- `outputs/reports/archive_blend_report.csv` 才是本轮 archive baseline 混合提交的记录。
- 本轮仍按本地严格验证执行: ONNX checker、forbidden ops、static shape、file size、onnxruntime train exact validation、zero-confidence 和 nonzero-padding 检查。
- 未通过本地严格验证的 archive 模型未进入 `submission.zip`。

剩余 12 个未进入 blended submission 的任务:
- `task004`, `task098`, `task120`, `task122`, `task344`: archive active grid 可疑但 padding 非零，且 train output shape 可变，未做静态 mask 修复。
- `task042`, `task094`, `task168`, `task184`, `task224`, `task288`: archive 在 ORT 子进程中返回 `3221225477`，视为运行时不可用。
- `task277`: archive shape inference 出现动态 shape，违反静态 shape 约束。

已修复的 fixed-shape padding archive 模型:
- `task099`, `task180`, `task266`, `task283`, `task331`
- 另有若干 archive fixed-shape padding 模型被修复后仍因当前本地模型 cost 更低而未被选中。

## 2026-06-01 五个优化方向落地结果

- Local train solved: 53 / 400
- Failed: 347 / 400
- 本轮新增 solved: `task065`, `task207`
- 新增 solved 规则: `DynamicQuadrantPanelSelectRule`
- solved 模型 estimated cost 总和: 369897
- solved 模型 ONNX file size 总和: 503527 bytes
- Submission: `outputs/submission.zip`
- Submission size: 60236 bytes
- 注意: 以上是本地 train validation 和本地 estimated cost，不是官方榜单分数。

### 本轮完成的五个优化方向

1. 剩余 7 个 candidate-discovery 任务优先处理
   - 已解决 `task065`, `task207`。
   - `task036`, `task079`, `task174` 仍有 probe 命中但未安全编译。
   - `task100`, `task153` 在本轮重建后不再出现在 candidate discovery 剩余列表中。
2. PanelSemanticRule -> dynamic quadrant selector
   - 新增正式 `DynamicQuadrantPanelSelectRule`。
   - 新增 `build_dynamic_quadrant_panel_select_model()`。
   - 支持 odd square 2x2 center-cross layout，按 unique max panel-difference 动态选择 quadrant，并支持 color map。
   - 通过 ONNX checker、onnxruntime train validation、submission inspection。
3. FrameInteriorRule probe -> builder
   - 新增 `build_dynamic_frame_interior_crop_model()`。
   - `FrameInteriorRule` 支持 buildable 的 `frame_interior_crop + color_map`。
   - 为安全起见，仅当 frame_color 的整体 bbox 等于矩形 frame bbox 时允许 builder；否则记录 `frame_color_bbox_contains_extra_cells`，不进入候选构建。
4. DynamicBBoxCropRule -> color-specific bbox
   - 新增 `build_dynamic_color_bbox_crop_model()`。
   - `DynamicBBoxCropRule` 支持 buildable 子集: `bbox_of_all_non_background`, `bbox_of_color`, `bbox_of_unique_color_component`，并支持 identity / horizontal mirror / vertical mirror + color map。
   - 对 component-not-touching-border 等尚无安全 ONNX builder 的子集保留 probe 记录，不构建。
5. ComposedRuleSearch -> safe extractor -> finisher
   - `ComposedRuleSearch` 支持安全子集: buildable bbox extractor -> identity / horizontal mirror / vertical mirror -> color map。
   - 非安全 extractor/finisher 仍记录 `requires_composed_rule`，不会生成正式候选。
   - `solve_task` 现在会显式跳过 `builder_available=False` 的匹配，只记录 blocked reason，避免 probe-only 分支抛异常。

### 本轮验证命令

```powershell
python -m pytest tests\test_pattern_rules.py -q
python -m pytest -q
python -m compileall src tests
python -m src.build_submission
python -m src.inspect_submission --zip outputs\submission.zip
python -m src.failure_taxonomy
python -m src.candidate_discovery_report
python -m src.probe_rules --data-dir task --report outputs\reports\probe_summary.csv --summary outputs\reports\summary.csv
git diff --check
```

验证结果:
- targeted pattern tests: 52 passed
- full pytest: 61 passed
- compileall: passed
- full submission rebuild: 400 tasks processed, 53 solved, 347 failed
- submission inspection: passed, 53 ONNX models
- failure taxonomy rows: 347
- rule near-miss rows: 347
- candidate discovery rows: 6
- probe summary scanned failed tasks: 347
- `git diff --check`: passed with line-ending warnings only

### 当前剩余 candidate discovery

- `task036`: `DynamicBBoxCropRule` blocked by `builder_missing_dynamic_bbox`; `FrameInteriorRule` blocked by `frame_color_bbox_contains_extra_cells`; `ComposedRuleSearch` blocked by `requires_composed_rule`
- `task079`: `FrameInteriorRule` blocked by `frame_color_bbox_contains_extra_cells`
- `task174`: `DynamicBBoxCropRule` blocked by `builder_missing_dynamic_bbox`; `ComposedRuleSearch` blocked by `requires_composed_rule`

## 当前目标

继续提高本地 train solved 数，同时保持 ONNX 合法性、严格验证、可复现性和 submission 安全。正确性优先于模型大小；probe-only 策略不得进入 `submission.zip`。

## 当前状态

- Local train solved: 51 / 400
- Failed: 349 / 400
- solved 模型 estimated cost 总和: 182027
- solved 模型 ONNX file size 总和: 214159 bytes
- Submission: `outputs/submission.zip`
- Submission size: 35042 bytes
- Summary: `outputs/reports/summary.csv`
- Failure taxonomy: `outputs/reports/failure_taxonomy.csv`
- Rule near-miss: `outputs/reports/rule_near_miss.csv`
- Candidate discovery: `outputs/reports/candidate_discovery_report.csv`
- Probe summary: `outputs/reports/probe_summary.csv`

## 本轮新增 solved

- `task031` -> DynamicNonBackgroundBBoxCropRule, cost 1113, file 3759
- `task150` -> DynamicActiveMirrorRule, cost 622, file 2887
- `task155` -> DynamicActiveMirrorRule, cost 622, file 2886
- `task177` -> DynamicNonBackgroundBBoxCropRule, cost 1113, file 3769
- `task259` -> DynamicNonBackgroundBBoxCropRule, cost 1113, file 3759
- `task290` -> DynamicBBoxExtremeColorSwapRule, cost 713, file 4864

## 本轮完成内容

- 新增 `PanelSemanticRule`, `DynamicBBoxCropRule`, `FrameInteriorRule`, `ObjectEditRule`, `ComposedRuleSearch` probe，并生成 `candidate_discovery_report.csv`。
- 新增正式 `DynamicNonBackgroundBBoxCropRule`：
  - 支持动态 active bbox crop。
  - 支持 background color 不固定为 0。
  - 支持 identity / horizontal mirror / vertical mirror。
- 新增正式 `DynamicActiveMirrorRule`：
  - 支持不同输入尺寸的 same-size horizontal/vertical mirror。
  - 不再要求所有 train case 共享同一 grid shape。
- 新增正式 `DynamicBBoxExtremeColorSwapRule`：
  - 支持非背景 bbox crop 后交换 bbox 内最多/最少颜色角色。
  - 不硬编码具体颜色值。
- 以上正式规则均已通过 ONNX checker、onnxruntime train validation、禁用算子、静态 shape、文件大小和 submission 检查。

## 本轮验证命令

```powershell
python -m compileall src tests
python -m pytest -q
python -m src.build_submission
python -m src.inspect_submission --zip outputs\submission.zip
python -m src.failure_taxonomy
python -m src.candidate_discovery_report
python -m src.probe_rules --data-dir task --report outputs\reports\probe_summary.csv --summary outputs\reports\summary.csv
git diff --check
```

验证结果：

- `compileall`: passed
- `pytest`: 57 passed
- full submission rebuild: 400 tasks processed, 51 solved, 349 failed
- submission inspection: passed, 51 ONNX models
- failure taxonomy rows: 349
- rule near-miss rows: 349
- candidate discovery rows: 11
- probe summary scanned failed tasks: 349
- `git diff --check`: passed

## Candidate Discovery 结果

`outputs/reports/candidate_discovery_report.csv` 当前剩余 7 个失败任务、11 条 train 可解释 probe 候选：

- FrameInteriorRule: 5
- DynamicBBoxCropRule: 2
- ComposedRuleSearch: 2
- PanelSemanticRule: 2

剩余命中任务：

`task036`, `task065`, `task079`, `task100`, `task153`, `task174`, `task207`.

这些剩余候选主要需要动态 panel selector、component selection 或 frame interior builder；尚未安全编译进 submission。

## solved 任务

`task001`, `task002`, `task003`, `task006`, `task016`, `task026`, `task031`, `task048`, `task053`, `task072`, `task073`, `task081`, `task087`, `task095`, `task116`, `task130`, `task135`, `task140`, `task144`, `task147`, `task150`, `task155`, `task164`, `task171`, `task172`, `task177`, `task210`, `task223`, `task236`, `task251`, `task258`, `task259`, `task267`, `task272`, `task276`, `task287`, `task290`, `task291`, `task294`, `task307`, `task309`, `task311`, `task318`, `task326`, `task337`, `task346`, `task352`, `task355`, `task380`, `task385`, `task386`.

## 规则分布

- PanelSeparatorBinaryOpRule: 7
- LocalNeighborhoodFillRule: 6
- ColorMapRule: 5
- MirrorConcatRule: 5
- SubstructureExtractRule: 4
- DynamicNonBackgroundBBoxCropRule: 3
- RotateRule: 3
- HoleFillRule: 2
- CropRule: 2
- DynamicActiveMirrorRule: 2
- ScaleRepeatRule: 2
- SymmetryCompletionRule: 2
- SelfKronMaskRule: 1
- PeriodicExtensionColorMapRule: 1
- OneStepTranslationRule: 1
- MultiStepTranslationRule: 1
- StridedSubsampleRule: 1
- RectangleAndLineRule: 1
- LocalNeighborhoodRewriteRule: 1
- DynamicBBoxExtremeColorSwapRule: 1

## 下一步建议

1. 优先分析 `task065` 和 `task207`，它们是 2x2 panel semantic selection，可能可做成动态 quadrant selection builder。
2. 对 `task036` 和 `task174` 谨慎处理；当前命中依赖 component selection，不应直接用宽松 probe 进入正式规则。
3. `FrameInteriorRule` 剩余 5 个命中里，先找是否存在静态 frame position 或可用 color-role bbox 子类，再写 builder。

## 2026-06-02 - Archive repair reached 400/400 local validated models

- Added reproducible archive repair paths in `src/repair_archive_padding.py`.
- Repaired `task277` by replacing two dynamic Pad pads inputs with static int64 initializers. Repaired model is valid with estimated cost 29994 and file size 57173 bytes.
- Repaired raw archive ORT default crashes for `task042`, `task094`, `task168`, `task184`, `task224`, and `task288` by rewriting Conv nodes with negative `pads` into `Slice` plus Conv with non-negative pads.
- Cost after shared Slice constants:
  - `task042`: cost 1356, file 11833
  - `task094`: cost 910, file 4915
  - `task168`: cost 1667, file 9227
  - `task184`: cost 340, file 7610
  - `task224`: cost 1240, file 7753
  - `task288`: cost 821, file 13869
- Built `outputs/archive_all_repaired_candidate_onnx` with 400 candidate ONNX files.
- Full strict blend output:
  - selected tasks: 400 / 400
  - missing tasks: 0
  - source counts: archive 384, current 16
  - selected estimated cost total: 10530917
  - selected ONNX file size total: 14815565 bytes
- Final validated artifacts:
  - `outputs/archive_all_repaired_verified_onnx`
  - `outputs/reports/archive_all_repaired_blend_report.csv`
  - `outputs/submission_validated_400.zip`
  - `outputs/submission.zip`
- `outputs/submission.zip` inspection passed with 400 ONNX models.

Validation run this round:

```powershell
python -m pytest -q tests\test_repair_archive_padding.py
python -m src.repair_archive_padding --archive-dir archive --output-dir outputs\archive_static_shape_repaired --repair-report outputs\reports\archive_static_shape_repair_report.csv --task-ids task277 --mode task277_static_pads
python -m src.repair_archive_padding --archive-dir archive --output-dir outputs\archive_negative_conv_repaired --repair-report outputs\reports\archive_negative_conv_repair_report.csv --task-ids "task042,task094,task168,task184,task224,task288" --mode negative_conv_pads
python -m src.blend_archive_submission --archive-dir outputs\archive_all_repaired_candidate_onnx --current-dir outputs\onnx --blended-dir outputs\archive_all_repaired_verified_onnx --report outputs\reports\archive_all_repaired_blend_report.csv --zip outputs\submission_validated_400.zip --timeout-seconds 120
python -m src.inspect_submission --zip outputs\submission_validated_400.zip
python -m src.inspect_submission --zip outputs\submission.zip
python -m compileall src tests
python -m pytest -q
git diff --check
```

Notes:

- Local train validation is not a guaranteed official leaderboard score.
- The repaired six raw archive models were already checker/static/forbidden-op clean; the failure was default onnxruntime graph optimization crashing on negative Conv pads.

## 2026-06-02 - Repository cleanup

- Removed Python/pytest caches: `.pytest_cache`, `src/__pycache__`, `tests/__pycache__`.
- Removed duplicate or obsolete generated ONNX directories from earlier archive repair/blend rounds.
- Kept the current validated 400-model directory: `outputs/archive_all_repaired_verified_onnx`.
- Removed duplicate and obsolete zip artifacts, including `outputs/submission_validated_400.zip` after confirming its SHA-256 hash matched `outputs/submission.zip`.
- Removed debug-only artifacts: `debug_pack.zip`, `outputs/blend_debug`, and `outputs/blend_debug_042.zip`.
- Added `.gitignore` entries for caches and generated zip/archive output directories so future runs do not reintroduce these files into git status.

Current retained submission artifact: `outputs/submission.zip`.

## 2026-06-03 - Current ONNX bank promoted to independent 400-model submission

- Promoted the best-known validated 400 ONNX models into `outputs/onnx`.
- Added `src.build_current_model_submission` so `outputs/submission.zip` can be rebuilt from `outputs/onnx` without reading `archive`.
- Added focused tests for local model-bank packaging and incomplete-bank rejection.
- Full local model-bank validation selected 400 / 400 tasks with 0 missing or invalid tasks.
- Current estimated cost total: 10530917.
- Current ONNX file size total: 14815565 bytes.
- `outputs/submission.zip` inspection passed with 400 ONNX entries.
- Added `EXTERNAL_OPTIMIZATION_CONTEXT.md` for external optimization review.
- Final verification passed: full pytest reported 64 passed and 2 skipped.
- `python -m compileall src tests` passed.
- `git diff --check` passed with only LF-to-CRLF warnings for Markdown logs.
- Removed `archive`, stale archive/blend reports, duplicate verified model directories, temporary validation output, and cache directories.

Validation commands used:

```powershell
python -m src.build_current_model_submission --data-dir task --model-dir outputs\onnx --validated-dir outputs\current_model_bank_verified_onnx --report outputs\reports\current_model_bank_report.csv --zip outputs\submission.zip --timeout-seconds 120
python -m src.inspect_submission --zip outputs\submission.zip
python -m pytest -q tests\test_build_current_model_submission.py
python -m pytest -q
python -m compileall src tests
git diff --check
```

Notes:

- `archive` is baseline history only and is no longer required to rebuild the current submission.
- Local train validation is not a guaranteed official leaderboard score.

## 2026-06-03 - High-cost replacement search workflow

- Added `src.diagnose_high_cost_tasks`.
  - Reads `outputs/reports/current_model_bank_report.csv` and `task/*.json`.
  - Writes `outputs/reports/high_cost_task_diagnosis.csv`.
  - Writes per-task Markdown analyses under `outputs/reports/high_cost_task_analysis/`.
- Added `src.search_symbolic_replacements`.
  - Runs formal `first_version_rules()` against selected high-cost tasks.
  - Builds candidate ONNX only for conservative MATCH rules with available builders.
  - Validates candidates through isolated `src.evaluate_onnx_candidate`.
  - Marks `replace_recommended=True` only when validation passes and candidate cost is lower than current cost.
  - Supports `--replace`, but no replacement is copied unless the strict lower-cost condition holds.
- Added focused tests in `tests/test_high_cost_replacement_search.py`.

First-round target tasks:

`task133`, `task084`, `task209`, `task076`, `task157`, `task200`, `task233`.

Results:

- Diagnosed tasks: 7.
- Replacement search rows: 217.
- Formal rules searched per task: 31.
- Replacement count: 0.
- `outputs/onnx` model contents were not changed by the search because every top-7 task was rejected at matcher stage by all current formal rules.
- Current estimated cost total after rebuild: 10530917.
- Current ONNX file size total after rebuild: 14815565 bytes.
- `outputs/submission.zip` size: 1466160 bytes.
- Full local model-bank rebuild selected 400 / 400 tasks with 0 missing or invalid tasks.

Validation commands used:

```powershell
python -m pytest -q tests\test_high_cost_replacement_search.py
python -m compileall src tests
python -m src.diagnose_high_cost_tasks --data-dir task --current-report outputs\reports\current_model_bank_report.csv --report outputs\reports\high_cost_task_diagnosis.csv --analysis-dir outputs\reports\high_cost_task_analysis --top-k 7 --task-ids task133,task084,task209,task076,task157,task200,task233
python -m src.search_symbolic_replacements --data-dir task --current-model-dir outputs\onnx --current-report outputs\reports\current_model_bank_report.csv --candidate-dir outputs\candidates\replacements --report outputs\reports\replacement_search_report.csv --top-k 7 --task-ids task133,task084,task209,task076,task157,task200,task233 --replace --timeout-seconds 120
python -m src.build_current_model_submission --data-dir task --model-dir outputs\onnx --validated-dir outputs\current_model_bank_verified_onnx --report outputs\reports\current_model_bank_report.csv --zip outputs\submission.zip --timeout-seconds 120
python -m src.inspect_submission --zip outputs\submission.zip
python -m pytest -q
git diff --check
```

Validation results:

- Focused replacement tests: 2 passed.
- `compileall`: passed.
- `inspect_submission`: passed, 400 ONNX models.
- Full pytest: 66 passed, 2 skipped.
- `git diff --check`: passed.

## 2026-06-03 - task084 diagonal-bottom fill cost optimization

- Added `DiagonalBottomFillRule` and `build_dynamic_left_column_diagonal_bottom_fill_model`.
  - Matcher is conservative: square same-size grids, solid nonzero left column, all other input cells zero, exact anti-diagonal color 2 and bottom-row color 4 in every train case.
  - Builder infers the active square from one-hot padding, preserves the input left/background cells, paints only the dynamic anti-diagonal and bottom row, and zeros padding output.
- Replaced `outputs/onnx/task084.onnx` only after strict validation and lower-cost check.
  - Old cost: 1390970.
  - New cost: 722.
  - Cost delta: 1390248.
  - Old file size: 1127799 bytes.
  - New file size: 4301 bytes.
- Rebuilt current model bank and `outputs/submission.zip`.
  - selected tasks: 400 / 400.
  - missing or invalid tasks: 0.
  - estimated cost total: 9140669.
  - ONNX file size total: 13692067 bytes.
  - `outputs/submission.zip` inspection passed with 400 ONNX entries.

Validation commands used:

```powershell
python -m pytest -q tests\test_pattern_rules.py
python -m src.search_symbolic_replacements --data-dir task --current-model-dir outputs\onnx --current-report outputs\reports\current_model_bank_report.csv --candidate-dir outputs\candidates\replacements --report outputs\reports\replacement_search_report.csv --task-ids task084 --replace --timeout-seconds 120
python -m src.build_current_model_submission --data-dir task --model-dir outputs\onnx --validated-dir outputs\current_model_bank_verified_onnx --report outputs\reports\current_model_bank_report.csv --zip outputs\submission.zip --timeout-seconds 120
python -m src.inspect_submission --zip outputs\submission.zip
python -m pytest -q tests\test_pattern_rules.py tests\test_high_cost_replacement_search.py tests\test_build_current_model_submission.py
python -m compileall src tests
python -m pytest -q
git diff --check
```

Validation results:

- `tests\test_pattern_rules.py`: 54 passed.
- Focused combined tests: 59 passed.
- Full pytest: 68 passed, 2 skipped.
- `compileall`: passed.
- `inspect_submission`: passed, 400 ONNX models.
- `git diff --check`: no whitespace errors; only LF-to-CRLF warnings.

Note: all scores and costs above are local estimated values from strict train validation, not guaranteed official leaderboard scores.

## 2026-06-03 - task200 bottom-marker vertical stripe cost optimization

- Added `BottomMarkerVerticalStripeRule` and `build_dynamic_bottom_marker_vertical_stripes_model`.
  - Matcher is conservative: same-size grids with exactly one nonzero marker on the bottom row, and every train output must equal the marker-color vertical stripe pattern with color-5 top/bottom connectors.
  - Builder infers active size from one-hot padding, infers the marker column and marker color from the input, preserves active background cells, and zeros padding output.
- Replaced `outputs/onnx/task200.onnx` only after strict validation and lower-cost check.
  - Old cost: 990050.
  - New cost: 992.
  - Cost delta: 989058.
  - Old file size: 797905 bytes.
  - New file size: 14264 bytes.
- Rebuilt current model bank and `outputs/submission.zip`.
  - selected tasks: 400 / 400.
  - missing or invalid tasks: 0.
  - estimated cost total: 8151611.
  - ONNX file size total: 12908426 bytes.
  - `outputs/submission.zip` inspection passed with 400 ONNX entries.

Validation commands used:

```powershell
python -m pytest -q tests\test_pattern_rules.py
python -m src.search_symbolic_replacements --data-dir task --current-model-dir outputs\onnx --current-report outputs\reports\current_model_bank_report.csv --candidate-dir outputs\candidates\replacements --report outputs\reports\replacement_search_report_task200.csv --task-ids task200 --replace --timeout-seconds 120
python -m src.build_current_model_submission --data-dir task --model-dir outputs\onnx --validated-dir outputs\current_model_bank_verified_onnx --report outputs\reports\current_model_bank_report.csv --zip outputs\submission.zip --timeout-seconds 120
python -m src.inspect_submission --zip outputs\submission.zip
python -m pytest -q tests\test_pattern_rules.py tests\test_high_cost_replacement_search.py tests\test_build_current_model_submission.py
python -m compileall src tests
python -m pytest -q
git diff --check
```

Validation results:

- `tests\test_pattern_rules.py`: 56 passed.
- Focused combined tests: 61 passed.
- Full pytest: 70 passed, 2 skipped.
- `compileall`: passed.
- `inspect_submission`: passed, 400 ONNX models.
- `git diff --check`: no whitespace errors; only LF-to-CRLF warnings.

Note: `task157`, `task025`, `task367`, and `task363` were inspected as possible follow-up targets, but their rules were not clear enough for a safe quick MATCH in this round.

## 2026-06-03 - external optimization context handoff

- Recreated `EXTERNAL_OPTIMIZATION_CONTEXT.md` for external review.
- The handoff summarizes:
  - project constraints and validation gates;
  - current canonical model bank and submission status;
  - recent `task084` and `task200` cost reductions;
  - current top-25 remaining high-cost tasks;
  - useful scripts and reproduction commands;
  - recommended response format for external optimization suggestions.
- No ONNX model or code behavior changed in this documentation-only step.

## 风险提示

- 剩余 probe-only 规则仍不能加入 `first_version_rules()`。
- 本地 train 验证不是官方榜单分数。
- 动态 bbox / mirror 新 builder 使用 ArgMax、ArgMin、Gather、Where、ReduceSum 等允许算子；当前已通过本地 ONNX checker、onnxruntime 和 submission inspection。
## 2026-06-03 - task396 largest-frame recolor crop optimization

- Added `DynamicLargestFrameRecolorCropRule` and
  `build_dynamic_largest_frame_recolor_crop_model`.
  - Matcher is conservative: exactly two nonzero colors, unique most-frequent
    source/frame color, unique least-frequent marker color, unique largest
    source-color rectangular frame with dimensions 4..8, and exact output equal
    to the selected frame crop with every nonzero cell recolored to the marker
    color.
  - Builder uses static-shape ONNX only. It dynamically selects source and
    marker colors by channel counts, enumerates 4..8 frame kernels, selects the
    largest valid frame, crops it with `Gather`, preserves real zero cells as
    color 0, recolors nonzero cells to the marker channel, and zeros padding.
- Replaced `outputs/onnx/task396.onnx` after strict validation and lower-cost
  check.
  - Old cost: 115080.
  - New cost: 6009.
  - Cost delta: 109071.
  - Old file size: 148123 bytes.
  - New file size: 50393 bytes.
- Extra confidence check for `task396`:
  - train: 3 / 3 passed.
  - test: 1 / 1 passed.
  - arc-gen: 262 / 262 passed.
- Rebuilt current model bank and `outputs/submission.zip`.
  - selected tasks: 400 / 400.
  - missing or invalid tasks: 0.
  - estimated cost total: 8042540.
  - ONNX file size total: 12810696 bytes.
  - `outputs/submission.zip` inspection passed with 400 ONNX entries.

Validation commands used:

```powershell
python -m pytest -q tests\test_pattern_rules.py -k largest_frame_recolor
python -m src.search_symbolic_replacements --data-dir task --current-model-dir outputs\onnx --current-report outputs\reports\current_model_bank_report.csv --candidate-dir outputs\candidates\replacements --report outputs\reports\replacement_search_report_task396.csv --task-ids task396 --replace --timeout-seconds 120
python -c "import json; from src.validate_onnx_model import validate_cases; d=json.load(open('task/task396.json', encoding='utf-8')); model='outputs/onnx/task396.onnx'; for split in ['train','test','arc-gen']: r=validate_cases(model, d.get(split, [])); print(split, r['passed'], r['num_cases'], r['num_failed_cases'])"
python -m src.build_current_model_submission --data-dir task --model-dir outputs\onnx --validated-dir outputs\current_model_bank_verified_onnx --report outputs\reports\current_model_bank_report.csv --zip outputs\submission.zip --timeout-seconds 120
python -m src.inspect_submission --zip outputs\submission.zip
python -m pytest -q tests\test_pattern_rules.py
python -m pytest -q tests\test_high_cost_replacement_search.py
python -m compileall src tests
python -m pytest -q
git diff --check
```

Validation results:

- Focused largest-frame test: 1 passed.
- `tests\test_pattern_rules.py`: 57 passed.
- `tests\test_high_cost_replacement_search.py`: 2 passed.
- Full pytest: 71 passed, 2 skipped.
- `compileall`: passed.
- `inspect_submission`: passed, 400 ONNX models.
- `git diff --check`: no whitespace errors; only LF-to-CRLF warnings.

Note: `task133`, `task076`, and `task157` were reviewed as first-priority
same-shape mask targets, but no safe MATCH rule was promoted in this round.
The `task396` result is a local strict-validation replacement and an estimated
cost improvement, not a guaranteed official leaderboard score.
