# 当前进度

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

## 风险提示

- 剩余 probe-only 规则仍不能加入 `first_version_rules()`。
- 本地 train 验证不是官方榜单分数。
- 动态 bbox / mirror 新 builder 使用 ArgMax、ArgMin、Gather、Where、ReduceSum 等允许算子；当前已通过本地 ONNX checker、onnxruntime 和 submission inspection。
