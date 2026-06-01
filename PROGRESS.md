# 当前进度

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
