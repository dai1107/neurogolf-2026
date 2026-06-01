# 实验日志

## 2026-05-31 21:55 - Rectangle/line builders + shape-polymorphic single-color translation

### 目标

继续完成 `模型分析及优化策略.md` 中未完成的优化策略，重点处理：

- `RectangleAndLineRule` 的 probe-only builder 补齐。
- `ShapePolymorphicTranslationRule` 中 single-color translation 仍依赖共享 shape 的缺口。

### 修改文件

- `src/onnx_builders.py`
- `src/pattern_rules.py`
- `tests/test_pattern_rules.py`
- 重新生成 `outputs/reports/probe_summary.csv`
- 重新生成 `outputs/reports/summary.csv`
- 重新生成 `outputs/reports/failure_taxonomy.csv`
- 重新生成 `outputs/reports/rule_near_miss.csv`
- 重新生成 `outputs/onnx/*.onnx`
- 重新生成 `outputs/candidates/*.onnx`
- 重新生成 `outputs/logs/*.json`
- 重新生成 `outputs/submission.zip`

### 实现内容

- `RectangleAndLineRule`
  - 新增 `build_static_overlay_model()`。
  - 新增 `build_line_extension_model()`。
  - `bbox_fill` / `bbox_frame` 在所有 train case 共享同一绘制 mask 时可构建静态 overlay ONNX。
  - `connect_two_points_horizontal` / `connect_two_points_vertical` 在共享绘制 mask 时可构建静态 overlay ONNX。
  - `extend_line` 的 horizontal / vertical 模式使用动态 ONNX：从输入中找含目标颜色的 active 行/列，并在 active 区域内延展。
  - diagonal connect/extend 仍保持 probe-only，不进入提交候选。
- `MultiStepTranslationRule`
  - single-color translation 不再要求所有 train case 共享同一 shape。
  - 新增 `build_dynamic_single_color_translation_model()`，用 padding active mask 推断真实区域，避免改写 padding。
  - matcher 从首个样例的目标色位置推导候选 dy/dx，再严格验证所有 train case，避免暴力枚举导致 probe 超时。
  - dy/dx 搜索范围扩展到 `[-15, 15]`。
- 新增/更新测试：
  - 静态 bbox fill builder。
  - 静态水平 connect builder。
  - 动态水平 extend builder。
  - 可变 shape single-color translation builder。

### 结果

- Local train solved: 45 / 400
- failed: 355 / 400
- 本轮没有新增 best solved task。
- `MultiStepTranslationRule` probe: 3 / 400，命中 `task073 task276 task309`。
  - `task276` 和 `task309` 已由 `ColorMapRule` 以 cost 500 解决，因此不会被更高成本 translation 覆盖。
- `RectangleAndLineRule` probe: 1 / 400，仍只命中 `task171`。
- solved 模型 estimated cost 总和: 176731
- `outputs/submission.zip`: 27700 bytes
- `rule_near_miss.csv`: 355 rows
- near-miss 分布：
  - panel_rule_near_miss: 167
  - blocked_by_shared_shape: 127
  - blocked_by_shared_output_shape: 61

### 验证命令

已运行：

```powershell
python -m pytest tests\test_pattern_rules.py -q
python -m src.probe_rules --data-dir task --report outputs\reports\probe_summary.csv --all-tasks
python -B -m pytest -q
python -m compileall src tests
python -m src.build_submission
python -m src.inspect_submission --zip outputs\submission.zip
python -m src.failure_taxonomy
```

验证结果：

- targeted rule tests: 38 passed
- full pytest: 47 passed
- full rebuild: 400 tasks processed, 45 solved, 355 failed
- submission inspection: passed, 45 ONNX models
- failure taxonomy rows: 355
- rule_near_miss rows: 355

### 当前判断

`模型分析及优化策略.md` 中明确列出的本轮工程项已经完成可安全提交版本：Local shape-polymorphic、HoleFill、AutoPeriod 单轴、Translation shape-polymorphic、rule_near_miss、Rectangle/Line builder、PanelSelect 保守 probe/builder。未达到 50 solved 的主要原因不再是这些 builder 缺失，而是剩余任务需要新的 panel/object/crop 语义规则。

## 2026-05-31 21:20 - Auto-period periodic extension

### 目标

继续按 `模型分析及优化策略.md` 和 `PROGRESS.md` 推进，优先实现 `AutoPeriodExtensionRule` 方向，目标覆盖 `task003` 这类不同 train case 最小周期不同的周期延展任务。

### 修改文件

- `src/onnx_builders.py`
- `src/pattern_rules.py`
- `tests/test_pattern_rules.py`
- 重新生成 `outputs/reports/probe_summary.csv`
- 重新生成 `outputs/reports/summary.csv`
- 重新生成 `outputs/reports/failure_taxonomy.csv`
- 重新生成 `outputs/reports/rule_near_miss.csv`
- 重新生成 `outputs/onnx/*.onnx`
- 重新生成 `outputs/candidates/*.onnx`
- 重新生成 `outputs/logs/*.json`
- 重新生成 `outputs/submission.zip`

### 实现内容

- 扩展 `PeriodicExtensionColorMapRule`：
  - 保留原固定 `period_y/period_x` 路径。
  - 新增单轴 auto-period matcher，当前支持 row/col 单轴扩展。
  - 每个 train case 单独从 input 推断最小周期；所有 case 共享同一种“自动推断周期并延展”的语义。
  - 要求共享 input/output shape，且仅处理单轴变大，保持保守。
- 新增 `build_auto_periodic_extension_color_map_model()`：
  - ONNX 图中并行构造所有候选周期 remap。
  - 用 `Gather`/`Sub`/`Abs`/`ReduceSum`/`Less` 判定每个候选周期是否解释 input。
  - 用最小有效周期的 selector 选择候选输出。
  - 使用静态 shape，不使用 Loop/Scan/NonZero/Unique。
  - 输出 active 区域外保持全零。
- 新增测试覆盖不同 train case 周期分别为 4、2、3 的 row periodic extension，并验证 ONNX runtime 精确输出。

### 结果

- Local train solved: 45 / 400
- failed: 355 / 400
- 本轮新增 solved：
  - `task003` -> `PeriodicExtensionColorMapRule`, cost 49335, file 45604
- solved 模型 estimated cost 总和: 176731
- `outputs/submission.zip`: 27700 bytes
- `probe_summary.csv`: `PeriodicExtensionColorMapRule` 1 / 400，命中 `task003`
- `rule_near_miss.csv`: 355 rows

### 验证命令

已运行：

```powershell
python -m src.probe_rules --data-dir task --report outputs\reports\probe_summary.csv --all-tasks
python -B -m pytest -q
python -m compileall src tests
python -m src.build_submission
python -m src.inspect_submission --zip outputs\submission.zip
python -m src.failure_taxonomy
```

验证结果：

- `pytest`: 44 passed
- full rebuild: 400 tasks processed, 45 solved, 355 failed
- submission inspection: passed, 45 ONNX models
- failure taxonomy rows: 355
- rule_near_miss rows: 355

### 下一步建议

`task003` 的新模型正确但成本偏高，原因是 auto-period builder 同时保留多个候选周期图。继续冲 solved 数时，优先级仍应高于 cost 微调：建议下一轮先补 `RectangleAndLineRule` 的 bbox_fill / bbox_frame / connect / extend builder；如果要降 cost，可专门压缩 auto-period selector 图。

## 2026-05-30 23:55 - HoleFillRule + dynamic translation probe support

### 目标

继续按 `模型分析及优化策略.md` 推进八个方向中的剩余项。本轮完成：

- 新增 `HoleFillRule`
- 为 `MultiStepTranslationRule` 增加 shape-polymorphic 整图平移支持
- 新增保守 `PanelSelectByColorRule` probe/builder

没有完成全部八个方向；当前仍未达到 50 solved。

### 修改文件

- `src/onnx_builders.py`
- `src/pattern_rules.py`
- `tests/test_pattern_rules.py`
- 重新生成 `outputs/reports/probe_summary.csv`
- 重新生成 `outputs/reports/summary.csv`
- 重新生成 `outputs/reports/failure_taxonomy.csv`
- 重新生成 `outputs/reports/rule_near_miss.csv`
- 重新生成 `outputs/onnx/*.onnx`
- 重新生成 `outputs/logs/*.json`
- 重新生成 `outputs/submission.zip`

### 实现内容

- `HoleFillRule`
  - Python matcher 枚举 `background_color` 和 `fill_color`。
  - 找 background connected components；不接触真实 grid 边界的 background component 视为 hole。
  - ONNX builder 使用固定 30 次 4-neighbor Conv 膨胀做 flood fill，不使用 Loop/Scan/NonZero/Unique。
  - 动态 active mask 用于识别真实 grid 边界，避免把 padding 当背景。
- `MultiStepTranslationRule`
  - 新增 `build_dynamic_fill_translation_model()`。
  - 对不同 train case shape 的 same-size 整图平移，用 input one-hot 动态计算 active cells。
  - padding 区域保持全零，不被填成真实颜色 0。
  - 当前真实 probe 没新增命中，但测试覆盖保留。
- `PanelSelectByColorRule`
  - 保守实现：要求共享 output shape、共享 panel layout、同一 panel index，并且有 selector 证据。
  - selector 当前支持 contains unique color / most non-background / least non-background。
  - 当前真实 probe 0 / 400，未新增 solved。

### 结果

- Local train solved: 44 / 400
- failed: 356 / 400
- 本轮相对 42 solved 新增：
  - `task002` -> HoleFillRule, cost 205, file 12507
  - `task251` -> HoleFillRule, cost 205, file 12507
- 相对 38 solved 的本轮累计新增：
  - `task002`, `task147`, `task251`, `task258`, `task272`, `task352`
- solved 模型 estimated cost 总和: 127396
- `outputs/submission.zip`: 25818 bytes

### 验证命令

已运行：

```powershell
python -m src.probe_rules --data-dir task --report outputs\reports\probe_summary.csv --all-tasks
python -B -m pytest -q
python -m compileall src tests
python -m src.build_submission
python -m src.inspect_submission --zip outputs\submission.zip
python -m src.failure_taxonomy
```

验证结果：

- `pytest`: 43 passed
- full rebuild: 400 tasks processed, 44 solved, 356 failed
- submission inspection: passed, 44 ONNX models
- failure taxonomy rows: 356
- rule_near_miss rows: 356

### 当前未完成项

`模型分析及优化策略.md` 中八个方向没有全部完成。当前完成/部分完成状态：

- shared-shape 过度限制：部分完成，LocalFill/Rewrite 完成；Translation 做了动态整图平移但真实无新增。
- ShapePolymorphicLocalRule：完成，带来 `task147`, `task258`, `task272`, `task352`。
- HoleFillRule：完成，带来 `task002`, `task251`。
- AutoPeriodExtensionRule：未完成。
- ShapePolymorphicTranslationRule：部分完成，测试通过但真实 probe 无新增。
- rule_near_miss.csv：完成。
- RectangleAndLineRule builder 补齐：未完成。
- PanelSelectByColorRule：保守版完成，但真实 probe 0 / 400，需扩展 layout/selector。

### 下一步建议

继续冲 50 solved 时，优先做：

1. `AutoPeriodExtensionRule` Python probe，先确认是否能覆盖 `task003` 及类似任务。
2. `RectangleAndLineRule` 的 bbox_fill / bbox_frame / connect/extend builder 补齐。
3. 扩展 panel 选择，不只限于现有 `_enumerate_panel_layouts_for_grid()` 支持的等宽/等高 panel。

## 2026-05-30 23:30 - Shape-polymorphic local rules

### 目标

根据 `模型分析及优化策略.md`，优先解除局部规则的 shared-shape 过度限制，并生成 `rule_near_miss.csv` 用于下一轮决策。

### 修改文件

- `src/pattern_rules.py`
- `src/onnx_builders.py`
- `src/failure_taxonomy.py`
- `tests/test_pattern_rules.py`
- 重新生成 `outputs/reports/probe_summary.csv`
- 重新生成 `outputs/reports/summary.csv`
- 重新生成 `outputs/reports/failure_taxonomy.csv`
- 新增/重新生成 `outputs/reports/rule_near_miss.csv`
- 重新生成 `outputs/onnx/*.onnx`
- 重新生成 `outputs/logs/*.json`
- 重新生成 `outputs/submission.zip`

### 实现内容

- `LocalNeighborhoodFillRule` 不再要求所有 train case 共享同一 grid shape。
  - 现在只要求每个 case 内部 `input_shape == output_shape`。
  - ONNX builder 原本已经在 padding 全零语义下安全，不会改写 padding。
- `LocalNeighborhoodRewriteRule` 不再要求所有 train case 共享同一 grid shape。
  - 移除了固定 `active_height/active_width` metadata。
  - builder 去掉静态 `ActiveMask`，因为 padding 全 channel 为 0，`target_mask` 为 0，不会触发 rewrite。
- `build_local_neighborhood_fill_model()` 去掉 `ge` 条件下未使用的 `Threshold` initializer，消除 ORT unused initializer 警告并略降 cost。
- `src.failure_taxonomy` 现在同时生成 `outputs/reports/rule_near_miss.csv`。

### 结果

- Local train solved: 42 / 400
- failed: 358 / 400
- 本轮新增 solved:
  - `task147` -> LocalNeighborhoodFillRule, cost 560, file 1314
  - `task258` -> LocalNeighborhoodFillRule, cost 565, file 1476
  - `task272` -> LocalNeighborhoodRewriteRule, cost 560, file 1383
  - `task352` -> LocalNeighborhoodFillRule, cost 565, file 1476
- solved 模型 estimated cost 总和: 126986
- `outputs/submission.zip`: 22262 bytes

### 验证命令

已运行：

```powershell
python -m src.probe_rules --data-dir task --report outputs\reports\probe_summary.csv --all-tasks
python -B -m pytest -q
python -m compileall src tests
python -m src.build_submission
python -m src.inspect_submission --zip outputs\submission.zip
python -m src.failure_taxonomy
```

验证结果：

- `pytest`: 40 passed
- probe:
  - LocalNeighborhoodFillRule: 8 / 400 matched
  - LocalNeighborhoodRewriteRule: 9 / 400 matched
- full rebuild: 400 tasks processed, 42 solved, 358 failed
- submission inspection: passed, 42 ONNX models
- failure taxonomy rows: 358
- rule_near_miss rows: 358

### 当前 rule_near_miss 分布

- panel_rule_near_miss: 168
- blocked_by_shared_shape: 129
- blocked_by_shared_output_shape: 61

### 下一步备注

当前最大 near-miss 类别已经变成 panel 方向。下一轮优先考虑 `PanelSelectByColorRule` 或扩展 `GeneralizedPanelRule`，目标不再只是二元 AND/OR/XOR，而是：

- 输出某个 panel
- 输出含目标颜色的 panel
- 输出去掉 separator 后的某个区域
- 输出多个 panel 中唯一颜色不同的那个
- 输出多个 panel 的局部组合

同时仍有 129 个 `blocked_by_shared_shape`，可继续按策略改造 translation / symmetry / rectangle-line 等规则。

## 2026-05-30 22:40 - ObjectSelection 保守工程化 + ColorMap Gather 优化

### 目标

按 `PROGRESS.md` 的下一步继续优化，但保持保守：先把 `ObjectSelectionRule` 从 probe-only 改成只有静态 bbox 才能生成 ONNX 的正式规则；随后对已 solved 的颜色置换模型做 cost 优化。

### 修改文件

- `src/pattern_rules.py`
- `src/onnx_builders.py`
- `tests/test_pattern_rules.py`
- 重新生成 `outputs/onnx/*.onnx`
- 重新生成 `outputs/candidates/*.onnx`
- 重新生成 `outputs/logs/*.json`
- 重新生成 `outputs/reports/summary.csv`
- 重新生成 `outputs/reports/probe_summary.csv`
- 重新生成 `outputs/reports/failure_taxonomy.csv`
- 重新生成 `outputs/submission.zip`

### 实现内容

- `ObjectSelectionRule` 新增保守 ONNX builder：
  - 新增 `_selected_object_for_case()`，保留选中对象的 bbox 和 crop。
  - 只有所有 train case 选出的 `(top, left, height, width)` 完全一致时才返回 MATCH。
  - build 阶段复用 `build_spatial_remap_model()` 做静态 crop + optional color map。
  - 已加入 `first_version_rules()`。
  - 当前 probe 结果只命中 `task016`, `task267`，都是已 solved，因此没有新增 solved 数。
- `build_color_map_model()` 新增置换优化：
  - 如果完整 color map 是双射，使用 `Gather(axis=1)` 做 channel permutation。
  - 如果不是双射，保持原 1x1 Conv，不改变语义。

### 结果

- Local train solved: 38 / 400
- failed: 362 / 400
- solved 模型 estimated cost 总和: 124736
- `outputs/submission.zip`: 19905 bytes
- `outputs/reports/summary.csv`: 855095 bytes
- `outputs/reports/failure_taxonomy.csv`: 39852 bytes

ColorMap 优化结果：

- `task016` ColorMapRule: cost 500 -> 50, file 599 -> 238
- `task337` ColorMapRule: cost 500 -> 50, file 599 -> 238
- `task267`, `task276`, `task309` 是非双射或存在 channel collision，继续使用 Conv，cost 保持 500。

### 验证命令

已运行：

```powershell
python -m src.probe_rules --data-dir task --report outputs\reports\probe_summary.csv --all-tasks
python -m pytest
python -m src.build_submission
python -m src.inspect_submission --zip outputs\submission.zip
python -m src.failure_taxonomy
python -m compileall src tests
```

验证结果：

- `probe_rules`: ObjectSelectionRule 2 / 400 matched: `task016 task267`
- `pytest`: 38 passed
- full rebuild: 400 tasks processed, 38 solved, 362 failed
- submission inspection: passed, 38 ONNX models
- failure taxonomy rows: 362

### 下一步备注

保守静态 ObjectSelection 不足以解决 `task031`, `task036`, `task259`, `task300`，因为这些任务需要动态对象选择或动态 top-left normalization。下一步如果继续攻这些任务，应先做新的 probe，明确是否能用合法 ONNX 静态图实现动态 bbox/translation，避免直接写高成本或不合规模型。

## 2026-05-30 22:10 - Bool mask cost 优化

### 目标

在不扩大规则匹配面、不引入未验证模型的前提下，降低当前已解决 ONNX 模型的 estimated cost 和文件体积。

正确性仍然是硬门槛：最终模型必须通过 ONNX checker、禁用算子检查、静态 shape 检查、严格 train 验证、文件大小检查和 cost 统计。

### 当前背景

- 本地 train solved: 38 / 400
- failed: 362 / 400
- `outputs/submission.zip` 已生成，并通过 `inspect_submission`
- 当前仓库没有 `README.md`。新线程接手时优先读 `AGENTS.md`、`PROGRESS.md`、本文档、`outputs/reports/summary.csv` 和核心源码。
- `git status --short -- .` 当前显示 `?? ./`，说明这个目录整体在可见 Git 基线里是未跟踪状态。不要依赖 tracked diff 来恢复进度。

### 修改文件

- `src/onnx_builders.py`
- 重新生成 `outputs/onnx/*.onnx`
- 重新生成 `outputs/candidates/*.onnx`
- 重新生成 `outputs/logs/*.json`
- 重新生成 `outputs/reports/summary.csv`
- 重新生成 `outputs/reports/failure_taxonomy.csv`
- 重新生成 `outputs/submission.zip`

### 实现内容

- 新增 `_bool_mask()` 和 `_cast_to_float()` helper。
- 将只表示 0/1 的大尺寸 mask initializer 从 `float32` 改成 `bool`。
- 在 ONNX 图内用显式 `Cast(..., to=FLOAT)` 转回 float，再参与 `Mul` / `Sub`。
- 对全 1 的 `ActiveMask` 直接省略；不需要 mask 时输出改成 `Identity`。
- 覆盖的 builder：
  - spatial remap
  - one-step translation
  - zero-fill translation
  - single-color translation
  - panel binary op
  - generalized panel op
  - periodic extension color map
  - symmetry completion
  - self-kron mask
  - local neighborhood rewrite
  - rotate

### 结果

- Local train solved: 38 / 400
- solved 模型 estimated cost 总和: 125636
- `outputs/submission.zip`: 19920 bytes
- `outputs/reports/summary.csv`: 816091 bytes
- `outputs/reports/failure_taxonomy.csv`: 39852 bytes

全量重建后的主要 cost 改善：

- `task073` MultiStepTranslationRule: 13718 -> 8318
- `task053` OneStepTranslationRule: 9350 -> 3950
- 通用 spatial remap 类任务：常见 9063 -> 4563
- panel 类任务：约 5250 -> 2550 / 2555
- `task001` SelfKronMaskRule: 5200 -> 2500
- `task287` / `task385` SymmetryCompletionRule: 4950 -> 2250

### 验证命令

已运行：

```powershell
python -m pytest
python -m src.build_submission
python -m src.inspect_submission --zip outputs\submission.zip
python -m src.failure_taxonomy
python -m compileall src tests
```

验证结果：

- `pytest`: 37 passed
- 全量 rebuild: 400 tasks processed, 38 solved, 362 failed
- submission inspection: passed, 38 ONNX models
- failure taxonomy rows: 362
- compileall: passed

### solved 规则分布

- PanelSeparatorBinaryOpRule: 7
- MirrorConcatRule: 5
- ColorMapRule: 5
- SubstructureExtractRule: 4
- RotateRule: 3
- LocalNeighborhoodFillRule: 3
- CropRule: 2
- ScaleRepeatRule: 2
- SymmetryCompletionRule: 2
- OneStepTranslationRule: 1
- SelfKronMaskRule: 1
- RectangleAndLineRule: 1
- StridedSubsampleRule: 1
- MultiStepTranslationRule: 1

### 当前 solved 任务列表

`task001`, `task006`, `task016`, `task026`, `task048`, `task053`, `task072`, `task073`, `task081`, `task087`, `task095`, `task116`, `task130`, `task135`, `task140`, `task144`, `task164`, `task171`, `task172`, `task210`, `task223`, `task236`, `task267`, `task276`, `task287`, `task291`, `task294`, `task307`, `task309`, `task311`, `task318`, `task326`, `task337`, `task346`, `task355`, `task380`, `task385`, `task386`.

### 当前仍然 cost 较高的 solved 模型

- `task073` MultiStepTranslationRule: cost 8318, file 6966
- `task048` / `task291` / `task346` / `task355` SubstructureExtractRule: cost 6863, file 5569
- `task326` CropRule: cost 6363, file 5058
- 多个 spatial remap 模型：cost 4563, file 4053

不要在没有严格验证的情况下继续压这些模型。进一步压缩 translation 可能需要新的 ONNX 结构，例如 `Pad` 或更小中间张量，应作为单独实验，并先确认算子合法性和静态 shape。

### 失败任务分类

来自 `outputs/reports/failure_taxonomy.csv`：

- variable_shapes: 194
- same_size: 122
- shrinks_or_crop: 34
- integer_scale_or_tile: 11
- expands_non_integer: 1

### probe 备注

当前 `outputs/reports/probe_summary.csv` 不是 bool mask 优化后重新生成的，但仍可作为方向参考：

- ObjectSelectionRule 命中 9 个任务：`task016 task031 task036 task259 task267 task276 task300 task309 task337`
- 其中已 solved：`task016`, `task267`, `task276`, `task309`, `task337`
- 如果实现保守 ONNX builder，潜在未 solved 目标：`task031`, `task036`, `task259`, `task300`

### 下一步假设

1. 优先实现保守版 `ObjectSelectionRule` ONNX builder。只允许 selected bbox 坐标和输出 shape 在 train 样例中静态一致的情况进入 MATCH。
2. 在继续 same-size 局部规则前，优先扩展 shrink/crop/object normalization 类规则。
3. 继续使用 probe-first 流程：先写 Python matcher，跑 `probe_rules`，确认 MATCH 数量后再写 ONNX builder。
4. 新候选必须通过所有验证和约束检查后，才能进入 `submission.zip`。
## 2026-05-31 23:10 - 动态 bbox / active mirror / color-role swap 正式 builder

### 目标

继续根据 `candidate_discovery_report.csv` 中的未解决任务，把已经能被 Python probe 严格解释的策略转成安全 ONNX builder，扩大模型适用范围。

### 修改文件

- `src/onnx_builders.py`
- `src/pattern_rules.py`
- `tests/test_pattern_rules.py`
- 重新生成 `outputs/onnx/*.onnx`
- 重新生成 `outputs/candidates/*.onnx`
- 重新生成 `outputs/logs/*.json`
- 重新生成 `outputs/reports/summary.csv`
- 重新生成 `outputs/reports/failure_taxonomy.csv`
- 重新生成 `outputs/reports/rule_near_miss.csv`
- 重新生成 `outputs/reports/candidate_discovery_report.csv`
- 重新生成 `outputs/reports/probe_summary.csv`
- 重新生成 `outputs/submission.zip`
- 更新 `PROGRESS.md`
- 更新 `EXPERIMENT_LOG.md`

### 实现内容

- 新增 `build_dynamic_non_background_bbox_crop_model()`：
  - 动态推断所有非背景像素 bbox。
  - 使用 `ArgMax`/反向 `ArgMax` 找 top/left/bottom/right。
  - 使用动态 `Gather` 将 bbox crop 放到输出左上角。
  - 支持 `identity`, `mirror_horizontal`, `mirror_vertical`。
  - 支持 color map。
- 新增正式规则 `DynamicNonBackgroundBBoxCropRule`：
  - 解决 `task031`, `task259`。
  - 扩展 mirror 后解决 `task177`。
- 新增 `build_dynamic_active_mirror_model()`：
  - 动态推断 top-left active rectangle 的高宽。
  - 支持不同 train shape 的 horizontal/vertical mirror。
- 新增正式规则 `DynamicActiveMirrorRule`：
  - 解决 `task150`, `task155`。
- 新增 `build_dynamic_bbox_extreme_color_swap_model()`：
  - bbox crop 后统计 bbox 内每个颜色的频次。
  - 将最多颜色与最少非零颜色互换。
  - 不硬编码颜色值。
- 新增正式规则 `DynamicBBoxExtremeColorSwapRule`：
  - 解决 `task290`。
- 新增相关合成测试，覆盖动态 bbox crop、bbox+mirror、shape-polymorphic mirror、bbox color-role swap。

### 验证命令

已运行：

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
- full rebuild: 400 tasks processed, 51 solved, 349 failed
- `inspect_submission`: passed, 51 ONNX models
- `failure_taxonomy`: 349 rows
- `rule_near_miss`: 349 rows
- `candidate_discovery_report`: 11 rows
- `probe_summary`: scanned 349 failed tasks
- `git diff --check`: passed

### 结果

- Local train solved: 51 / 400
- failed: 349 / 400
- 本轮新增 solved: 6
- solved 模型 estimated cost 总和: 182027
- solved 模型 ONNX file size 总和: 214159 bytes
- `outputs/submission.zip`: 35042 bytes

本轮新增 solved：

- `task031` -> DynamicNonBackgroundBBoxCropRule, cost 1113, file 3759
- `task150` -> DynamicActiveMirrorRule, cost 622, file 2887
- `task155` -> DynamicActiveMirrorRule, cost 622, file 2886
- `task177` -> DynamicNonBackgroundBBoxCropRule, cost 1113, file 3769
- `task259` -> DynamicNonBackgroundBBoxCropRule, cost 1113, file 3759
- `task290` -> DynamicBBoxExtremeColorSwapRule, cost 713, file 4864

### 当前剩余 probe 候选

`candidate_discovery_report.csv` 当前剩余 7 个失败任务、11 条候选：

- FrameInteriorRule: 5
- DynamicBBoxCropRule: 2
- ComposedRuleSearch: 2
- PanelSemanticRule: 2

剩余命中任务：

`task036`, `task065`, `task079`, `task100`, `task153`, `task174`, `task207`.

### 结论

本轮从“策略 probe-only”继续推进到正式 ONNX builder，solved 从 45 提升到 51。剩余候选需要动态 panel selection、component selection 或更细 frame interior builder；这些仍需逐项保守实现，不能直接把宽松 probe 放进 submission。

## 2026-05-31 22:20 - 优化策略 probe/discovery 全量接入

### 目标

根据更新后的 `优化策略.md`，逐项补齐 Panel v2、DynamicBBox/ObjectNormalize/FrameInterior/MarkerGuided crop、Frame/Interior/Border、ObjectEdit、Rule Composition、候选发现报告和验证流程。

本轮采取 probe-first 策略：先实现能够严格解释所有 train 样例的 Python matcher 和机器可读 discovery report；没有安全 ONNX builder 的策略不进入 `first_version_rules()`，也不进入 `submission.zip`。

### 修改文件

- `src/pattern_rules.py`
- `src/candidate_discovery_report.py`
- `tests/test_pattern_rules.py`
- `outputs/reports/candidate_discovery_report.csv`
- `outputs/reports/probe_summary.csv`
- `outputs/reports/failure_taxonomy.csv`
- `outputs/reports/rule_near_miss.csv`
- `outputs/reports/summary.csv`
- `outputs/onnx/*.onnx`
- `outputs/candidates/*.onnx`
- `outputs/logs/*.json`
- `outputs/submission.zip`
- `PROGRESS.md`
- `EXPERIMENT_LOG.md`
- `优化策略.md`

### 实现内容

- 新增 `PanelSemanticRule` probe：
  - 支持变布局 panel 发现。
  - 支持 unique color、color absent from others、most/least non-background、different shape、output after color map、crop、rotate/mirror 等 selector。
  - 当前阻塞：`builder_missing_dynamic_panel_select`。
- 新增 `DynamicBBoxCropRule` probe：
  - 支持 all non-background bbox、per-color bbox、largest/smallest component、unique component、not-touching-border component。
  - 输出匹配支持 crop + color map。
  - 当前阻塞：`builder_missing_dynamic_bbox`。
- 新增 `FrameInteriorRule` probe：
  - 支持矩形 frame 枚举。
  - 支持 frame interior crop 和 frame fill。
  - 当前阻塞：`builder_missing_dynamic_bbox`。
- 新增 `ObjectEditRule` probe：
  - 支持 isolated noise removal。
  - 支持 object outline。
  - 真实失败任务本轮 0 命中，暂不优先 builder。
- 新增 `ComposedRuleSearch` probe：
  - 支持 bbox/panel extract -> identity/mirror/rotate -> color map 的两步组合。
  - 当前阻塞：`requires_composed_rule`。
- 新增 `src/candidate_discovery_report.py`：
  - 输出字段：`task_id`, `candidate_rule`, `python_transform_passed`, `onnx_builder_available`, `onnx_validation_passed`, `blocked_reason`, `estimated_codegen_difficulty`, `expected_gain_bucket`。
  - 默认只扫描当前 failed 任务。
  - 对 probe-only 规则只记录发现结果，不构建 ONNX。
- 将新增 probe 接入 `third_round_probe_rules()`，用于 aggregate probe summary。
- 未将新增 probe-only 规则加入 `first_version_rules()`。
- 新增 6 个相关测试，覆盖新 matcher 和 discovery report 写出。

### 验证命令

已运行：

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
- `pytest`: 53 passed
- full rebuild: 400 tasks processed, 45 solved, 355 failed
- `inspect_submission`: passed, 45 ONNX models
- `failure_taxonomy`: 355 rows
- `rule_near_miss`: 355 rows
- `candidate_discovery_report`: 21 rows
- `probe_summary`: scanned 355 failed tasks
- `git diff --check`: passed

### 当前结果

- Local train solved: 45 / 400
- Failed: 355 / 400
- solved 模型 estimated cost 总和: 176731
- solved 模型 ONNX file size 总和: 192235 bytes
- `outputs/submission.zip`: 27700 bytes

`candidate_discovery_report.csv` 发现 13 个失败任务、21 条 train 可解释候选：

- PanelSemanticRule: 2
- DynamicBBoxCropRule: 4
- FrameInteriorRule: 8
- ObjectEditRule: 0
- ComposedRuleSearch: 7

命中任务：

`task031`, `task036`, `task065`, `task079`, `task100`, `task150`, `task153`, `task155`, `task174`, `task177`, `task207`, `task259`, `task290`.

### 结论

`优化策略.md` 中要求的策略已逐项完成到安全状态：Python probe、候选发现报告、测试和全量验证均已完成；动态 selector / bbox / frame / composition 的 ONNX builder 仍按安全原则阻断，未进入 submission。

下一轮应优先从 `candidate_discovery_report.csv` 中同时命中 DynamicBBox/Frame/Composition 的任务开始做最小安全 builder，尤其是 `task031`, `task036`, `task174`, `task259`。
