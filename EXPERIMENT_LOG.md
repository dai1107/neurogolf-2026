# 实验日志

## 2026-06-03 23:20 - Continue high-cost top-task optimization

### Goal

Continue optimizing the remaining high-cost tasks after the earlier
`task396` replacement. Correctness and local strict validation remained the
gate for every replacement; no candidate was copied into `outputs/onnx` unless
it passed validation and reduced estimated cost.

### Implementation

- Added `DynamicLineProjectionRule` and
  `build_dynamic_line_projection_model()`.
  - Targets same-size grids with one or more full horizontal/vertical lines.
  - Projects stray cells of a full-line color to the adjacent row/column next
    to that line.
  - Used for `task025`.
- Added `DynamicRectangularCavityFillRule` and
  `build_dynamic_rectangular_cavity_fill_model()`.
  - Targets same-size 0/5 inputs where output only changes color 0 to color 4.
  - Fills rectangular color-0 cavities bounded by color-5 top/bottom walls and
    color-5 or grid-boundary side walls.
  - Includes a side-boundary termination check; this was required to pass the
    labelled `task367` test and arc-gen splits without overfilling exterior
    rectangular gaps.
- Added `TwoMarkerHorizontalBandRule` and
  `build_two_marker_horizontal_bands_model()`.
  - Targets shared-shape grids with exactly two nonzero marker cells on shared
    rows.
  - Dynamically extracts the two marker colors and draws fixed top/bottom
    horizontal band frames.
  - Used for `task028`.
- Added focused tests for the new rectangular-cavity and two-marker band
  builders; the line-projection focused test was also kept in
  `tests/test_pattern_rules.py`.

### Replacement Results

- `task025`
  - rule: `DynamicLineProjectionRule`
  - old estimated cost: 332565
  - new estimated cost: 1289
  - delta: 331276
  - extra validation: train 3/3, test 1/1, arc-gen 262/262 passed
- `task367`
  - rule: `DynamicRectangularCavityFillRule`
  - old estimated cost: 295949
  - new estimated cost: 200585
  - delta: 95364
  - extra validation: train 3/3, test 1/1, arc-gen 262/262 passed
- `task028`
  - rule: `TwoMarkerHorizontalBandRule`
  - old estimated cost: 63050
  - new estimated cost: 22600
  - delta: 40450
  - extra validation: train 2/2, test 1/1, arc-gen 262/262 passed

### Model Bank Result

Final rebuild used a 300s per-model timeout after a first 120s rebuild hit a
transient `task191` timeout. Standalone `task191` validation passed before the
final rebuild.

- selected tasks: 400 / 400
- missing or invalid tasks: 0
- estimated cost total: 7575450
- ONNX file size total: 12203846 bytes
- `outputs/submission.zip`: 1379041 bytes
- `python -m src.inspect_submission --zip outputs\submission.zip`: passed,
  400 ONNX models

### Remaining High-Cost Tasks

After the rebuild, the highest-cost tasks are:

`task133`, `task209`, `task076`, `task157`, `task233`, `task366`,
`task367`, `task363`, `task319`, `task255`.

Manual inspection notes:

- `task319`: dynamic crop/extraction, but selected component is not safely
  determined by current color/bbox/area selectors.
- `task076`: orientation-aware copy of marker patterns around color-4 objects;
  no compact safe builder was promoted.
- `task157`: transfers bottom color-5 masks into a top color-2 template; not a
  simple shift or local rewrite.
- `task255`: large-region color-3 fill over 30x30 grids; likely needs a more
  specialized large blank-region rule.

### Commands

```powershell
python -m pytest -q tests\test_pattern_rules.py -k line_projection
python -m pytest -q tests\test_pattern_rules.py -k "rectangular_cavity or two_marker_horizontal"
python -m src.search_symbolic_replacements --data-dir task --current-model-dir outputs\onnx --current-report outputs\reports\current_model_bank_report.csv --candidate-dir outputs\candidates\replacements --report outputs\reports\replacement_search_report_task025.csv --task-ids task025 --replace --timeout-seconds 120
python -m src.search_symbolic_replacements --data-dir task --current-model-dir outputs\onnx --current-report outputs\reports\current_model_bank_report.csv --candidate-dir outputs\candidates\replacements --report outputs\reports\replacement_search_report_task367.csv --task-ids task367 --replace --timeout-seconds 120
python -m src.search_symbolic_replacements --data-dir task --current-model-dir outputs\onnx --current-report outputs\reports\current_model_bank_report.csv --candidate-dir outputs\candidates\replacements --report outputs\reports\replacement_search_report_task028.csv --task-ids task028 --replace --timeout-seconds 120
python -m pytest -q tests\test_pattern_rules.py
python -m pytest -q tests\test_high_cost_replacement_search.py
python -m compileall src tests
python -m src.evaluate_onnx_candidate --model outputs\onnx\task191.onnx --task task\task191.json
python -m src.build_current_model_submission --data-dir task --model-dir outputs\onnx --validated-dir outputs\current_model_bank_verified_onnx --report outputs\reports\current_model_bank_report.csv --zip outputs\submission.zip --timeout-seconds 300
python -m src.inspect_submission --zip outputs\submission.zip
python -m pytest -q
git diff --check
```

### Validation

- Focused rectangular/band tests: 2 passed.
- `tests\test_pattern_rules.py`: 60 passed.
- `tests\test_high_cost_replacement_search.py`: 2 passed.
- Full pytest: 74 passed, 2 skipped.
- `python -m compileall src tests`: passed.
- `inspect_submission`: passed, 400 ONNX models.
- `git diff --check`: no whitespace errors; only LF-to-CRLF warnings.

### Risk

These are local strict-validation and local estimated-cost improvements, not
guaranteed official leaderboard scores. The new rules are conservative MATCH
rules; inspected but uncertain tasks were left unchanged.

## 2026-06-01 22:40 - 官方错误五任务排除 + output-name-safe padding repair

### 目标

用户反馈官方处理 `task099`, `task180`, `task266`, `task283`, `task331` 的 ONNX 网络失败。本轮先把这五个任务从主 `submission.zip` 中排除，再定位原因并生成不破坏已成功任务的修复候选。

### 修改文件

- `src/blend_archive_submission.py`
- `src/repair_archive_padding.py`
- `outputs/submission.zip`
- `outputs/submission_candidate_active_static.zip`
- `outputs/reports/archive_blend_report.csv`
- `outputs/reports/archive_blend_active_static_report.csv`
- `outputs/reports/archive_padding_repair_active_report.csv`
- `outputs/reports/archive_padding_repair_active_static_report.csv`
- `outputs/archive_repaired_active/*.onnx`
- `outputs/archive_repaired_active_static/*.onnx`
- `outputs/archive_blended_active_static_onnx/*.onnx`
- `PROGRESS.md`
- `EXPERIMENT_LOG.md`

### 实现内容

- `blend_archive_submission` 新增 `--exclude-task-ids`。
  - 先重建主 `outputs/submission.zip`，排除 `task099`, `task180`, `task266`, `task283`, `task331`。
  - 主包现在是安全回退版，383 个 ONNX。
- 修复 `repair_archive_padding` 的 output name 问题。
  - 原修复版输出名变成 `masked_output`，疑似导致官方 processing error。
  - 新修复保留 graph output 名 `output`，内部把原 producer 输出改成 `output_unmasked`。
- 修复 dtype 问题。
  - `task004` 原 archive output 是 `float16`，追加 float32 mask 会导致 ORT 类型错误。
  - 现在静态 mask dtype 与 graph output dtype 一致；动态 active mask cast 到 graph output dtype。
- 新增 active-mask repair 模式。
  - 用 1x1 `Conv` 从 input one-hot 计算 active 区域，避免 `ReduceSum` axes 在高 opset 模型上的兼容问题。
  - 修复 input/output 同尺寸但 train shape 可变的 padding 非零任务。

### 验证命令

```powershell
python -m compileall src tests
python -m pytest -q
python -m src.blend_archive_submission --archive-dir outputs\archive_repaired --current-dir outputs\onnx --blended-dir outputs\archive_blended_onnx --report outputs\reports\archive_blend_report.csv --zip outputs\submission.zip --timeout-seconds 120 --exclude-task-ids task099,task180,task266,task283,task331
python -m src.inspect_submission --zip outputs\submission.zip
python -m src.repair_archive_padding --output-dir outputs\archive_repaired_active --repair-report outputs\reports\archive_padding_repair_active_report.csv --mode active --task-ids task004,task098,task099,task120,task122,task266,task283,task331,task344
python -m src.repair_archive_padding --archive-dir outputs\archive_repaired_active --output-dir outputs\archive_repaired_active_static --repair-report outputs\reports\archive_padding_repair_active_static_report.csv --task-ids task180
python -m src.blend_archive_submission --archive-dir outputs\archive_repaired_active_static --current-dir outputs\onnx --blended-dir outputs\archive_blended_active_static_onnx --report outputs\reports\archive_blend_active_static_report.csv --zip outputs\submission_candidate_active_static.zip --timeout-seconds 120
python -m src.inspect_submission --zip outputs\submission_candidate_active_static.zip
```

### 结果

- `compileall`: passed
- `pytest`: 61 passed
- Main safe submission: `outputs/submission.zip`
  - inspection passed
  - 383 ONNX models
  - 1396158 bytes
  - excludes the five official-error tasks
- Candidate repaired submission: `outputs/submission_candidate_active_static.zip`
  - inspection passed
  - 393 ONNX models
  - 1448132 bytes
  - source counts: archive/repaired archive 377, current local optimized 16
  - estimated cost total: 10494589
  - file size total: 14703185 bytes

### 新增候选修复任务

- Dynamic active-mask repair: `task004`, `task098`, `task099`, `task120`, `task122`, `task266`, `task283`, `task331`, `task344`
- Static output mask repair with output name preserved: `task180`

### 剩余风险和判断

- 主提交保持 383，避免再次提交已知官方处理失败的五个模型。
- `outputs/submission_candidate_active_static.zip` 是修复候选包，保留 `input` / `output` graph 名，理论上修复了前一版官方 processing error 的直接原因。
- 仍未解决: `task042`, `task094`, `task168`, `task184`, `task224`, `task288` 的 archive 模型本地 ORT 子进程访问冲突；`task277` 动态 shape。

## 2026-06-01 21:30 - archive baseline 验证、padding 修复与 blended submission

### 目标

用户确认 `archive/` 中的 400 个外部 baseline ONNX 可以用于提交。本轮目标是在不放松本地严格验证的前提下，把外部 baseline 作为候选源，与当前本地优化模型按 cost 合并，生成新的 `submission.zip`。

### 修改文件

- `src/blend_archive_submission.py`
- `src/evaluate_onnx_candidate.py`
- `src/repair_archive_padding.py`
- `outputs/reports/archive_blend_report.csv`
- `outputs/reports/archive_padding_repair_report.csv`
- `outputs/archive_repaired/*.onnx`
- `outputs/archive_blended_onnx/*.onnx`
- `outputs/submission.zip`
- `PROGRESS.md`
- `EXPERIMENT_LOG.md`

### 实现内容

- 新增 `src.evaluate_onnx_candidate`。
  - 单个 ONNX 候选在独立 Python 子进程中执行 checker / forbidden ops / static shape / cost / onnxruntime train validation。
  - 避免 archive 中个别模型触发 ORT 访问冲突时中断整轮验证。
- 新增 `src.blend_archive_submission`。
  - 对 `archive` 与当前 `outputs/onnx` 逐任务评估。
  - 只选择通过本地严格验证的模型。
  - 在 archive 与 current 都通过时，按 `estimated_cost`, `file_size_bytes`, source preference 选择最低成本模型。
  - 输出 `outputs/reports/archive_blend_report.csv` 和 blended `outputs/submission.zip`。
- 新增 `src.repair_archive_padding`。
  - 对 archive 中仅因 fixed-shape padding 非零失败的模型追加静态 output active mask。
  - 生成 `outputs/archive_repaired`。
  - 变输出尺寸任务不做静态 mask，避免错误验证。

### 验证命令

```powershell
python -m compileall src tests
python -m pytest -q
python -m src.blend_archive_submission --archive-dir archive --current-dir outputs\onnx --blended-dir outputs\archive_blended_onnx --report outputs\reports\archive_blend_report.csv --zip outputs\submission.zip --timeout-seconds 120
python -m src.repair_archive_padding
python -m src.blend_archive_submission --archive-dir outputs\archive_repaired --current-dir outputs\onnx --blended-dir outputs\archive_blended_onnx --report outputs\reports\archive_blend_report.csv --zip outputs\submission.zip --timeout-seconds 120
python -m src.inspect_submission --zip outputs\submission.zip
```

### 结果

- `compileall`: passed
- `pytest`: 61 passed
- First archive blend: 383 selected, 17 missing
- Padding repair rows: 17
- Final repaired archive blend: 388 selected, 12 missing
- Final source counts:
  - archive/repaired archive: 372
  - current local optimized: 16
- `inspect_submission`: passed, 388 ONNX models
- `outputs/submission.zip`: 1420939 bytes
- Blended selected estimated cost total: 10474834
- Blended selected ONNX file size total: 14683926 bytes

### 剩余未纳入任务

- `task004`, `task098`, `task120`, `task122`, `task344`: archive 模型 active 区域之外 padding 非零，但 train output shape 可变，当前未做动态 mask 包装。
- `task042`, `task094`, `task168`, `task184`, `task224`, `task288`: archive 模型在 ORT 子进程中返回 `3221225477`，本地视为运行时不可用。
- `task277`: archive 模型 shape inference 出现动态 shape，违反静态 shape 约束。

### 判断

当前最终 `outputs/submission.zip` 是一个保守 blended submission，只包含本地严格验证通过的 ONNX。虽然 archive 来源声称覆盖 400 任务，但本项目不把未通过本地严格验证、运行时崩溃或动态 shape 的模型放入提交包。

## 2026-06-01 18:45 - 五个优化方向正式化: dynamic panel / frame interior / color bbox / safe composition

### 目标

根据 `优化策略.md` 的五个方向，把剩余 candidate-discovery 任务中已经能被 Python probe 解释的规则，尽量转成保守、可验证、可提交的 ONNX builder。正确性和 submission 安全优先于扩大命中面。

### 修改文件

- `src/onnx_builders.py`
- `src/pattern_rules.py`
- `src/solve_task.py`
- `src/candidate_discovery_report.py`
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

- 新增 `build_dynamic_quadrant_panel_select_model()` 和 `DynamicQuadrantPanelSelectRule`。
  - 针对 odd square 2x2 center-cross panel。
  - 用 panel 间差异和选出 unique max-difference quadrant。
  - 支持输出 selected panel + color map。
  - 解决 `task065`, `task207`。
- 新增 `build_dynamic_frame_interior_crop_model()`。
  - 支持 color-specific frame bbox 的 interior crop。
  - 支持 interior + color map。
  - `FrameInteriorRule` 只在 frame_color 全体像素 bbox 等于目标 frame bbox 时允许 builder；否则记录 `frame_color_bbox_contains_extra_cells`。
- 新增 `build_dynamic_color_bbox_crop_model()`。
  - 支持 `bbox_of_color` / `bbox_of_unique_color_component`。
  - 支持 identity / horizontal mirror / vertical mirror + color map。
- 扩展 `DynamicBBoxCropRule`。
  - buildable 子集: `bbox_of_all_non_background`, `bbox_of_color`, `bbox_of_unique_color_component`。
  - component-selection 类候选仍保持 blocked，不构建。
- 扩展 `ComposedRuleSearch`。
  - 安全子集: buildable bbox extractor -> identity / horizontal mirror / vertical mirror -> color map。
  - panel / component / rotate 组合仍保持 blocked，不构建。
- 更新 `solve_task`。
  - 对 `metadata["builder_available"] is False` 的匹配显式跳过构建，只记录 blocked reason。
  - 避免 probe-only 分支以异常形式污染失败日志。
- 更新 `candidate_discovery_report`。
  - `DynamicBBoxCropRule`, `FrameInteriorRule`, `ComposedRuleSearch` 改为按 metadata 判断 builder availability。
- 新增/更新测试。
  - dynamic quadrant panel select 的 unique color / unique pattern。
  - color-specific dynamic bbox crop builder。
  - dynamic frame interior crop builder。
  - composed safe bbox extractor + mirror builder。

### 验证命令

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

### 结果

- targeted pattern tests: 52 passed
- full pytest: 61 passed
- compileall: passed
- full rebuild: 400 tasks processed, 53 solved, 347 failed
- `inspect_submission`: passed, 53 ONNX models
- `failure_taxonomy`: 347 rows
- `rule_near_miss`: 347 rows
- `candidate_discovery_report`: 6 rows
- `probe_summary`: scanned 347 failed tasks
- `git diff --check`: passed with line-ending warnings only
- `outputs/submission.zip`: 60236 bytes

本轮新增 solved:
- `task065` -> `DynamicQuadrantPanelSelectRule`, cost 93935, file 144684
- `task207` -> `DynamicQuadrantPanelSelectRule`, cost 93935, file 144684

当前汇总:
- Local train solved: 53 / 400
- Failed: 347 / 400
- solved 模型 estimated cost 总和: 369897
- solved 模型 ONNX file size 总和: 503527 bytes

### 剩余阻塞

`candidate_discovery_report.csv` 当前剩余 6 行:
- `task036`: `DynamicBBoxCropRule` -> `builder_missing_dynamic_bbox`
- `task036`: `FrameInteriorRule` -> `frame_color_bbox_contains_extra_cells`
- `task036`: `ComposedRuleSearch` -> `requires_composed_rule`
- `task079`: `FrameInteriorRule` -> `frame_color_bbox_contains_extra_cells`
- `task174`: `DynamicBBoxCropRule` -> `builder_missing_dynamic_bbox`
- `task174`: `ComposedRuleSearch` -> `requires_composed_rule`

这些剩余项需要真正的 component-selection 或更细的 frame/color-role selector；当前不能安全地用 color bbox 或 frame bbox 近似，否则会破坏泛化或 train validation。

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

## 2026-06-02 - Archive ONNX repair and 400-model validated submission

### Goal

Continue from the interrupted archive-blend work: analyze unfinished tasks, repair safe existing models, and rebuild a validated submission without adding unverified models.

### Findings

- Existing validated package had 393 models.
- `task277` archive model passed train validation but failed only static shape inference on dynamic Pad outputs.
- The remaining six raw archive models (`task042`, `task094`, `task168`, `task184`, `task224`, `task288`) passed ONNX checker, forbidden-op checks, static-shape checks, and cost checks, but default onnxruntime evaluation crashed with return code 3221225477.
- Running those six with ORT optimizations disabled succeeded, pointing to an ORT graph-optimization crash rather than a rule mismatch.
- Graph inspection found Conv nodes with negative `pads` attributes in those six models. The repair rewrites each such Conv as `Slice` crop plus Conv with non-negative pads, preserving output behavior under default ORT.

### Implementation

- Added `repair_task277_static_pads()`:
  - replaces `task277` dynamic Pad pads inputs with static int64 initializers;
  - keeps graph output shape static `[1, 10, 30, 30]`;
  - validates with shape inference and ONNX checker before saving.
- Added `repair_negative_conv_pads()`:
  - detects Conv nodes with negative `pads`;
  - inserts static `Slice` nodes to crop the input;
  - rewrites Conv `pads` to non-negative values;
  - reuses identical Slice constants within each model to reduce initializer cost.
- Added `tests/test_repair_archive_padding.py` covering both repair paths against archive fixtures.

### Repaired Tasks

- `task277`: static Pad repair, valid, estimated cost 29994, file size 57173.
- `task042`: negative Conv pads repair, valid, estimated cost 1356, file size 11833.
- `task094`: negative Conv pads repair, valid, estimated cost 910, file size 4915.
- `task168`: negative Conv pads repair, valid, estimated cost 1667, file size 9227.
- `task184`: negative Conv pads repair, valid, estimated cost 340, file size 7610.
- `task224`: negative Conv pads repair, valid, estimated cost 1240, file size 7753.
- `task288`: negative Conv pads repair, valid, estimated cost 821, file size 13869.

### Commands

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

### Results

- Full blend selected 400 / 400 tasks.
- Missing tasks: 0.
- Source counts: archive 384, current 16.
- Selected estimated cost total: 10530917.
- Selected ONNX file size total: 14815565 bytes.
- `outputs/submission_validated_400.zip`: inspection passed, 400 ONNX models.
- `outputs/submission.zip`: updated from the validated 400 zip and inspection passed, 400 ONNX models.
- Full tests: 63 passed.
- `git diff --check`: passed, with only the existing LF-to-CRLF warning for `src/repair_archive_padding.py`.

### Risk

- This is strict local train validation plus local default onnxruntime validation, not a guaranteed official leaderboard score.
- The six repaired archive models are behavioral graph rewrites of already small archive models; they should remain more reliable than disabling ORT optimizations in the validator.

## 2026-06-02 - Repository cleanup and artifact deduplication

### Goal

Remove stale generated files and exact duplicates while preserving the current validated submission path.

### Actions

- Deleted Python cache directories and pytest cache.
- Deleted obsolete archive repair/blend output directories from previous rounds.
- Deleted candidate/validated zip files superseded by `outputs/submission.zip`.
- Verified before deletion that `outputs/submission_validated_400.zip` and `outputs/submission.zip` had identical SHA-256 hash:
  `EB144EC0C55BE84897142F7A636F11F776E10778DF4B4E2DD382E644AE166C3A`.
- Kept `outputs/archive_all_repaired_verified_onnx` as the canonical validated 400-model ONNX directory.
- Added `.gitignore` coverage for caches, generated zip files, debug packs, and archive output directories.

### Risk

- This cleanup did not change model source code or ONNX graph contents.
- Historical reports were retained for auditability; obsolete generated model directories can be regenerated from the logged commands if needed.

## 2026-06-03 - Independent current model-bank build

### Goal

Make `outputs/onnx` the canonical best-known model bank so `archive` is no
longer needed to rebuild the 400-task submission.

### Implementation

- Copied the previously selected best validated models into `outputs/onnx`.
- Added `src/build_current_model_submission.py` to validate every
  `outputs/onnx/taskNNN.onnx`, copy passing models to a temporary verified
  directory, and build `outputs/submission.zip`.
- Added `tests/test_build_current_model_submission.py` for successful local
  model-bank packaging and incomplete-bank rejection.
- Added a guard so the verified output directory cannot be the same directory
  as the source model bank, and stale verified `task*.onnx` files are removed
  before each rebuild.
- Added `EXTERNAL_OPTIMIZATION_CONTEXT.md` with validation commands, current
  metrics, and the highest-cost optimization targets.

### Results

- Full local validation selected 400 / 400 tasks.
- Missing or invalid tasks: 0.
- Estimated cost total: 10530917.
- ONNX file size total: 14815565 bytes.
- `outputs/submission.zip`: inspection passed with 400 ONNX models.
- Highest-cost current targets: `task133`, `task084`, `task209`, `task076`,
  `task157`, `task200`, `task233`, `task025`, `task367`, `task366`.
- Final cleanup removed the baseline `archive` directory, obsolete archive/blend
  reports, duplicate generated ONNX directories, temporary validation output,
  and cache directories.
- Full pytest passed with 64 passed and 2 skipped.
- `python -m compileall src tests` passed.
- `git diff --check` passed with only LF-to-CRLF warnings for Markdown logs.

### Validation Commands

```powershell
python -m src.build_current_model_submission --data-dir task --model-dir outputs\onnx --validated-dir outputs\current_model_bank_verified_onnx --report outputs\reports\current_model_bank_report.csv --zip outputs\submission.zip --timeout-seconds 120
python -m src.inspect_submission --zip outputs\submission.zip
python -m pytest -q tests\test_build_current_model_submission.py
python -m pytest -q
python -m compileall src tests
git diff --check
```

### Risk

- This is strict local train validation, not a guaranteed official leaderboard
  score.
- The archived baseline is no longer a required dependency, but the current
  models still inherit some high-cost baseline graphs that should be optimized
  next.

## 2026-06-03 - High-cost symbolic replacement search workflow

### Goal

Implement the requested high-cost model replacement workflow and run the first
round on the top seven current cost targets without admitting unverified models.

### Implementation

- Added `src/diagnose_high_cost_tasks.py`.
  - Reads `outputs/reports/current_model_bank_report.csv`.
  - Loads task JSON files from `task/`.
  - Writes `outputs/reports/high_cost_task_diagnosis.csv`.
  - Writes per-task readable analyses under
    `outputs/reports/high_cost_task_analysis/taskNNN.md`.
- Added `src/search_symbolic_replacements.py`.
  - Selects top-k high-cost tasks or explicit task IDs from the current report.
  - Runs every formal `first_version_rules()` rule.
  - Builds ONNX candidates only for MATCH results with available builders.
  - Validates each candidate through isolated `src.evaluate_onnx_candidate`,
    including checker, forbidden ops, static shapes, default onnxruntime, exact
    train validation, and file-size/cost checks.
  - Marks `replace_recommended=True` only when the candidate is valid and lower
    cost than the current model.
  - Supports `--replace`; replacement copying is still gated by the validated
    lower-cost check.
- Added `tests/test_high_cost_replacement_search.py`.

### First-Round Tasks

`task133`, `task084`, `task209`, `task076`, `task157`, `task200`, `task233`.

### Results

- `outputs/reports/high_cost_task_diagnosis.csv`: 7 rows.
- Per-task analysis Markdown files written for all 7 tasks.
- `outputs/reports/replacement_search_report.csv`: 217 rows.
- Formal rules searched: 31 per task.
- Replacement count: 0.
- No `outputs/onnx/taskNNN.onnx` files were replaced because all seven tasks
  were rejected by the current formal matchers before candidate construction.
- Full model-bank rebuild still selected 400 / 400 tasks with 0 missing or
  invalid tasks.
- Current estimated cost total remains 10530917.
- Current ONNX file size total remains 14815565 bytes.
- `outputs/submission.zip` size remains 1466160 bytes.

### Commands

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

### Validation

- Focused tests: 2 passed.
- `python -m compileall src tests`: passed.
- `python -m src.inspect_submission --zip outputs\submission.zip`: passed,
  400 ONNX models.
- Full pytest: 66 passed, 2 skipped.
- `git diff --check`: passed.

### Next Target

The top seven remain high-cost because current formal rules cannot explain them.
Next work should add narrow probe-first rules for same-size mask algebra and
variable-size crop/object extraction, starting from the generated Markdown
diagnoses rather than trying to compress the existing baseline graphs blindly.

## 2026-06-03 - task084 DiagonalBottomFillRule replacement

### Goal

Continue the user-requested cost optimization by inspecting the task text,
`PROGRESS.md`, `EXPERIMENT_LOG.md`, and the current model-bank report, then
replace an existing high-cost model only if a compact formal rule passes strict
validation and lowers estimated cost.

### Analysis

`task084` had the second-highest current estimated cost:

- old estimated cost: 1390970
- old ONNX file size: 1127799 bytes
- train shapes: 15x15, 3x3, 7x7

All train cases satisfy the same transformation:

- input is an `n x n` square;
- first column is one repeated nonzero color;
- all other input cells are color 0;
- output preserves the first column;
- rows `0..n-2` draw color 2 on the anti-diagonal;
- final row columns `1..n-1` are color 4;
- all remaining active cells stay color 0.

### Implementation

- Added `build_dynamic_left_column_diagonal_bottom_fill_model()` in
  `src/onnx_builders.py`.
  - Uses static 30x30 shape.
  - Infers active square size from one-hot padding via `Conv`, `ReduceSum`,
    reverse `ArgMax`, and static `Offsets`.
  - Builds dynamic boolean masks for active cells, anti-diagonal cells, and
    bottom-row cells.
  - Preserves input cells outside the draw masks, including color-0 active
    background, and zeros padded output.
- Added `DiagonalBottomFillRule` in `src/pattern_rules.py`.
  - Returns MATCH only when every train case exactly matches the narrow
    left-column / anti-diagonal / bottom-row specification.
  - Added to `first_version_rules()`.
- Added focused tests in `tests/test_pattern_rules.py`.
  - Positive build/validate cases with multiple active sizes.
  - Rejection checks for non-square shape and extra non-left input cells.

### Replacement Result

`src.search_symbolic_replacements` was run for `task084` only with `--replace`.
The replacement gate selected the new candidate because it passed validation and
was strictly lower cost:

- rule: `DiagonalBottomFillRule`
- new estimated cost: 722
- new ONNX file size: 4301 bytes
- cost delta: 1390248
- replacement path: `outputs/onnx/task084.onnx`
- candidate path: `outputs/candidates/replacements/task084_DiagonalBottomFillRule.onnx`

### Model Bank Result

After rebuilding the current model bank and submission:

- selected tasks: 400 / 400
- missing or invalid tasks: 0
- estimated cost total: 9140669
- ONNX file size total: 13692067 bytes
- `outputs/submission.zip` inspection passed with 400 ONNX models

### Commands

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

### Validation

- `tests/test_pattern_rules.py`: 54 passed.
- Focused combined tests: 59 passed.
- Full pytest: 68 passed, 2 skipped.
- `python -m compileall src tests`: passed.
- `python -m src.inspect_submission --zip outputs\submission.zip`: passed,
  400 ONNX models.
- `git diff --check`: no whitespace errors; only LF-to-CRLF warnings.

### Risk

This is strict local train validation and local estimated cost. It is not a
guaranteed official leaderboard score.

## 2026-06-03 - external optimization context handoff

### Goal

Generate a concise file that can be handed to an external reviewer or optimizer
so they can understand the current repository state and propose safe cost
optimization ideas.

### Output

- Created `EXTERNAL_OPTIMIZATION_CONTEXT.md`.

### Contents

The handoff document includes:

- hard ONNX and competition constraints;
- local cost definition;
- current canonical model bank status;
- current submission and validation status;
- recent successful replacements for `task084` and `task200`;
- top-25 remaining high-cost tasks;
- relevant source files, reports, and scripts;
- commands to rebuild and inspect the current submission;
- recommended external review priorities;
- suggested response format for external optimizer feedback;
- open questions for the highest-cost remaining tasks.

### Validation

No code or ONNX behavior changed in this step. The file is documentation-only.

## 2026-06-03 - task200 BottomMarkerVerticalStripeRule replacement

### Goal

Continue quick cost optimization after the user reported a modest online score
increase. Prefer tasks with a clear, narrow symbolic rule and avoid admitting
uncertain probe-only rules into the model bank.

### Analysis

After `task084` was replaced, the next high-cost same-shape targets included
`task133`, `task076`, `task157`, `task200`, and `task025`. `task200` was the
clearest quick replacement candidate:

- old estimated cost: 990050
- old ONNX file size: 797905 bytes
- train shapes: 10x10 in all cases
- each input has exactly one nonzero marker on the bottom row
- marker colors in train: 2, 3, 4

The rule inferred from all train cases:

- preserve active background cells as color 0;
- draw the marker color in every active row at columns
  `marker_col, marker_col + 2, marker_col + 4, ...`;
- draw color 5 on the top row at columns `marker_col + 1 + 4k`;
- draw color 5 on the bottom row at columns `marker_col + 3 + 4k`;
- clip all generated positions to the active width and zero padded output.

`task157`, `task025`, `task367`, and `task363` were also inspected. Their
patterns looked plausible but not sufficiently clear for a conservative MATCH
within this quick pass, so no candidates were generated for them.

### Implementation

- Added `build_dynamic_bottom_marker_vertical_stripes_model()` in
  `src/onnx_builders.py`.
  - Infers active height/width from one-hot padding.
  - Infers the unique nonzero marker column via `Conv`, `ReduceSum`, and
    `ArgMax`.
  - Infers the marker color via masked `ReduceMax`.
  - Builds static-shape dynamic column masks using `Offsets`, scalar deltas,
    `Add`, `Equal`, and `Or`.
  - Outputs `input * keep + marker_color * vertical_mask + color5 * connector_mask`.
- Added `BottomMarkerVerticalStripeRule` in `src/pattern_rules.py`.
  - Returns MATCH only when every train case exactly matches the bottom-marker
    vertical stripe specification.
  - Added to `first_version_rules()`.
- Added focused tests in `tests/test_pattern_rules.py`.
  - Positive build/validate cases with different active sizes, marker columns,
    and marker colors.
  - Rejection checks for non-bottom and multiple markers.

### Replacement Result

`src.search_symbolic_replacements` was run for `task200` only with `--replace`.
The replacement gate selected the new candidate because it passed validation and
was strictly lower cost:

- rule: `BottomMarkerVerticalStripeRule`
- new estimated cost: 992
- new ONNX file size: 14264 bytes
- cost delta: 989058
- replacement path: `outputs/onnx/task200.onnx`
- candidate path: `outputs/candidates/replacements/task200_BottomMarkerVerticalStripeRule.onnx`

### Model Bank Result

After rebuilding the current model bank and submission:

- selected tasks: 400 / 400
- missing or invalid tasks: 0
- estimated cost total: 8151611
- ONNX file size total: 12908426 bytes
- `outputs/submission.zip` inspection passed with 400 ONNX models

### Commands

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

### Validation

- `tests/test_pattern_rules.py`: 56 passed.
- Focused combined tests: 61 passed.
- Full pytest: 70 passed, 2 skipped.
- `python -m compileall src tests`: passed.
- `python -m src.inspect_submission --zip outputs\submission.zip`: passed,
  400 ONNX models.
- `git diff --check`: no whitespace errors; only LF-to-CRLF warnings.

### Risk

This is strict local train validation and local estimated cost. It is not a
guaranteed official leaderboard score.
## 2026-06-03 - task396 DynamicLargestFrameRecolorCropRule replacement

### Goal

Continue the requested optimization round using the task priorities in
`任务文本`. After inspecting the first-priority same-shape mask targets
(`task133`, `task076`, `task157`) without finding a safe quick MATCH, switch to
the second-priority dynamic crop group and target `task396`.

### Analysis

`task396` has two nonzero colors in every inspected example. The larger-count
color forms several rectangular hollow frames; the smaller-count color is a
marker/noise color scattered inside and outside the frames. The output is the
largest frame crop, with every nonzero cell inside that crop recolored to the
marker color while real zero cells remain color 0.

Additional task data was used only for confidence checking, not as ONNX runtime
state:

- train cases: 3
- test cases with labels: 1
- arc-gen cases: 262
- observed output dimensions across all available cases: 4..8 by 4..8
- observed nonzero color sets: exactly two nonzero colors per case

### Implementation

- Added `build_dynamic_largest_frame_recolor_crop_model()` in
  `src/onnx_builders.py`.
  - Counts input color channels and dynamically selects the most frequent
    nonzero source/frame color and least frequent nonzero marker color.
  - Builds a source-color mask.
  - Enumerates static frame kernels for all 4..8 height/width combinations.
  - Selects the largest valid frame by ONNX graph logic.
  - Uses `Gather` to crop the selected bbox.
  - Outputs marker color for nonzero crop cells, color 0 for real zero crop
    cells, and all-zero padding outside the selected crop.
- Added `DynamicLargestFrameRecolorCropRule` in `src/pattern_rules.py`.
  - MATCH only when every train case has exactly two nonzero colors, a unique
    larger-count source color, a unique smaller-count marker color, a unique
    largest source-color frame in 4..8 dimensions, and exact output equality.
  - Added to `first_version_rules()`.
- Added a focused synthetic build/validate test in
  `tests/test_pattern_rules.py`.

### Replacement Result

`src.search_symbolic_replacements` was run for `task396` with `--replace`.
The new candidate passed validation and was selected:

- rule: `DynamicLargestFrameRecolorCropRule`
- old estimated cost: 115080
- new estimated cost: 6009
- cost delta: 109071
- old ONNX file size: 148123 bytes
- new ONNX file size: 50393 bytes
- replacement path: `outputs/onnx/task396.onnx`
- candidate path:
  `outputs/candidates/replacements/task396_DynamicLargestFrameRecolorCropRule.onnx`

Extra confidence validation on all labelled `task396` splits:

- train: 3 / 3 passed
- test: 1 / 1 passed
- arc-gen: 262 / 262 passed

### Model Bank Result

After rebuilding the current model bank and submission:

- selected tasks: 400 / 400
- missing or invalid tasks: 0
- estimated cost total: 8042540
- ONNX file size total: 12810696 bytes
- `outputs/submission.zip` inspection passed with 400 ONNX models

### Commands

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

### Validation

- Focused largest-frame test: 1 passed.
- `tests/test_pattern_rules.py`: 57 passed.
- `tests/test_high_cost_replacement_search.py`: 2 passed.
- Full pytest: 71 passed, 2 skipped.
- `python -m compileall src tests`: passed.
- `python -m src.inspect_submission --zip outputs\submission.zip`: passed,
  400 ONNX models.
- `git diff --check`: no whitespace errors; only LF-to-CRLF warnings.

### Risk

This is strict local validation and local estimated cost. The extra `test` and
`arc-gen` validation increases confidence for this rule, but it is still not a
guaranteed official leaderboard score.
