# External Optimization Context

Date: 2026-06-03

This repository is a NeuroGolf / ARC-AGI ONNX network-golf project. The goal is
not to train a large model. The goal is to compile each ARC task into one small,
valid, deterministic ONNX network that exactly solves the task transformation.

Use this document to review the current state and propose further optimization
ideas. Please prioritize correctness and compliance over size reduction.

## Hard Constraints

- One ONNX model per task: `task001.onnx` through `task400.onnx`.
- Static tensor shapes only.
- Single ONNX file size must be at most 1.44 MB.
- Forbidden ONNX ops:
  - `Loop`
  - `Scan`
  - `NonZero`
  - `Unique`
  - `Script`
  - `Function`
  - `Compress`
- Inference must not depend on Python runtime, external files, randomness, or
  training-mode behavior.
- Every candidate must pass:
  - `onnx.checker.check_model`
  - forbidden-op check
  - static-shape check
  - exact train-set grid validation
  - padding-output check
  - file-size check
  - cost estimate
- Do not recommend adding unverified or uncertain models to `submission.zip`.

## Cost Definition

Local estimated cost is:

```text
estimated_cost = num_initializer_elements + initializer_memory_bytes
```

Local estimated score is not guaranteed to match the official leaderboard.

## Current Canonical State

- Canonical model bank: `outputs/onnx`
- Current submission: `outputs/submission.zip`
- Current report: `outputs/reports/current_model_bank_report.csv`
- Submission zip size: 1,456,956 bytes
- Current selected tasks: 400 / 400
- Missing or invalid tasks: 0
- Current estimated cost total: 8,151,611
- Current ONNX file size total: 12,908,426 bytes
- Latest full local validation:
  - `python -m src.inspect_submission --zip outputs\submission.zip`: passed,
    400 ONNX models
  - `python -m pytest -q`: 70 passed, 2 skipped
  - `python -m compileall src tests`: passed
  - `git diff --check`: no whitespace errors; only LF-to-CRLF warnings

## Recent Successful Cost Reductions

### task084

- Old cost: 1,390,970
- New cost: 722
- Delta: 1,390,248
- Old file size: 1,127,799 bytes
- New file size: 4,301 bytes
- Rule: `DiagonalBottomFillRule`
- Builder: `build_dynamic_left_column_diagonal_bottom_fill_model`
- Pattern:
  - Input is an `n x n` square.
  - First column is one repeated nonzero color.
  - Other input cells are color 0.
  - Output preserves the first column.
  - Color 2 is drawn on the anti-diagonal except the bottom row.
  - Color 4 fills the bottom row except the first column.

### task200

- Old cost: 990,050
- New cost: 992
- Delta: 989,058
- Old file size: 797,905 bytes
- New file size: 14,264 bytes
- Rule: `BottomMarkerVerticalStripeRule`
- Builder: `build_dynamic_bottom_marker_vertical_stripes_model`
- Pattern:
  - Input is 10x10 in train.
  - Each input has exactly one nonzero marker on the bottom row.
  - Output draws marker-color vertical stripes at
    `marker_col, marker_col + 2, marker_col + 4, ...`.
  - Output draws color 5 on the top row at `marker_col + 1 + 4k`.
  - Output draws color 5 on the bottom row at `marker_col + 3 + 4k`.

## Highest Remaining Cost Targets

Current top-cost tasks after the `task084` and `task200` replacements:

| Rank | Task | Cost | File bytes | Initial direction |
| ---: | --- | ---: | ---: | --- |
| 1 | `task133` | 1,406,822 | 783,434 | same-shape mask algebra or local rule |
| 2 | `task209` | 1,170,350 | 1,294,052 | shrink/crop/object extraction |
| 3 | `task076` | 1,147,810 | 948,872 | same-shape object/pattern completion |
| 4 | `task157` | 1,023,477 | 857,911 | same-shape mask/recolor; inspected, rule not yet safe |
| 5 | `task233` | 668,250 | 938,774 | shrink/crop/object extraction |
| 6 | `task025` | 332,565 | 826,752 | same-shape line/object movement; inspected, rule unclear |
| 7 | `task367` | 295,949 | 293,653 | same-shape fill enclosed regions; inspected, rule unclear |
| 8 | `task366` | 266,691 | 1,269,927 | crop/panel/object extraction |
| 9 | `task363` | 193,941 | 170,389 | same-shape line/pattern completion; inspected, rule unclear |
| 10 | `task396` | 115,080 | 148,123 | crop/frame extraction |
| 11 | `task319` | 78,739 | 247,340 | object selection/extraction |
| 12 | `task028` | 63,050 | 51,218 | inspect for compact rule |
| 13 | `task255` | 58,680 | 106,804 | inspect for compact rule |
| 14 | `task382` | 54,738 | 65,997 | inspect for compact rule |
| 15 | `task107` | 42,039 | 31,191 | inspect for compact rule |
| 16 | `task313` | 40,857 | 39,887 | inspect for compact rule |
| 17 | `task105` | 36,364 | 35,291 | inspect for compact rule |
| 18 | `task027` | 36,154 | 31,135 | inspect for compact rule |
| 19 | `task009` | 33,147 | 30,956 | inspect for compact rule |
| 20 | `task058` | 32,257 | 27,439 | inspect for compact rule |
| 21 | `task277` | 29,994 | 57,173 | previously repaired static Pad |
| 22 | `task153` | 26,811 | 862,351 | file-size-heavy candidate |
| 23 | `task037` | 25,086 | 20,424 | inspect for compact rule |
| 24 | `task099` | 24,605 | 20,220 | inspect for compact rule |
| 25 | `task324` | 20,090 | 30,635 | inspect for compact rule |

Per-task JSON files are in `task/taskNNN.json`.

Existing high-cost diagnosis reports, if present:

- `outputs/reports/high_cost_task_diagnosis.csv`
- `outputs/reports/high_cost_task_analysis/taskNNN.md`

## Useful Existing Code

Rule system:

- `src/pattern_rules.py`
  - `first_version_rules()` is the formal rule list used for candidate search.
  - Rules should return conservative `MATCH`, `POSSIBLE`, or `REJECT`-style
    results through `RuleResult.confidence`.
  - Only `MATCH` with a buildable ONNX builder should generate formal
    replacement candidates.

ONNX builders:

- `src/onnx_builders.py`
  - Reusable builders for identity, color maps, dynamic bbox crops, dynamic
    active mirrors, panel ops, local fills, spatial remaps, and recent
    task-specific compact dynamic builders.
  - Preferred ops include `Conv`, `Add`, `Sub`, `Mul`, `Clip`, `Gather`,
    `ReduceSum`, `ArgMax`, `Equal`, `Greater`, `Less`, `And`, `Or`, `Not`,
    `Cast`, `Where`.

Validation and cost:

- `src/validate_onnx_model.py`
- `src/cost_estimator.py`
- `src/evaluate_onnx_candidate.py`
- `src/build_current_model_submission.py`
- `src/inspect_submission.py`

High-cost workflow:

- `src/diagnose_high_cost_tasks.py`
  - Produces high-cost shape/color/task summaries.
- `src/search_symbolic_replacements.py`
  - Runs formal rules against high-cost tasks.
  - Builds candidates only for conservative `MATCH` rules with builders.
  - Validates candidates in isolated subprocesses.
  - Replaces a model only if validation passes and estimated cost is strictly
    lower than the current model.

## Commands To Reproduce The Current Model Bank

From the repository root:

```powershell
python -m src.build_current_model_submission --data-dir task --model-dir outputs\onnx --validated-dir outputs\current_model_bank_verified_onnx --report outputs\reports\current_model_bank_report.csv --zip outputs\submission.zip --timeout-seconds 120
python -m src.inspect_submission --zip outputs\submission.zip
python -m pytest -q
python -m compileall src tests
git diff --check
```

## Recommended External Review Focus

Please prioritize suggestions in this order:

1. Narrow symbolic rules for one or more high-cost tasks that can explain every
   train case exactly and plausibly generalize to test cases.
2. ONNX builder sketches using static shapes and allowed ops only.
3. Lossless graph optimizations for the existing high-cost ONNX models.
4. Focused tests or validation checks needed before admitting a replacement.

Avoid suggestions that:

- memorize train outputs as lookup tables;
- depend on dynamic shape, loops, `NonZero`, or external runtime logic;
- rely on unclear assumptions not supported by all train cases;
- improve file size but increase estimated cost or fail validation.

## Suggested Response Format For External Optimizer

For each proposed optimization, please return:

```text
task_id:
observed_rule:
why_rule_matches_all_train_cases:
uncertainties:
estimated_builder_ops:
expected_cost_range:
expected_file_size_range:
implementation_plan:
tests_to_add:
risk_level:
```

If a suggestion is only a probe and not safe for submission, label it clearly as
`PROBE_ONLY`.

## Current Open Questions

- Can `task133`, `task076`, or `task157` be reduced with a compact same-shape
  mask algebra rule rather than the current high-cost baseline graphs?
- Can `task209`, `task233`, `task366`, `task396`, or `task319` be expressed as
  dynamic bbox/object extraction with a small static-shape ONNX graph?
- Can `task367` be expressed as a conservative fill-enclosed-region or hole-fill
  variant without overfitting?
- Can `task363` be expressed as a directional line/pattern completion rule with
  local convolution-style masks?
- Are there safe lossless optimizations for large sparse initializers such as
  float 0/1 masks converted to bool + Cast, or int64 index tensors converted to
  int32 where ONNX Runtime accepts them and validation still passes?
