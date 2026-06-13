# NeuroGolf 2026 Current Model/Algorithm Summary

This repository solves the NeuroGolf / ARC-AGI ONNX golf task. The goal is not
to train one large network. The goal is to compile each ARC task into one small,
static, legal ONNX network that exactly maps the task input grid to the output
grid while minimizing official cost.

## Current Best Submission

- Active submit-ready package:
  - `outputs/submission.zip`
  - byte-identical to user-provided `6348.56submission.zip`
  - known online score: `6348.56`
- Layout:
  - `base_submission/task001.onnx` ... `task400.onnx`
  - `overrides/task001.onnx` ... `task400.onnx`
  - 800 ONNX files total, 400 task IDs
- This is an external hybrid stack. It is structurally validated locally, but
  the exact per-task online contribution is unknown.
- Previous known baseline:
  - `outputs/submissions/6275_normalized_submission.zip`
  - flat layout: `taskNNN.onnx`
  - known online score: `6275.09`

Important result: replacing 6348.56 tasks with 6275.09 tasks by local cost did
not produce any online improvement in one-task tests. Therefore local structural
cost is not a safe proxy for online task-level improvement.

## Data Representation

- ARC grids are discrete colors `0..9`.
- Internal tensor convention is one-hot channels:
  - color 0 -> channel 0
  - ...
  - color 9 -> channel 9
- Tensor layout is usually NCHW:
  - `[1, 10, 30, 30]`
- Most generated models use static shapes and fixed-size padded grids.
- Padding semantics are important. Some local strict failures are due to
  nonzero padding cells, even when online packages are trusted from leaderboard
  evidence.

## ONNX Constraints

All candidate models must satisfy:

- static tensor shapes;
- no Python runtime at inference;
- no external files;
- deterministic inference;
- `onnx.checker.check_model` passes;
- no forbidden ops:
  - `Loop`
  - `Scan`
  - `NonZero`
  - `Unique`
  - `Script`
  - `Function`
  - project also treats `Compress` as forbidden;
- each ONNX file under the contest file-size limit;
- exact grid equality on all train examples when locally validated.

## Solver Pipeline

The local solver is rule-based:

1. Load one ARC task.
2. Analyze train examples:
   - input/output shapes;
   - color sets;
   - background;
   - changed cells;
   - object/panel relations.
3. Run conservative rule matchers.
4. For every matched rule, build a candidate ONNX model.
5. Validate candidate:
   - ONNX checker;
   - forbidden ops;
   - static shape check;
   - onnxruntime train-case inference;
   - exact decoded grid equality;
   - file size and cost estimate.
6. Select the lowest-cost valid candidate for that task.
7. Write reports/logs and package only validated models.

Core files:

- `src/solve_task.py`
- `src/pattern_rules.py`
- `src/onnx_builders.py`
- `src/evaluate_onnx_candidate.py`
- `src/inspect_submission.py`
- `src/cost_estimator.py`

## Rule Families Implemented Locally

The local rule system includes conservative matchers/builders for:

- identity;
- global color mapping;
- one-step and multi-step translations;
- mirror and rotate;
- crop / bbox crop / non-background bbox crop;
- scale-repeat, tile-repeat, mirror-concat;
- periodic extension with optional color map;
- panel separator binary operations;
- generalized panel operations;
- quadrant/panel selection;
- object selection and object edit;
- local neighborhood fill / rewrite;
- hole fill;
- symmetry completion;
- line extension / line projection;
- rectangle and line rules;
- frame interior extraction;
- largest-frame recolor/crop;
- rectangular cavity fill;
- marker-based stripes and bands;
- diagonal-bottom fill;
- dynamic active mirror;
- static overlay and spatial remap;
- self-Kronecker mask patterns;
- composed rule search.

The intended philosophy is: prefer symbolic, explainable transformations that
generalize from all train examples, not train-output lookup tables.

## ONNX Builder Patterns

Common compact graph structures:

- `Identity` for exact identity tasks.
- `Gather` for channel permutations or spatial indexing.
- `Conv` 1x1 for color remaps/channel mixing.
- Small fixed spatial remaps using row/column `Gather`.
- Elementwise `Add`, `Sub`, `Mul`, `Relu`, `Clip`.
- Boolean/float masks stored as small initializers.
- Avoid large constants unless no safer symbolic builder exists.

The best local optimization opportunities have often involved replacing dense
one-hot tables with indexed `Gather` patterns, or pruning unused rows/ranges
from existing tables.

## Cost Model Used Locally

Local estimated cost currently:

```text
estimated_cost = num_initializer_parameters + initializer_memory_bytes
estimated_score = max(1, 25 - ln(estimated_cost))
```

This local cost is useful for comparing structurally similar models, but it has
repeatedly failed as a reliable proxy for online score when the graph structure,
dtypes, or submission layout changes.

## Known Online Evidence

Promoted/safe-ish patterns from previous online tests:

- gather/index structural rewrites;
- conservative row-bank prefix pruning;
- observed prior-range pruning;
- unreachable enumeration-table row pruning;
- simple geometric rule builders that were verified online one task at a time.

Rejected/risky patterns from previous online tests:

- dtype compression, including int64 -> int32 Gather indices;
- replacing Conv with Pad/Slice/Concat structures;
- task-specific semantic templates that pass local train/labelled tests but
  regress online;
- semantic cavity/frame/panel detection rules without online one-task proof;
- batch merges based only on local cost;
- zero-initializer changes that display as same-score locally/online rounded
  but regress when merged.

## Recent 6275.09 vs 6348.56 Experiment

We built a local-cost candidate by choosing among:

- 6275 flat model: `outputs/reference_6275_flat/taskNNN.onnx`;
- 6348 base lane: `outputs/reference_6348_56_stack/base_submission/taskNNN.onnx`;
- 6348 overrides lane: `outputs/reference_6348_56_stack/overrides/taskNNN.onnx`.

Local-cost source split:

- 53 tasks from 6275;
- 50 tasks from 6348 base;
- 297 tasks from 6348 overrides;
- local estimated cost sum: `673,220`.

We also generated one-task ablations where each package kept 6348.56 as the
base and replaced both lanes for exactly one selected task with the 6275 model.
User reported none was better online. This strongly suggests not promoting
6275-over-6348 replacements based only on local cost.

## What We Want From an External Model

Please propose optimization ideas under these constraints:

1. Do not suggest training a large CNN as the primary strategy.
2. Do not rely on dynamic shapes, runtime Python, external files, randomness,
   or forbidden ONNX ops.
3. Do not assume local estimated cost is equivalent to official online score.
4. Prefer transformations that preserve graph semantics and dtype behavior.
5. Prefer one-task-at-a-time online-testable changes.
6. Explain how to validate each idea locally and how to design a one-task
   ablation for online verification.

Useful directions to analyze:

- Can the 6348.56 hybrid stack be decompiled into families of reusable graph
  templates?
- Can repeated graph motifs across tasks be simplified without changing dtypes
  or operator classes?
- Are there safe Gather/table pruning opportunities inside the 6348.56 stack?
- Are there unused initializers, duplicate initializers, or dead nodes that can
  be removed while preserving exact graph outputs?
- Are there tasks where both 6348 lanes are byte-identical or semantically
  equivalent, and can the layout be simplified without hurting evaluator
  behavior?
- Can official dtype-aware memory/cost be approximated more accurately than the
  current local estimator?
- Can we identify task families in the 6348 stack and target the highest-cost
  family with conservative rewrites?
- Can one-task online ablation order be improved using graph features rather
  than local cost alone?

Deliverable requested from the external model:

- A prioritized list of concrete optimization experiments.
- For each experiment:
  - expected benefit;
  - risk level;
  - exact graph pattern to detect;
  - exact rewrite rule;
  - local validation procedure;
  - one-task ablation strategy;
  - reason it should be safer than previously rejected approaches.

The active baseline to optimize against is `6348.56`, not `6275.09`.
