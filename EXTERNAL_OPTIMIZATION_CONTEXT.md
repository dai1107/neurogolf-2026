# External Optimization Context

This repository builds small ONNX models for the 400 NeuroGolf / ARC tasks.
The current canonical model bank is:

- `outputs/onnx/task001.onnx` through `outputs/onnx/task400.onnx`
- `outputs/submission.zip`
- `outputs/reports/current_model_bank_report.csv`

The old `archive` directory was only a baseline source used to promote better
models into `outputs/onnx`. It should not be used as a runtime or build
dependency.

## Current Validation State

Last full local model-bank validation:

```powershell
python -m src.build_current_model_submission --data-dir task --model-dir outputs\onnx --validated-dir outputs\current_model_bank_verified_onnx --report outputs\reports\current_model_bank_report.csv --zip outputs\submission.zip --timeout-seconds 120
```

Result:

- selected tasks: 400 / 400
- missing or invalid tasks: 0
- estimated cost total: 10530917
- ONNX file size total: 14815565 bytes
- zip artifact: `outputs/submission.zip`
- zip size: about 1466160 bytes

Submission structure check:

```powershell
python -m src.inspect_submission --zip outputs\submission.zip
```

Result:

- inspection passed
- num_models = 400

Targeted regression tests:

```powershell
python -m pytest -q tests\test_build_current_model_submission.py
```

Result:

- 3 passed

## Rules For Any Replacement Model

Any proposed replacement for `outputs/onnx/taskNNN.onnx` must satisfy all of
the following before it can be considered:

- pass `onnx.checker.check_model`
- have static input, output, and intermediate shapes
- contain no forbidden ops: `Loop`, `Scan`, `NonZero`, `Unique`, `Script`, `Function`
- run with default `onnxruntime`
- exactly match every train output grid for that task
- be deterministic, with no Python or external-file dependency at inference
- be smaller or lower cost than the current validated model, unless it fixes a correctness issue
- keep single-file size below 1.44 MB

After replacing any model, rerun:

```powershell
python -m src.build_current_model_submission --data-dir task --model-dir outputs\onnx --validated-dir outputs\current_model_bank_verified_onnx --report outputs\reports\current_model_bank_report.csv --zip outputs\submission.zip --timeout-seconds 120
python -m src.inspect_submission --zip outputs\submission.zip
```

## Highest-Cost Optimization Targets

Start with the highest estimated-cost tasks from
`outputs/reports/current_model_bank_report.csv`:

| task_id | estimated_cost | file_size_bytes |
| --- | ---: | ---: |
| task133 | 1406822 | 783434 |
| task084 | 1390970 | 1127799 |
| task209 | 1170350 | 1294052 |
| task076 | 1147810 | 948872 |
| task157 | 1023477 | 857911 |
| task200 | 990050 | 797905 |
| task233 | 668250 | 938774 |
| task025 | 332565 | 826752 |
| task367 | 295949 | 293653 |
| task366 | 266691 | 1269927 |
| task363 | 193941 | 170389 |
| task396 | 115080 | 148123 |
| task319 | 78739 | 247340 |
| task028 | 63050 | 51218 |
| task255 | 58680 | 106804 |
| task382 | 54738 | 65997 |
| task107 | 42039 | 31191 |
| task313 | 40857 | 39887 |
| task105 | 36364 | 35291 |
| task027 | 36154 | 31135 |

These tasks dominate total cost. A small number of symbolic replacements here
would improve the submission more than broad changes to already-small models.

## Recommended Optimization Workflow

1. Inspect one high-cost task JSON under `task/taskNNN.json`.
2. Infer a conservative symbolic rule that explains every train pair.
3. Build a static ONNX candidate using existing builders or a new narrow rule.
4. Validate the candidate with `src.evaluate_onnx_candidate`.
5. Compare cost and file size with `current_model_bank_report.csv`.
6. Replace `outputs/onnx/taskNNN.onnx` only when the candidate is valid and cheaper.
7. Rebuild and inspect the full 400-model submission.

Preferred model forms:

- 1x1 Conv for color remapping and channel mixing
- small Conv kernels for local neighborhood rules
- simple mask algebra with Add/Sub/Mul/Relu/Clip
- static Slice/Pad/Concat patterns for fixed geometric transforms

Avoid:

- large lookup tables or memorized train outputs
- dynamic shapes
- non-deterministic behavior
- adding dependencies such as PyTorch or TensorFlow
- using the removed archive directory as a hidden source of truth

## Important Caveat

The validation above is strict local train validation and submission structure
inspection. It is not a guaranteed official leaderboard score.
