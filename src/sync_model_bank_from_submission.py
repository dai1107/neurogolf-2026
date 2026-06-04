"""Synchronize a local ONNX model bank from a validated submission zip."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .arc_io import load_all_tasks
from .inspect_submission import TASK_ONNX_RE, inspect_submission


def _safe_remove_tree(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"refusing to remove path outside allowed root: {path}")
    if path.exists():
        shutil.rmtree(path)


def sync_model_bank_from_submission(
    zip_path: str,
    data_dir: str,
    model_dir: str,
    require_all_tasks: bool = True,
) -> dict[str, Any]:
    """Extract validated task ONNX files into ``model_dir``.

    The zip is inspected before any model-bank file is replaced. Extraction is
    staged in a sibling temporary directory so a failed zip never leaves a
    partially synchronized model bank.
    """
    submission_path = Path(zip_path)
    model_root = Path(model_dir)
    task_ids = set(load_all_tasks(data_dir))
    inspect_submission(str(submission_path))

    with zipfile.ZipFile(submission_path, "r") as archive:
        names = sorted(archive.namelist())
        zip_task_ids = {name[:-5] for name in names if TASK_ONNX_RE.match(name)}
        missing = sorted(task_ids - zip_task_ids)
        extra = sorted(zip_task_ids - task_ids)
        if require_all_tasks and (missing or extra):
            raise ValueError(
                "submission task set does not match data_dir: "
                f"missing={missing[:10]} extra={extra[:10]}"
            )

        temp_root = model_root.parent / f".{model_root.name}_sync_tmp"
        allowed_root = model_root.parent
        _safe_remove_tree(temp_root, allowed_root)
        temp_root.mkdir(parents=True, exist_ok=False)
        try:
            for name in names:
                if not TASK_ONNX_RE.match(name):
                    continue
                destination = temp_root / name
                destination.write_bytes(archive.read(name))

            model_root.mkdir(parents=True, exist_ok=True)
            for stale_model in model_root.glob("task*.onnx"):
                stale_model.unlink()
            copied = 0
            for staged_model in sorted(temp_root.glob("task*.onnx")):
                shutil.copyfile(staged_model, model_root / staged_model.name)
                copied += 1
        finally:
            _safe_remove_tree(temp_root, allowed_root)

    summary = {
        "zip_path": str(submission_path),
        "model_dir": str(model_root),
        "copied_models": copied,
        "require_all_tasks": require_all_tasks,
        "missing_tasks": missing,
        "extra_tasks": extra,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", default="outputs/submission.zip")
    parser.add_argument("--data-dir", default="task")
    parser.add_argument("--model-dir", default="outputs/onnx")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    sync_model_bank_from_submission(
        zip_path=args.zip_path,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        require_all_tasks=not args.allow_partial,
    )


if __name__ == "__main__":
    main()
