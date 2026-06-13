"""Inspect a submission.zip for allowed filenames and ONNX constraints."""

from __future__ import annotations

import argparse
import re
import tempfile
import zipfile
from pathlib import Path

import onnx

from .cost_estimator import FILE_SIZE_LIMIT_BYTES, check_forbidden_ops


TASK_ONNX_RE = re.compile(r"^task(\d{3})\.onnx$")
HYBRID_STACK_DIRS = ("base_submission", "overrides")
HYBRID_STACK_RE = re.compile(r"^(base_submission|overrides)/(task(\d{3})\.onnx)$")


def _task_number_from_flat_name(name: str) -> int:
    match = TASK_ONNX_RE.match(name)
    if match is None:
        raise ValueError(f"invalid zip entry name: {name}")
    task_number = int(match.group(1))
    if task_number < 1 or task_number > 400:
        raise ValueError(f"task number out of range in zip entry: {name}")
    return task_number


def _detect_layout(names: list[str], requested_layout: str) -> str:
    if requested_layout != "auto":
        return requested_layout
    if all(TASK_ONNX_RE.match(name) for name in names):
        return "flat"
    if all(HYBRID_STACK_RE.match(name) for name in names):
        return "hybrid_stack"
    return "unknown"


def _check_onnx_entry(archive: zipfile.ZipFile, name: str) -> None:
    info = archive.getinfo(name)
    if info.file_size > FILE_SIZE_LIMIT_BYTES:
        raise ValueError(f"{name} exceeds file size limit: {info.file_size}")

    data = archive.read(name)
    model = onnx.load_model_from_string(data)
    onnx.checker.check_model(model)
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as handle:
        temp_path = Path(handle.name)
        handle.write(data)
    try:
        forbidden = check_forbidden_ops(str(temp_path))
        if not forbidden["passed"]:
            raise ValueError(f"{name} contains forbidden ops: {forbidden['forbidden_ops_found']}")
    finally:
        temp_path.unlink(missing_ok=True)


def _inspect_flat_submission(archive: zipfile.ZipFile, names: list[str]) -> dict[str, int | bool]:
    seen: set[str] = set()
    for name in names:
        if "/" in name or "\\" in name:
            raise ValueError(f"flat zip entry must not contain a directory: {name}")
        _task_number_from_flat_name(name)
        if name in seen:
            raise ValueError(f"duplicate zip entry: {name}")
        seen.add(name)
        _check_onnx_entry(archive, name)

    print("submission inspection passed")
    print(f"num_models = {len(seen)}")
    return {"passed": True, "num_models": len(seen)}


def _inspect_hybrid_stack_submission(
    archive: zipfile.ZipFile,
    names: list[str],
) -> dict[str, int | bool | str]:
    seen: set[str] = set()
    task_ids_by_dir: dict[str, set[str]] = {folder: set() for folder in HYBRID_STACK_DIRS}
    for name in names:
        if "\\" in name:
            raise ValueError(f"zip entry must use forward slashes only: {name}")
        match = HYBRID_STACK_RE.match(name)
        if match is None:
            raise ValueError(f"invalid hybrid stack zip entry name: {name}")
        folder, task_name, task_number_text = match.groups()
        task_number = int(task_number_text)
        if task_number < 1 or task_number > 400:
            raise ValueError(f"task number out of range in zip entry: {name}")
        if name in seen:
            raise ValueError(f"duplicate zip entry: {name}")
        seen.add(name)
        task_ids_by_dir[folder].add(task_name[:-5])
        _check_onnx_entry(archive, name)

    task_sets = list(task_ids_by_dir.values())
    if task_sets and any(task_set != task_sets[0] for task_set in task_sets[1:]):
        missing_by_dir = {
            folder: sorted(set.union(*task_sets) - task_ids)
            for folder, task_ids in task_ids_by_dir.items()
        }
        raise ValueError(f"hybrid stack task sets differ by directory: {missing_by_dir}")

    print("submission inspection passed")
    print("layout = hybrid_stack")
    print(f"num_models = {len(seen)}")
    print(f"num_task_ids = {len(task_sets[0]) if task_sets else 0}")
    return {
        "passed": True,
        "num_models": len(seen),
        "layout": "hybrid_stack",
        "num_task_ids": len(task_sets[0]) if task_sets else 0,
    }


def inspect_submission(
    zip_path: str,
    allow_empty: bool = False,
    layout: str = "auto",
) -> dict[str, int | bool | str]:
    """Validate zip structure and embedded ONNX files."""
    if layout not in {"auto", "flat", "hybrid_stack"}:
        raise ValueError(f"layout must be auto, flat, or hybrid_stack: {layout}")
    path = Path(zip_path)
    if not path.is_file():
        raise FileNotFoundError(f"submission zip does not exist: {zip_path}")

    with zipfile.ZipFile(path, "r") as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        directory_entries = [name for name in archive.namelist() if name.endswith("/")]
        if directory_entries:
            raise ValueError(f"zip must not contain directory entries: {directory_entries[:5]}")
        if not names and not allow_empty:
            raise ValueError("submission zip is empty")
        selected_layout = _detect_layout(names, layout)
        if selected_layout == "flat":
            return _inspect_flat_submission(archive, names)
        if selected_layout == "hybrid_stack":
            return _inspect_hybrid_stack_submission(archive, names)
        raise ValueError("submission zip layout is neither flat taskNNN.onnx nor hybrid stack")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", default="outputs/submission.zip")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--layout", choices=["auto", "flat", "hybrid_stack"], default="auto")
    args = parser.parse_args()
    inspect_submission(args.zip_path, allow_empty=args.allow_empty, layout=args.layout)


if __name__ == "__main__":
    main()
