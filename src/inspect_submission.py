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


def inspect_submission(zip_path: str, allow_empty: bool = False) -> dict[str, int | bool]:
    """Validate zip structure and embedded ONNX files."""
    path = Path(zip_path)
    if not path.is_file():
        raise FileNotFoundError(f"submission zip does not exist: {zip_path}")

    seen: set[str] = set()
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if not names and not allow_empty:
            raise ValueError("submission zip is empty")
        for name in names:
            if name.endswith("/") or "/" in name or "\\" in name:
                raise ValueError(f"zip entry must be a flat file: {name}")
            match = TASK_ONNX_RE.match(name)
            if match is None:
                raise ValueError(f"invalid zip entry name: {name}")
            task_number = int(match.group(1))
            if task_number < 1 or task_number > 400:
                raise ValueError(f"task number out of range in zip entry: {name}")
            if name in seen:
                raise ValueError(f"duplicate zip entry: {name}")
            seen.add(name)

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

    print("submission inspection passed")
    print(f"num_models = {len(seen)}")
    return {"passed": True, "num_models": len(seen)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", default="outputs/submission.zip")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()
    inspect_submission(args.zip_path, allow_empty=args.allow_empty)


if __name__ == "__main__":
    main()
