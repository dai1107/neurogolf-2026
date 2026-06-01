from __future__ import annotations

import json

import pytest

from src.arc_io import load_all_tasks, load_task


def _write_json(path, data) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def test_load_task_accepts_valid_task(tmp_path) -> None:
    path = tmp_path / "task001.json"
    _write_json(
        path,
        {
            "train": [
                {
                    "input": [[0, 1], [2, 3]],
                    "output": [[0, 1], [2, 3]],
                }
            ]
        },
    )

    task = load_task(str(path))

    assert task["train"][0]["input"] == [[0, 1], [2, 3]]


def test_load_task_rejects_non_rectangular_grid(tmp_path) -> None:
    path = tmp_path / "task001.json"
    _write_json(path, {"train": [{"input": [[0], [1, 2]], "output": [[0], [1]]}]})

    with pytest.raises(ValueError, match="rectangular"):
        load_task(str(path))


def test_load_task_rejects_color_out_of_range(tmp_path) -> None:
    path = tmp_path / "task001.json"
    _write_json(path, {"train": [{"input": [[10]], "output": [[0]]}]})

    with pytest.raises(ValueError, match="outside 0..9"):
        load_task(str(path))


def test_load_task_rejects_missing_train_input_or_output(tmp_path) -> None:
    missing_train = tmp_path / "task001.json"
    _write_json(missing_train, {"test": []})
    with pytest.raises(ValueError, match="train"):
        load_task(str(missing_train))

    missing_output = tmp_path / "task002.json"
    _write_json(missing_output, {"train": [{"input": [[0]]}]})
    with pytest.raises(ValueError, match="missing output"):
        load_task(str(missing_output))


def test_load_all_tasks_uses_explicit_task_ids(tmp_path) -> None:
    _write_json(tmp_path / "task002.json", {"train": [{"input": [[0]], "output": [[0]]}]})
    _write_json(tmp_path / "task001.json", {"train": [{"input": [[1]], "output": [[1]]}]})

    tasks = load_all_tasks(str(tmp_path))

    assert list(tasks) == ["task001", "task002"]


def test_load_all_tasks_rejects_unmapped_hash_names(tmp_path) -> None:
    _write_json(tmp_path / "abcdef.json", {"train": [{"input": [[0]], "output": [[0]]}]})

    with pytest.raises(ValueError, match="official mapping"):
        load_all_tasks(str(tmp_path))
