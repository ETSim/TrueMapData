"""Tests for ``tmd.cli.commands.batch.BatchProcessor``."""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from tmd.cli.commands import batch as batch_mod


def test_batch_processor_processes_files_without_rich(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.tmd").write_bytes(b"x")
    (data_dir / "b.tmd").write_bytes(b"y")

    monkeypatch.setattr(batch_mod, "HAS_RICH", False)

    names: list[str] = []

    def process_one(path: Path) -> bool:
        names.append(path.name)
        return True

    processor = batch_mod.BatchProcessor(data_dir, recursive=False)
    result = processor.process_files(process_one, description="Unit batch")

    assert result["total"] == 2
    assert result["success"] == 2
    assert result["failed"] == 0
    assert sorted(names) == ["a.tmd", "b.tmd"]


def test_batch_processor_no_matching_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    warnings: list[str] = []
    monkeypatch.setattr(batch_mod, "print_warning", lambda msg: warnings.append(msg))

    processor = batch_mod.BatchProcessor(empty, recursive=False, pattern="*.tmd")
    result = processor.process_files(lambda p: True)

    assert result["total"] == 0
    assert result["success"] == 0
    assert result["files"] == []
    assert warnings and "*.tmd" in warnings[0]


def test_batch_processor_find_files_recursive(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    sub = root / "nested"
    sub.mkdir(parents=True)
    (root / "top.tmd").write_bytes(b"a")
    (sub / "deep.tmd").write_bytes(b"b")

    flat = batch_mod.BatchProcessor(root, recursive=False)
    assert sorted(p.name for p in flat.find_files()) == ["top.tmd"]

    deep = batch_mod.BatchProcessor(root, recursive=True)
    assert sorted(p.name for p in deep.find_files()) == ["deep.tmd", "top.tmd"]


def test_batch_processor_process_files_with_rich_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class DummyProgress:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def add_task(self, *args, **kwargs):
            return 0

        def update(self, *args, **kwargs):
            pass

        def advance(self, *args, **kwargs):
            pass

    data_dir = tmp_path / "rich_batch"
    data_dir.mkdir()
    (data_dir / "only.tmd").write_bytes(b"x")

    monkeypatch.setattr(batch_mod, "HAS_RICH", True)
    monkeypatch.setattr("rich.progress.Progress", DummyProgress)
    monkeypatch.setattr(batch_mod, "print_success", lambda *a, **k: None)

    processor = batch_mod.BatchProcessor(data_dir, recursive=False)
    result = processor.process_files(lambda p: True, description="Rich path")

    assert result["total"] == 1
    assert result["success"] == 1
    assert result["failed"] == 0


def test_batch_processor_process_func_raises_without_rich(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir = tmp_path / "err_plain"
    data_dir.mkdir()
    (data_dir / "boom.tmd").write_bytes(b"x")
    monkeypatch.setattr(batch_mod, "HAS_RICH", False)
    printed: list[str] = []

    def fake_print(*args, **kwargs) -> None:
        printed.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(builtins, "print", fake_print)
    monkeypatch.setattr(batch_mod, "print_success", lambda *a, **k: None)

    def process_one(path: Path) -> bool:
        raise RuntimeError("no")

    processor = batch_mod.BatchProcessor(data_dir, recursive=False)
    result = processor.process_files(process_one)

    assert result["total"] == 1
    assert result["success"] == 0
    assert result["failed"] == 1
    blob = "\n".join(printed)
    assert "boom.tmd" in blob and "Error" in blob


def test_batch_processor_process_func_raises_with_rich_progress(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class DummyProgress:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def add_task(self, *args, **kwargs):
            return 0

        def update(self, *args, **kwargs):
            pass

        def advance(self, *args, **kwargs):
            pass

    data_dir = tmp_path / "err_rich"
    data_dir.mkdir()
    (data_dir / "bad.tmd").write_bytes(b"x")
    monkeypatch.setattr(batch_mod, "HAS_RICH", True)
    monkeypatch.setattr("rich.progress.Progress", DummyProgress)
    monkeypatch.setattr(batch_mod, "print_success", lambda *a, **k: None)
    errors: list[str] = []
    monkeypatch.setattr(batch_mod, "print_error", lambda msg: errors.append(msg))

    def boom(_path: Path) -> bool:
        raise ValueError("x")

    processor = batch_mod.BatchProcessor(data_dir, recursive=False)
    result = processor.process_files(boom, description="x")

    assert result["failed"] == 1
    assert errors
