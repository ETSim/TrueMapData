"""Coverage for ``tmd.cli.commands.maps`` export and batch paths."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tmd.cli.commands import maps as maps_cmd


class _FakeProgInner:
    def add_task(self, *a, **k):
        return 0

    def update(self, *a, **k):
        pass


class _FakeProgress:
    def __enter__(self):
        return _FakeProgInner()

    def __exit__(self, *a, **k):
        return False


def test_export_map_fails_without_height_map(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Loaded:
        height_map = None
        metadata = {}

    monkeypatch.setattr(maps_cmd.TMD, "load", classmethod(lambda cls, p: _Loaded()))
    errs: list[str] = []

    monkeypatch.setattr(maps_cmd, "print_error", lambda m: errs.append(m))
    monkeypatch.setattr(maps_cmd, "console", SimpleNamespace(print=lambda *a, **k: None))
    inp = tmp_path / "in.tmd"
    inp.write_bytes(b"x")
    out = tmp_path / "o.png"
    assert maps_cmd.export_map("normal", inp, out) is False
    assert errs and "height map" in errs[0].lower()


def test_export_all_maps_missing_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[str] = []

    monkeypatch.setattr(maps_cmd.console, "print", lambda m: printed.append(str(m)))
    maps_cmd.export_all_maps(tmp_path / "nope.tmd", output_dir=tmp_path / "out")


def test_batch_export_maps_missing_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[str] = []

    monkeypatch.setattr(maps_cmd.console, "print", lambda m: printed.append(str(m)))
    maps_cmd.batch_export_maps(tmp_path / "nonexistent_dir_xyz")


def test_batch_export_maps_empty_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[str] = []

    monkeypatch.setattr(maps_cmd.console, "print", lambda m: printed.append(str(m)))
    maps_cmd.batch_export_maps(tmp_path, output_dir=tmp_path / "exp")


def test_export_map_success_mocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hm = np.ones((3, 3), dtype=np.float32)

    class _Loaded:
        height_map = hm
        metadata = {"x_length": 1.0, "y_length": 1.0}

    monkeypatch.setattr(maps_cmd.TMD, "load", classmethod(lambda cls, p: _Loaded()))
    monkeypatch.setattr(maps_cmd, "console", SimpleNamespace(print=lambda *a, **k: None))
    monkeypatch.setattr(maps_cmd, "Panel", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(maps_cmd, "display_tmd_info", lambda *a, **k: None)
    monkeypatch.setattr(maps_cmd, "Progress", lambda: _FakeProgress())
    monkeypatch.setitem(maps_cmd.export_funcs, "normal", lambda *a, **k: None)
    monkeypatch.setattr(maps_cmd, "print_success", lambda *a, **k: None)

    inp = tmp_path / "in.tmd"
    inp.write_bytes(b"x")
    out = tmp_path / "sub" / "n.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    assert maps_cmd.export_map("normal", inp, out, compress=0, format="png") is True
