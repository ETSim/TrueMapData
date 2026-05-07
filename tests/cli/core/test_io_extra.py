"""Extra coverage for tmd.cli.core.io."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import pytest

from tmd.cli.core import io as io_mod


@pytest.fixture(autouse=True)
def reroute_load_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Make ``load_config`` deterministic and writes safe under tmp_path."""

    def _fake_load_config():
        return {"output_dir": str(tmp_path / "outputs"), "image_format": "png", "auto_open": True}

    monkeypatch.setattr(io_mod, "load_config", _fake_load_config)
    yield


def test_create_output_dir_default_uses_config(tmp_path: Path) -> None:
    out = io_mod.create_output_dir()
    assert out.exists()
    assert out == tmp_path / "outputs"


def test_create_output_dir_with_subdir(tmp_path: Path) -> None:
    out = io_mod.create_output_dir(subdir="vis")
    assert out.exists()
    assert out.name == "vis"


def test_create_output_dir_existing_dir_does_not_recreate(tmp_path: Path) -> None:
    target = tmp_path / "preexisting"
    target.mkdir()
    out = io_mod.create_output_dir(base_dir=str(target))
    assert out == target


def test_get_file_extension_for_plotly() -> None:
    assert io_mod.get_file_extension("plotly") == ".html"


def test_get_file_extension_uses_config() -> None:
    assert io_mod.get_file_extension("matplotlib") == ".png"


def test_get_output_filename_with_explicit_output(tmp_path: Path) -> None:
    out = tmp_path / "explicit.png"
    res = io_mod.get_output_filename(tmp_path / "fixture.tmd", "matplotlib", "2d", output=out)
    assert res == out


def test_get_output_filename_generates_default(tmp_path: Path) -> None:
    res = io_mod.get_output_filename(tmp_path / "fixture.tmd", "matplotlib", "2d", subdir="visualizations")
    assert res.suffix == ".png"
    assert "fixture_2d_matplotlib" in res.name


def test_load_tmd_file_uses_cache_when_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_height = np.zeros((3, 3), dtype=np.float32)
    fake_meta = {"x_length": 1.0}

    class _FakeCaching:
        def get_cached_tmd_data(self, _path):
            return fake_height, fake_meta

        def cache_tmd_data(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(io_mod, "_get_caching_module", lambda: _FakeCaching())

    fake_path = tmp_path / "fixture.tmd"
    fake_path.write_bytes(b"")
    out = io_mod.load_tmd_file(fake_path, with_console_status=False, use_cache=True)
    assert out is not None


def test_load_tmd_file_falls_back_to_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_tmd_path: Path
) -> None:
    """When cache miss, the function still loads via TMD.load."""

    class _NoCacheCaching:
        def get_cached_tmd_data(self, _path):
            return None

        def cache_tmd_data(self, *_args, **_kwargs):
            return True

    monkeypatch.setattr(io_mod, "_get_caching_module", lambda: _NoCacheCaching())
    out = io_mod.load_tmd_file(tmp_tmd_path, with_console_status=False, use_cache=True)
    assert out is not None


def test_load_tmd_file_with_console_status(
    monkeypatch: pytest.MonkeyPatch, tmp_tmd_path: Path
) -> None:
    out = io_mod.load_tmd_file(tmp_tmd_path, with_console_status=True, use_cache=False)
    assert out is not None


def test_load_tmd_file_handles_load_failure_with_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bogus = tmp_path / "missing.tmd"
    bogus.write_bytes(b"junk")

    import tmd

    def _boom(path):
        raise OSError("simulated load failure")

    monkeypatch.setattr(tmd.TMD, "load", staticmethod(_boom))
    out = io_mod.load_tmd_file(bogus, with_console_status=True, use_cache=False)
    assert out is None


def test_load_tmd_file_handles_load_failure_raises_without_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bogus = tmp_path / "missing.tmd"
    bogus.write_bytes(b"junk")
    import tmd

    def _boom(path):
        raise OSError("simulated load failure")

    monkeypatch.setattr(tmd.TMD, "load", staticmethod(_boom))
    with pytest.raises(io_mod.FileError):
        io_mod.load_tmd_file(bogus, with_console_status=False, use_cache=False)


def test_find_files_by_pattern_with_matches(tmp_path: Path) -> None:
    for name in ("a.tmd", "b.tmd"):
        (tmp_path / name).write_bytes(b"")
    found = io_mod.find_files_by_pattern(tmp_path, pattern="*.tmd", recursive=False)
    assert len(found) == 2


def test_find_files_by_pattern_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "deep"
    sub.mkdir()
    (sub / "nested.tmd").write_bytes(b"")
    found = io_mod.find_files_by_pattern(tmp_path, pattern="*.tmd", recursive=True)
    assert any(p.name == "nested.tmd" for p in found)


def test_find_files_by_pattern_no_matches(tmp_path: Path) -> None:
    found = io_mod.find_files_by_pattern(tmp_path, pattern="*.notreal")
    assert found == []


def test_auto_open_file_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(io_mod, "load_config", lambda: {"auto_open": False})
    f = tmp_path / "x.png"
    f.write_bytes(b"")
    io_mod.auto_open_file(f)


def test_auto_open_file_handles_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(io_mod, "load_config", lambda: {"auto_open": True})

    def _boom(path):
        raise RuntimeError("can't open")

    monkeypatch.setattr(io_mod, "_open_file", _boom)
    f = tmp_path / "x.png"
    f.write_bytes(b"")
    io_mod.auto_open_file(f)
