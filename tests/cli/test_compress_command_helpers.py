"""Tests for helpers in ``tmd.cli.commands.compress``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tmd.cli.core as cli_core
from tmd.cli.commands import compress as compress_mod


def test_display_file_info_command_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tmd_path = tmp_path / "sample.tmd"
    tmd_path.write_bytes(b"x")

    class FakeData:
        def metadata(self):
            return {"unit": True}

        def height_map(self):
            return np.linspace(0, 1, 12, dtype=np.float64).reshape(3, 4)

    monkeypatch.setattr(compress_mod, "load_tmd_file", lambda *a, **k: FakeData())
    monkeypatch.setattr(cli_core, "display_metadata", lambda m: None)
    monkeypatch.setattr(compress_mod.console, "print", lambda *a, **k: None)

    assert compress_mod.display_file_info_command(tmd_path, show_sample=False) is True


def test_display_file_info_command_with_sample(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    tmd_path = tmp_path / "sample2.tmd"
    tmd_path.write_bytes(b"y")

    class FakeData:
        def metadata(self):
            return {}

        def height_map(self):
            return np.ones((6, 6), dtype=np.float32)

    monkeypatch.setattr(compress_mod, "load_tmd_file", lambda *a, **k: FakeData())
    monkeypatch.setattr(cli_core, "display_metadata", lambda m: None)
    monkeypatch.setattr(compress_mod.console, "print", lambda *a, **k: None)

    assert compress_mod.display_file_info_command(tmd_path, show_sample=True) is True


def test_compress_tmd_command_invalid_scale_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compress_mod, "print_error", lambda *a, **k: None)
    assert compress_mod.compress_tmd_command(Path("any.tmd"), mode="downsample", scale=1.0) is False


def test_compress_tmd_command_invalid_levels_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compress_mod, "print_error", lambda *a, **k: None)
    assert compress_mod.compress_tmd_command(Path("any.tmd"), mode="quantize", levels=1) is False


def test_compress_tmd_command_load_failure_returns_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tmd_path = tmp_path / "empty.tmd"
    tmd_path.write_bytes(b"z")
    monkeypatch.setattr(compress_mod, "load_tmd_file", lambda *a, **k: None)
    assert compress_mod.compress_tmd_command(tmd_path, mode="downsample", scale=0.5) is False


def test_compress_batch_command_delegates_to_compress_tmd_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tmd.cli.commands import batch as batch_mod

    data_dir = tmp_path / "batch_in"
    data_dir.mkdir()
    (data_dir / "one.tmd").write_bytes(b"1")

    monkeypatch.setattr(batch_mod, "HAS_RICH", False)

    calls: list = []

    def fake_compress_tmd(*, tmd_file: Path, **kwargs):
        calls.append({"tmd_file": tmd_file, **kwargs})
        return True

    monkeypatch.setattr(compress_mod, "compress_tmd_command", fake_compress_tmd)

    assert compress_mod.compress_batch_command(data_dir, mode="quantize", levels=128) is True
    assert len(calls) == 1
    assert calls[0]["tmd_file"] == data_dir / "one.tmd"
    assert calls[0]["mode"] == "quantize"
    assert calls[0]["levels"] == 128
    assert calls[0]["method"] == "bilinear"


def test_compress_batch_command_returns_false_when_any_file_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tmd.cli.commands import batch as batch_mod

    data_dir = tmp_path / "batch_fail"
    data_dir.mkdir()
    (data_dir / "ok.tmd").write_bytes(b"1")
    (data_dir / "bad.tmd").write_bytes(b"2")

    monkeypatch.setattr(batch_mod, "HAS_RICH", False)

    def fake_compress_tmd(*, tmd_file: Path, **kwargs):
        return tmd_file.name != "bad.tmd"

    monkeypatch.setattr(compress_mod, "compress_tmd_command", fake_compress_tmd)

    assert compress_mod.compress_batch_command(data_dir) is False


def test_compress_tmd_command_downsample_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise happy path: downsample, default output name, panel print."""
    tmd_path = tmp_path / "src.tmd"
    tmd_path.write_bytes(b"x" * 200)

    out_dir = tmp_path / "compressed"
    out_dir.mkdir()

    class FakeData:
        def height_map(self) -> np.ndarray:
            return np.ones((10, 10), dtype=np.float32)

        def metadata(self) -> dict:
            return {"x_length": 5.0, "y_length": 5.0, "x_offset": 0.0, "y_offset": 0.0}

    class DummyStatus:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(compress_mod, "load_tmd_file", lambda *a, **k: FakeData())
    monkeypatch.setattr(compress_mod, "create_output_dir", lambda *a, **k: out_dir)

    def fake_downsample(h: np.ndarray, nw: int, nh: int, method: str) -> np.ndarray:
        return np.full((nh, nw), 0.25, dtype=np.float32)

    monkeypatch.setattr(compress_mod.TMDUtils, "downsample_array", staticmethod(fake_downsample))

    written: list[Path] = []

    def fake_write_tmd(*args, **kwargs) -> None:
        p = Path(args[1] if len(args) > 1 else kwargs["output_path"])
        written.append(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"out")

    monkeypatch.setattr(compress_mod.TMDUtils, "write_tmd_file", staticmethod(fake_write_tmd))
    monkeypatch.setattr(compress_mod.console, "status", lambda *a, **k: DummyStatus())
    monkeypatch.setattr(compress_mod.console, "print", lambda *a, **k: None)

    assert compress_mod.compress_tmd_command(tmd_path, mode="downsample", scale=0.5) is True
    assert len(written) == 1
    assert written[0].suffix == ".tmd"
    assert "_ds50" in written[0].name


def test_compress_tmd_command_both_mode_uses_quantize(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tmd_path = tmp_path / "both.tmd"
    tmd_path.write_bytes(b"y" * 200)
    out_dir = tmp_path / "c2"
    out_dir.mkdir()

    class FakeData:
        def height_map(self) -> np.ndarray:
            return np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)

        def metadata(self) -> dict:
            return {"x_length": 1.0, "y_length": 1.0}

    class DummyStatus:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(compress_mod, "load_tmd_file", lambda *a, **k: FakeData())
    monkeypatch.setattr(compress_mod, "create_output_dir", lambda *a, **k: out_dir)
    monkeypatch.setattr(
        compress_mod.TMDUtils,
        "downsample_array",
        staticmethod(lambda h, nw, nh, m: np.ones((nh, nw), dtype=np.float32) * 0.5),
    )
    monkeypatch.setattr(
        compress_mod.TMDUtils,
        "quantize_array",
        staticmethod(lambda h, levels: np.clip(h, 0, 1)),
    )

    def fake_write(*args, **kwargs) -> None:
        p = Path(args[1] if len(args) > 1 else kwargs["output_path"])
        p.write_bytes(b"z")

    monkeypatch.setattr(compress_mod.TMDUtils, "write_tmd_file", staticmethod(fake_write))
    monkeypatch.setattr(compress_mod.console, "status", lambda *a, **k: DummyStatus())
    monkeypatch.setattr(compress_mod.console, "print", lambda *a, **k: None)

    assert compress_mod.compress_tmd_command(tmd_path, mode="both", scale=0.5, levels=8) is True
