"""Extra coverage for tmd.cli.compression package init."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from tmd.cli import compression as comp


def test_compress_height_map_downsample_nearest() -> None:
    hm = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = comp.compress_height_map(hm, mode="downsample", scale=0.5, method="nearest")
    assert out.shape == (2, 2)


def test_compress_height_map_quantize_flat_logs_warning() -> None:
    hm = np.full((3, 3), 5.0, dtype=np.float32)
    out = comp.compress_height_map(hm, mode="quantize", levels=4)
    np.testing.assert_array_equal(out, hm)


def test_compress_height_map_returns_original_if_scipy_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If scipy import fails, falls back to original height map."""
    import builtins as _builtins

    real_import = _builtins.__import__

    def _failing_import(name: str, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("simulated missing scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_builtins, "__import__", _failing_import)
    hm = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = comp.compress_height_map(hm, mode="downsample", scale=0.5)
    assert out.shape == hm.shape


class _FakeTMDObject:
    """Mimic the .height_map() / .metadata() / .save() surface used in the loader."""

    def __init__(self, height_map: np.ndarray, metadata: Dict[str, Any]) -> None:
        self._height_map = height_map
        self._metadata = metadata
        self.save_calls: list[tuple[str, int]] = []

    def height_map(self) -> np.ndarray:
        return self._height_map

    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def save(self, path: str, version: int = 2) -> None:
        Path(path).write_bytes(b"x" * 32)
        self.save_calls.append((path, version))


def test_compress_tmd_file_invalid_paths(tmp_path: Path) -> None:
    res = comp.compress_tmd_file(tmp_path / "missing.tmd")
    assert res["success"] is False
    assert "not found" in res["error"]


def test_compress_tmd_file_invalid_scale(tmp_path: Path) -> None:
    f = tmp_path / "x.tmd"
    f.write_bytes(b"data")
    res = comp.compress_tmd_file(f, scale=0.0)
    assert res["success"] is False


def test_compress_tmd_file_invalid_levels(tmp_path: Path) -> None:
    f = tmp_path / "x.tmd"
    f.write_bytes(b"data")
    res = comp.compress_tmd_file(f, levels=1)
    assert res["success"] is False


def test_compress_tmd_file_with_mocked_tmd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src.tmd"
    src.write_bytes(b"x" * 1024)
    out = tmp_path / "out.tmd"

    hm = np.linspace(0.0, 1.0, 256, dtype=np.float64).reshape(16, 16)
    fake_calls = {"args": None}

    class _FakeTMDClass:
        def __init__(self, *args, **kwargs) -> None:
            fake_calls["args"] = (args, kwargs)
            if len(args) == 1 and isinstance(args[0], str):
                self._height_map = hm
                self._metadata = {"x_length": 1.0, "y_length": 1.0}
            else:
                self._height_map = args[0]
                self._metadata = args[1]

        def height_map(self) -> np.ndarray:
            return self._height_map

        def metadata(self) -> Dict[str, Any]:
            return self._metadata

        def save(self, path: str, version: int = 2) -> None:
            Path(path).write_bytes(b"x" * 256)

    monkeypatch.setattr(comp, "TMD", _FakeTMDClass)

    res = comp.compress_tmd_file(src, output=out, mode="both", scale=0.5, levels=8)
    assert res["success"] is True
    assert res["output_file"] == str(out)
    assert "compressed_dimensions" in res


def test_compress_tmd_file_autogenerates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "fixture.tmd"
    src.write_bytes(b"y" * 2048)

    hm = np.linspace(0.0, 1.0, 64, dtype=np.float64).reshape(8, 8)

    class _FakeTMDClass:
        def __init__(self, *args, **kwargs) -> None:
            self._hm = hm
            self._md = {"x_length": 1.0}

        def height_map(self) -> np.ndarray:
            return self._hm

        def metadata(self) -> Dict[str, Any]:
            return self._md

        def save(self, path: str, version: int = 2) -> None:
            Path(path).write_bytes(b"k" * 128)

    monkeypatch.setattr(comp, "TMD", _FakeTMDClass)

    res = comp.compress_tmd_file(src, mode="downsample", scale=0.5)
    assert res["success"] is True
    assert "ds50" in res["output_file"]


def test_compress_tmd_file_handles_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "fixture.tmd"
    src.write_bytes(b"y" * 2048)

    class _Boom:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("simulated load failure")

    monkeypatch.setattr(comp, "TMD", _Boom)
    res = comp.compress_tmd_file(src, output=tmp_path / "o.tmd")
    assert res["success"] is False
    assert "simulated" in res["error"]


def test_display_compression_summary_failure() -> None:
    comp.display_compression_summary({"success": False, "error": "bad"})


def test_display_compression_summary_success_downsample(monkeypatch: pytest.MonkeyPatch) -> None:
    summary = {
        "success": True,
        "input_file": "in.tmd",
        "output_file": "out.tmd",
        "mode": "downsample",
        "original_dimensions": "16x16",
        "compressed_dimensions": "8x8",
        "original_size": 1024,
        "compressed_size": 256,
        "size_reduction": 0.75,
        "scale": 0.5,
        "method": "bilinear",
    }
    comp.display_compression_summary(summary)


def test_display_compression_summary_quantize() -> None:
    summary = {
        "success": True,
        "input_file": "in.tmd",
        "output_file": "out.tmd",
        "mode": "quantize",
        "original_dimensions": "16x16",
        "compressed_dimensions": "16x16",
        "original_size": 1024,
        "compressed_size": 200,
        "size_reduction": 0.8,
        "levels": 64,
    }
    comp.display_compression_summary(summary)


def test_display_compression_summary_both() -> None:
    summary = {
        "success": True,
        "input_file": "in.tmd",
        "output_file": "out.tmd",
        "mode": "both",
        "original_dimensions": "16x16",
        "compressed_dimensions": "8x8",
        "original_size": 1024,
        "compressed_size": 100,
        "size_reduction": 0.9,
        "scale": 0.5,
        "method": "nearest",
        "levels": 32,
    }
    comp.display_compression_summary(summary)
