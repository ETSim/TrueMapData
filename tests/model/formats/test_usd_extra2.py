"""More USD coverage for missing-pxr branches and convert_to_usdz failures."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tmd.model.base import ExportConfig, MeshData
from tmd.model.formats import usd as usd_mod


def test_export_returns_none_when_pxr_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usd_mod.importlib.util, "find_spec", lambda name: None)
    cfg = ExportConfig(x_length=1.0, y_length=1.0, z_scale=1.0, binary=False)
    out = usd_mod.USDExporter.export(np.linspace(0, 1, 25, dtype=np.float32).reshape(5, 5), str(tmp_path / "x.usda"), cfg)
    assert out is None


def test_export_mesh_to_usd_returns_none_when_pxr_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force the inner ``from pxr import ...`` to raise."""
    monkeypatch.setitem(sys.modules, "pxr", None)
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = MeshData(verts, faces)
    out = usd_mod.export_mesh_to_usd(mesh, str(tmp_path / "x.usda"), binary=False)
    assert out is None


def test_convert_to_usdz_returns_none_when_pxr_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pxr", None)
    fake = tmp_path / "scene.usdc"
    fake.write_bytes(b"hello")
    assert usd_mod.convert_to_usdz(str(fake)) is None


def test_convert_to_usdz_returns_none_when_input_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Even with pxr available, missing input file should short-circuit."""
    import types

    fake_pxr = types.ModuleType("pxr")

    class _UsdUtils:
        @staticmethod
        def CreateNewARKitUsdzPackage(_a, _b, _c):  # pragma: no cover
            raise AssertionError("should not be called")

    fake_pxr.UsdUtils = _UsdUtils
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    assert usd_mod.convert_to_usdz("/no/such/file/here.usdc") is None


def test_convert_to_usdz_handles_failure_return(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When ``CreateNewARKitUsdzPackage`` returns False we get None."""
    import types

    fake_pxr = types.ModuleType("pxr")

    class _UsdUtils:
        @staticmethod
        def CreateNewARKitUsdzPackage(_a, _b, _c):
            return False

    fake_pxr.UsdUtils = _UsdUtils
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    fake = tmp_path / "scene.usdc"
    fake.write_bytes(b"hello")
    assert usd_mod.convert_to_usdz(str(fake)) is None


def test_convert_to_usdz_handles_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Exceptions inside the conversion should resolve to ``None``."""
    import types

    fake_pxr = types.ModuleType("pxr")

    class _UsdUtils:
        @staticmethod
        def CreateNewARKitUsdzPackage(_a, _b, _c):
            raise RuntimeError("boom")

    fake_pxr.UsdUtils = _UsdUtils
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)
    fake = tmp_path / "scene.usdc"
    fake.write_bytes(b"hello")
    assert usd_mod.convert_to_usdz(str(fake)) is None


def test_generate_usd_texture_handles_missing_pillow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Force the inner ``from PIL import Image`` to ImportError."""
    monkeypatch.setitem(sys.modules, "PIL", None)
    out = tmp_path / "tex.png"
    hm = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    result = usd_mod._generate_usd_texture(hm, str(out), color_map="terrain", resolution=None)
    assert result is None
