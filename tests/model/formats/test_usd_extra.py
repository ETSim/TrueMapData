"""USD helpers and optional full export when ``pxr`` is available."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tmd.model.base import ExportConfig, MeshData
from tmd.model.formats import usd as usd_mod


def test_generate_usd_texture_png(tmp_path: Path) -> None:
    hm = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    out = tmp_path / "tex.png"
    path = usd_mod._generate_usd_texture(hm, str(out), color_map="terrain", resolution=(32, 32))
    assert path == str(out)
    assert out.exists()


def test_usd_exporter_invalid_heightmap(tmp_path: Path) -> None:
    cfg = ExportConfig(x_length=1.0, y_length=1.0, z_scale=1.0, binary=False)
    bad = np.zeros((1, 1), dtype=np.float32)
    out = str(tmp_path / "bad.usda")
    if importlib.util.find_spec("pxr") is None:
        assert usd_mod.USDExporter.export(np.ones((3, 3), dtype=np.float32), out, cfg) is None
    else:
        assert usd_mod.USDExporter.export(bad, out, cfg) is None


def test_convert_to_usdz_uses_usdutils_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    orig_spec = importlib.util.find_spec

    def find_spec(name, package=None):
        if name == "pxr":
            return MagicMock()
        return orig_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    pxr = types.ModuleType("pxr")

    class UsdUtils:
        @staticmethod
        def CreateNewARKitUsdzPackage(_usd: str, usdz: str, _paths: list) -> bool:
            Path(usdz).write_bytes(b"PK\x03\x04")
            return True

    pxr.UsdUtils = UsdUtils
    monkeypatch.setitem(sys.modules, "pxr", pxr)

    usd_file = tmp_path / "scene.usdc"
    usd_file.write_text("# mock\n", encoding="utf-8")
    out = usd_mod.convert_to_usdz(str(usd_file))
    assert out is not None
    assert str(out).endswith(".usdz")


@pytest.mark.skipif(importlib.util.find_spec("pxr") is None, reason="usd-core (pxr) not installed")
def test_export_mesh_to_usd_text_minimal(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = MeshData(verts, faces)
    mesh.ensure_normals()
    mesh.ensure_uvs()
    out = tmp_path / "m.usda"
    res = usd_mod.export_mesh_to_usd(mesh, str(out), binary=False, add_texture=False)
    assert res is not None
    text = Path(res).read_text(encoding="utf-8", errors="ignore")
    assert "points" in text and "faceVertexIndices" in text
