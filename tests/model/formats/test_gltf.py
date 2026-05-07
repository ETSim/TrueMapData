"""Tests for :mod:`tmd.model.formats.gltf`."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.model.base import ExportConfig
from tmd.model.formats.gltf import GLTFExporter

pygltflib = pytest.importorskip("pygltflib", reason="GLTF tests need pygltflib")


def test_gltf_export_minimal(tmp_path) -> None:
    hm = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    out = str(tmp_path / "m.gltf")
    res = GLTFExporter.export(
        hm,
        out,
        ExportConfig(triangulation_method="quadtree", max_triangles=200, x_length=4, y_length=4),
    )
    assert res is not None
    p = __import__("pathlib").Path(res)
    assert p.read_text()[:20].strip().startswith("{")
