"""Tests for ModelExporterFactory.export_heightmap."""

from __future__ import annotations

import numpy as np

from tmd.model.config import ExportConfig
from tmd.model.factory import ModelExporterFactory


def test_export_heightmap_stl_writes_file(tmp_path) -> None:
    h = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
    out = tmp_path / "mesh.stl"
    config = ExportConfig(
        triangulation_method="adaptive",
        method="adaptive",
        max_triangles=4000,
        error_threshold=0.05,
        calculate_normals=False,
        optimize=False,
    )
    path = ModelExporterFactory.export_heightmap(h, str(out), "stl", config)
    assert path is not None
    assert out.is_file()
    assert out.stat().st_size > 0


def test_export_heightmap_unknown_format_returns_none() -> None:
    h = np.zeros((8, 8), dtype=np.float32)
    config = ExportConfig(
        triangulation_method="adaptive",
        method="adaptive",
        max_triangles=100,
        error_threshold=0.1,
    )
    assert (
        ModelExporterFactory.export_heightmap(
            h,
            "nowhere.obj",
            "not_a_real_format_xyz",
            config,
        )
        is None
    )


def test_get_available_formats_includes_stl() -> None:
    fmt = ModelExporterFactory.get_available_formats()
    assert "stl" in fmt
