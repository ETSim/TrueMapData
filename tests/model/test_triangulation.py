"""Tests for heightmap triangulation and mesh creation."""

import numpy as np
import pytest

from tmd.model.config import ExportConfig
from tmd.model.factory import ModelExporterFactory
from tmd.model.formats.stl import STLExporter
from tmd.model.utils.heightmap import normalize_heightmap_for_triangulation
from tmd.surface.terrain import TMDTerrain


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    out = []
    for a, b, c in faces:
        p0, p1, p2 = vertices[a], vertices[b], vertices[c]
        out.append(0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0))))
    return np.array(out, dtype=np.float64)


def _synthetic_height_map(h: int = 36, w: int = 36, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    return (x + y + 0.08 * rng.standard_normal((h, w)).astype(np.float32)).astype(np.float32)


@pytest.mark.parametrize("method", ["adaptive", "quadtree"])
def test_create_mesh_synthetic_heightmap(method: str) -> None:
    height_map = _synthetic_height_map()
    config = ExportConfig(
        triangulation_method=method,
        method=method,
        max_triangles=9000,
        error_threshold=0.06,
        calculate_normals=False,
        optimize=False,
    )
    mesh = STLExporter.create_mesh_from_heightmap(height_map, config)

    assert mesh.vertices.ndim == 2 and mesh.vertices.shape[1] == 3
    assert mesh.faces.ndim == 2 and mesh.faces.shape[1] == 3
    assert len(mesh.faces) <= 9000
    assert not np.any(np.isnan(mesh.vertices))
    max_idx = mesh.vertices.shape[0] - 1
    assert int(mesh.faces.min()) >= 0 and int(mesh.faces.max()) <= max_idx
    areas = _triangle_areas(mesh.vertices, mesh.faces)
    assert np.all(areas > 1e-10)


def test_detail_boost_wired_for_adaptive() -> None:
    """Higher detail_boost refines more in non-flat regions (detail_boost is forwarded)."""
    h = w = 72
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    height_map = np.sin(xx * 0.35) * np.cos(yy * 0.28) + 0.15 * _synthetic_height_map(h, w, seed=4)
    common = dict(
        triangulation_method="adaptive",
        method="adaptive",
        max_triangles=15_000,
        error_threshold=0.008,
        calculate_normals=False,
        optimize=False,
    )
    low_boost = STLExporter.create_mesh_from_heightmap(
        height_map.copy(),
        ExportConfig(**common, detail_boost=0.02),
    )
    high_boost = STLExporter.create_mesh_from_heightmap(
        height_map.copy(),
        ExportConfig(**common, detail_boost=2.0),
    )
    assert len(high_boost.faces) > len(low_boost.faces)


def test_normalize_heightmap_for_triangulation_float_and_uint16() -> None:
    x = np.array([[0.0, 2.0], [1.0, 3.0]], dtype=np.float32)
    n = normalize_heightmap_for_triangulation(x)
    assert n.dtype == np.float32
    assert np.all(n >= 0) and np.all(n <= 1.0)

    u = np.array([[0, 32767], [65535, 16384]], dtype=np.uint16)
    nu = normalize_heightmap_for_triangulation(u)
    assert nu.dtype == np.float32
    assert np.isclose(float(nu.max()), 1.0, atol=1e-4)


def test_factory_export_synthetic_tmd(tmp_path) -> None:
    tmd_path = tmp_path / "sample.tmd"
    stl_path = tmp_path / "out.stl"
    TMDTerrain.generate_synthetic_tmd(
        output_path=str(tmd_path),
        width=40,
        height=40,
        pattern="dome",
    )
    config = ExportConfig(
        triangulation_method="adaptive",
        method="adaptive",
        max_triangles=6000,
        error_threshold=0.08,
    )
    config.extra["save_heightmap"] = False

    ok = ModelExporterFactory().export(str(tmd_path), str(stl_path), "stl", config)
    assert ok
    assert stl_path.is_file()
    assert stl_path.stat().st_size > 0
