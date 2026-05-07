"""Extra STL exporter and helper coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tmd.model.base import ExportConfig, MeshData
from tmd.model.formats.stl import (
    STLExporter,
    _calculate_face_normals,
    calculate_hillshade,
    generate_heightmap_texture,
    generate_heightmap_visualization,
    normalize_heightmap,
    write_ascii_stl,
)


def test_write_ascii_stl_and_face_normals(
    tmp_path: Path, triangle_vertices: np.ndarray, triangle_faces: np.ndarray
) -> None:
    mesh = MeshData(triangle_vertices, triangle_faces)
    out = tmp_path / "tri.stl"
    write_ascii_stl(mesh, str(out), z_scale=1.0)
    text = out.read_text(encoding="utf-8")
    assert "solid" in text and "facet normal" in text

    mesh2 = MeshData(triangle_vertices, triangle_faces)
    normals = _calculate_face_normals(mesh2)
    np.testing.assert_allclose(normals[0], [0, 0, 1], atol=1e-5)


def test_normalize_heightmap_flat_and_ramp() -> None:
    flat = np.zeros((3, 3), dtype=np.float32)
    np.testing.assert_array_equal(normalize_heightmap(flat), flat)
    ramp = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    n = normalize_heightmap(ramp)
    assert n.min() == 0 and n.max() == 1


def test_generate_heightmap_texture_resolution_branch() -> None:
    hm = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    rgb = generate_heightmap_texture(hm, colormap="terrain", resolution=(8, 8))
    assert rgb.shape == (8, 8, 3)


def test_calculate_hillshade_and_visualization(tmp_path: Path) -> None:
    hm = np.random.default_rng(0).random((12, 12)).astype(np.float32)
    shade = calculate_hillshade(hm)
    assert shade.shape == hm.shape
    out = tmp_path / "viz.png"
    path = generate_heightmap_visualization(hm, str(out), add_shading=True)
    assert path == str(out) or Path(path).exists()


def test_stl_exporter_ascii_with_heightmap_png(tmp_path: Path) -> None:
    hm = np.array(
        [[0.0, 0.5, 1.0], [0.2, 0.4, 0.6], [0.1, 0.3, 0.9]],
        dtype=np.float32,
    )
    cfg = ExportConfig(
        x_length=1.0,
        y_length=1.0,
        z_scale=1.0,
        binary=False,
        base_height=0.01,
        extra={"save_heightmap": True, "colormap": "terrain"},
    )
    stl_path = tmp_path / "out.stl"
    result = STLExporter.export(hm, str(stl_path), cfg)
    assert result is not None
    assert stl_path.exists()
    assert (tmp_path / "out_heightmap.png").exists()
