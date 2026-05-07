"""Tests for :mod:`tmd.model.formats.obj`."""

from __future__ import annotations

import numpy as np

from tmd.model.base import ExportConfig, MeshData
from tmd.model.formats.obj import OBJExporter, apply_uv_margin_to_obj_lines, write_obj


def test_apply_uv_margin_to_obj_lines() -> None:
    lines = ["vt 0.0 0.0", "vt 1.0 1.0", "v 0 0 0"]
    out = apply_uv_margin_to_obj_lines(lines, 0.1)
    assert any(line.startswith("vt ") for line in out)


def test_write_obj_file(tmp_path, triangle_vertices, triangle_faces) -> None:
    m = MeshData(triangle_vertices, triangle_faces)
    m.ensure_normals()
    m.ensure_uvs()
    p = str(tmp_path / "tri.obj")
    write_obj(m, p, include_materials=False)
    text = __import__("pathlib").Path(p).read_text()
    assert "v " in text and "f " in text


def test_obj_exporter_heightmap(tmp_path) -> None:
    hm = np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3)
    out = str(tmp_path / "hm.obj")
    path = OBJExporter.export(hm, out, ExportConfig(triangulation_method="quadtree", max_triangles=300))
    assert path is not None
