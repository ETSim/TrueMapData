"""Tests for :mod:`tmd.model.base`."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tmd.model.base import (
    ExportConfig,
    MeshData,
    ModelExporter,
    _triangulation_option,
    export_heightmap_to_model,
)


def test_export_config_scale_and_extra() -> None:
    cfg = ExportConfig(scale=2.0, x_length=10.0, y_length=10.0, extra={"foo": 1})
    assert cfg.x_scale == cfg.y_scale == cfg.z_scale == 2.0
    assert cfg.extra.get("foo") == 1


def test_export_config_triangulation_method_alias() -> None:
    cfg = ExportConfig(method="MeshMethod.ADAPTIVE")
    assert cfg.triangulation_method == "adaptive"


def test_triangulation_option_from_extra() -> None:
    cfg = SimpleNamespace(extra={"max_subdivisions": 3})
    assert _triangulation_option(cfg, "max_subdivisions", 99) == 3


def test_mesh_data_vertices_faces(triangle_vertices, triangle_faces) -> None:
    m = MeshData(triangle_vertices, triangle_faces)
    assert m.vertex_count == 3 and m.face_count == 1
    d = m.as_dict()
    assert "vertices" in d and "faces" in d
    assert "MeshData" in repr(m)


def test_mesh_data_ensure_normals_uvs(triangle_vertices, triangle_faces) -> None:
    m = MeshData(triangle_vertices, triangle_faces)
    m.ensure_normals()
    assert m.normals is not None
    m2 = MeshData(triangle_vertices, triangle_faces)
    m2.ensure_uvs(method="planar")
    assert m2.uvs is not None


class _StubExporter(ModelExporter):
    format_name = "stub"
    file_extensions = ["stub"]

    @classmethod
    def export(cls, height_map: np.ndarray, filename: str, config: ExportConfig) -> str | None:
        return filename


def test_model_exporter_ensure_extension() -> None:
    assert _StubExporter.ensure_extension("out").endswith(".stub")
    assert _StubExporter.ensure_extension("out.stub") == "out.stub"


def test_create_mesh_from_heightmap_small() -> None:
    h = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    cfg = ExportConfig(triangulation_method="quadtree", x_length=4.0, y_length=4.0, max_triangles=500)
    mesh = _StubExporter.create_mesh_from_heightmap(h, cfg)
    assert mesh.vertex_count > 0 and mesh.face_count > 0


def test_export_heightmap_to_model_stl(tmp_path) -> None:
    out = tmp_path / "m.stl"
    hm = np.linspace(0, 1, 25, dtype=np.float32).reshape(5, 5)
    path = export_heightmap_to_model(
        hm,
        str(out),
        "stl",
        triangulation_method="quadtree",
        max_triangles=200,
        x_length=5.0,
        y_length=5.0,
    )
    assert path is not None
    assert __import__("os").path.isfile(path)
