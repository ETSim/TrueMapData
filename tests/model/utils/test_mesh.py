"""Tests for :mod:`tmd.model.utils.mesh`."""

from __future__ import annotations

import numpy as np
from tmd.model.utils import mesh as mesh_u


def test_create_mesh_from_heightmap() -> None:
    h = np.array([[0.0, 0.5], [0.5, 1.0]], dtype=np.float32)
    v, f = mesh_u.create_mesh_from_heightmap(h, x_length=1.0, y_length=1.0)
    assert len(v) == 4 and len(f) == 2


def test_normals_from_triangle(triangle_vertices, triangle_faces) -> None:
    vn = mesh_u.calculate_vertex_normals(triangle_vertices, triangle_faces)
    assert vn.shape == triangle_vertices.shape
    fn = mesh_u.calculate_face_normals(triangle_vertices, triangle_faces)
    assert fn.shape[0] == len(triangle_faces)


def test_generate_uv_coordinates() -> None:
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        dtype=np.float32,
    )
    uvs_p = mesh_u.generate_uv_coordinates(verts, method="planar")
    assert uvs_p.shape[0] == 4
    uvs_c = mesh_u.generate_uv_coordinates(verts, method="cylindrical")
    assert uvs_c.shape[0] == 4
    uvs_s = mesh_u.generate_uv_coordinates(verts, method="spherical")
    assert uvs_s.shape[0] == 4
    uvs_d = mesh_u.generate_uv_coordinates(verts, method="unknown_defaults_planar")
    assert uvs_d.shape == uvs_p.shape


def test_optimize_mesh_noop_on_clean_triangle(triangle_vertices, triangle_faces) -> None:
    res = mesh_u.optimize_mesh(triangle_vertices, triangle_faces)
    assert res is not None
    v2, f2 = res
    assert v2.size > 0 and f2.size > 0
