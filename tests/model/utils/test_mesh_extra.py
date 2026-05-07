"""Coverage for :func:`_add_base_to_mesh` and related mesh helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.model.utils.mesh import (
    _add_base_to_mesh,
    calculate_face_normals,
    calculate_vertex_normals,
    create_mesh_from_heightmap,
    optimize_mesh,
)
from tmd.model.utils.mesh import _generate_spherical_uvs


def test_add_base_to_mesh_adds_floor_and_short_circuit() -> None:
    hm = np.array(
        [
            [0.0, 0.1, 0.2, 0.0],
            [0.1, 0.2, 0.3, 0.1],
            [0.0, 0.1, 0.0, 0.2],
            [0.2, 0.1, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    verts, faces = create_mesh_from_heightmap(hm, base_height=0.0)
    n0 = len(verts)
    f0 = len(faces)

    v2, f2 = _add_base_to_mesh(verts, faces, 0.0)
    assert len(v2) == n0 and len(f2) == f0

    v3, f3 = _add_base_to_mesh(verts, faces, 0.05)
    assert len(v3) == n0 + 5
    assert len(f3) > f0
    base_z = min(v[2] for v in v3)
    surf_z = min(v[2] for v in verts)
    assert base_z < surf_z - 1e-6


def test_calculate_vertex_and_face_normals(triangle_vertices: np.ndarray, triangle_faces: np.ndarray) -> None:
    vn = calculate_vertex_normals(triangle_vertices, triangle_faces)
    assert vn.shape == triangle_vertices.shape
    fn = calculate_face_normals(triangle_vertices, triangle_faces)
    assert fn.shape == (1, 3)
    np.testing.assert_allclose(fn[0], [0, 0, 1], atol=1e-5)


def test_generate_spherical_uvs_and_optimize_mesh() -> None:
    verts = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    uvs = _generate_spherical_uvs(verts)
    assert uvs.shape == (3, 2)

    v2 = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0.001]], dtype=np.float32)
    f2 = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    out = optimize_mesh(v2, f2, tolerance=1e-3)
    assert out is not None
    ov, of = out
    assert ov.shape[0] >= 3 and of.shape[0] >= 1
