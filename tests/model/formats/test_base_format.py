"""Tests for :mod:`tmd.model.formats.base`."""

from __future__ import annotations

import numpy as np

from tmd.model.formats.base import create_mesh_from_heightmap, validate_mesh


def test_create_mesh_from_heightmap_minimal() -> None:
    h = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    v, f = create_mesh_from_heightmap(h, x_length=1.0, y_length=1.0, z_scale=1.0)
    assert len(v) == 4 and len(f) >= 2
    assert validate_mesh(v, f) is True


def test_create_mesh_from_heightmap_with_base() -> None:
    h = np.ones((2, 2), dtype=np.float32)
    v, f = create_mesh_from_heightmap(h, base_height=0.1, x_length=1.0, y_length=1.0)
    assert len(v) > 4
    assert len(f) > 2


def test_validate_mesh_rejects_bad_index() -> None:
    v = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    f = [[0, 1, 99]]
    assert validate_mesh(v, f) is False


def test_validate_mesh_rejects_degenerate() -> None:
    assert validate_mesh([[0, 0, 0]], [[0, 0, 0]]) is False
