"""Tests for tmd.model.utils.validation."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.model.utils.validation import (
    ensure_directory_exists,
    validate_config,
    validate_faces,
    validate_heightmap,
    validate_mesh,
    validate_vertices,
)


def test_validate_vertices_ok() -> None:
    v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    assert validate_vertices(v) is True
    assert validate_vertices(v.tolist()) is True


@pytest.mark.parametrize(
    "bad",
    [
        None,
        np.zeros((2, 2)),
        np.zeros((3, 4)),
        np.array([[np.nan, 0.0, 0.0]]),
        np.array([[np.inf, 0.0, 0.0]]),
    ],
)
def test_validate_vertices_rejects(bad) -> None:
    assert validate_vertices(bad) is False


def test_validate_faces_ok() -> None:
    f = np.array([[0, 1, 2]], dtype=np.int64)
    assert validate_faces(f) is True
    assert validate_faces(f.tolist()) is True


@pytest.mark.parametrize(
    "bad",
    [
        None,
        np.array([[0, 1]]),
        np.array([[0, 1, 2, 3]]),
        np.array([[-1, 0, 1]]),
    ],
)
def test_validate_faces_rejects(bad) -> None:
    assert validate_faces(bad) is False


def test_validate_heightmap_ok() -> None:
    h = np.zeros((4, 5), dtype=np.float32)
    assert validate_heightmap(h) is True


@pytest.mark.parametrize(
    "bad",
    [
        None,
        np.zeros(10),
        np.zeros((1, 5)),
        np.zeros((5, 1)),
        np.array([[]]),
        "not_array",
    ],
)
def test_validate_heightmap_rejects(bad) -> None:
    assert validate_heightmap(bad) is False


def test_ensure_directory_exists_creates_parent(tmp_path) -> None:
    target = tmp_path / "nested" / "out.bin"
    assert ensure_directory_exists(str(target)) is True
    assert target.parent.is_dir()


def test_validate_mesh_valid() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    ok, issues = validate_mesh(vertices, faces)
    assert ok is True
    assert issues == []


def test_validate_mesh_invalid_vertices() -> None:
    vertices = np.array([[0.0, 0.0]], dtype=np.float64)
    faces = np.array([[0, 0, 0]], dtype=np.int64)
    ok, issues = validate_mesh(vertices, faces)
    assert ok is False
    assert any("vertex" in msg.lower() for msg in issues)


def test_validate_mesh_non_manifold_edge() -> None:
    """Three triangles share one edge — should flag non-manifold."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 1, 4],
        ],
        dtype=np.int64,
    )
    ok, issues = validate_mesh(vertices, faces)
    assert ok is False
    assert any("non-manifold" in msg.lower() for msg in issues)


def test_validate_config_accepts_good() -> None:
    cfg = {
        "z_scale": 1.0,
        "max_triangles": 100,
        "coordinate_system": "right-handed",
        "optimize": True,
    }
    ok, issues = validate_config(cfg)
    assert ok is True
    assert issues == []


@pytest.mark.parametrize(
    "cfg,snippet",
    [
        ({"z_scale": -1}, "z_scale"),
        ({"max_triangles": 0}, "max_triangles"),
        ({"coordinate_system": "bad"}, "coordinate_system"),
        ({"optimize": "yes"}, "optimize"),
    ],
)
def test_validate_config_rejects(cfg: dict, snippet: str) -> None:
    ok, issues = validate_config(cfg)
    assert ok is False
    assert any(snippet in msg for msg in issues)
