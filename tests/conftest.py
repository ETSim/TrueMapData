"""Shared pytest fixtures for TrueMapData tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tmd.utils.utils import TMDUtils


@pytest.fixture
def small_heightmap() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.1, 0.2],
            [0.1, 0.3, 0.4],
            [0.0, 0.2, 0.5],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def triangle_vertices() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


@pytest.fixture
def triangle_faces() -> np.ndarray:
    return np.array([[0, 1, 2]], dtype=np.int32)


@pytest.fixture
def obj_triangle_text() -> str:
    return """\
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
"""


@pytest.fixture
def tmp_tmd_path(tmp_path: Path, small_heightmap: np.ndarray) -> Path:
    path = tmp_path / "fixture.tmd"
    TMDUtils.write_tmd_file(
        small_heightmap,
        str(path),
        comment="pytest fixture\n",
        version=2,
    )
    return path
