"""Tests for tmd.model.utils.heightmap helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.model.utils.heightmap import (
    calculate_heightmap_center,
    calculate_terrain_complexity,
    get_heightmap_stats,
    normalize_heightmap,
    resample_heightmap,
    sample_heightmap,
    validate_heightmap,
)


def test_validate_heightmap_accepts_finite_2d() -> None:
    assert validate_heightmap(np.zeros((3, 4), dtype=np.float64)) is True


@pytest.mark.parametrize(
    "bad",
    [
        None,
        np.zeros(5),
        np.full((2, 2), np.nan),
        np.zeros((2, 2, 2)),
    ],
)
def test_validate_heightmap_rejects(bad) -> None:
    assert validate_heightmap(bad) is False


def test_normalize_heightmap_range() -> None:
    h = np.array([[0.0, 2.0], [4.0, 6.0]], dtype=np.float32)
    n = normalize_heightmap(h)
    assert float(n.min()) == 0.0
    assert float(n.max()) == 1.0


def test_normalize_heightmap_constant_is_zeros() -> None:
    h = np.ones((2, 3), dtype=np.float64)
    n = normalize_heightmap(h)
    assert np.all(n == 0)


def test_get_heightmap_stats_keys() -> None:
    h = np.arange(6, dtype=np.float64).reshape(2, 3)
    s = get_heightmap_stats(h)
    assert s["shape"] == (2, 3)
    assert s["size"] == 6
    assert s["min"] == 0.0
    assert s["max"] == 5.0
    assert "dtype" in s


def test_sample_heightmap_bilinear_center() -> None:
    h = np.array(
        [[0.0, 2.0],
         [2.0, 4.0]],
        dtype=np.float64,
    )
    # center of cell (0,0)-(1,1): x=0.5, y=0.5 → average of corners = 2.0
    assert pytest.approx(sample_heightmap(h, 0.5, 0.5)) == 2.0


def test_calculate_heightmap_center() -> None:
    assert calculate_heightmap_center(np.zeros((10, 20))) == (10.0, 5.0)


def test_calculate_terrain_complexity_positive() -> None:
    h = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float64)
    c = calculate_terrain_complexity(h, smoothing=0.0)
    assert c.shape == h.shape
    assert np.all(c >= 0)


def test_resample_heightmap_changes_shape() -> None:
    h = np.ones((4, 6), dtype=np.float64)
    out = resample_heightmap(h, (8, 12), method="nearest")
    assert out.shape == (8, 12)
