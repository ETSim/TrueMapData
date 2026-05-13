"""Tests for :mod:`tmd.surface.metrics.curvature` (mean curvature + surface normals)."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.image.maps.curvature import CurvatureMapGenerator
from tmd.surface.metrics import (
    mean_curvature,
    pixel_spacing_dy_dx_from_metadata,
    surface_normals_flat,
)


def test_mean_curvature_matches_generator_spot() -> None:
    z = np.random.default_rng(2).random((16, 16)).astype(np.float64) * 0.1
    cx, cy = 0.05, 0.05
    gen = CurvatureMapGenerator()
    h_gen = gen._calculate_curvature(z, "mean", cx, cy, 1.0)
    h_fn = mean_curvature(z, cx, cy, 1.0)
    assert np.allclose(h_gen, h_fn, rtol=1e-5, atol=1e-6)


def test_surface_normals_flat_rejects_all_non_finite() -> None:
    z = np.full((4, 5), np.nan, dtype=np.float64)
    with pytest.raises(ValueError, match="finite"):
        surface_normals_flat(z)


def test_pixel_spacing_requires_2d_shape() -> None:
    with pytest.raises(ValueError, match="2D"):
        pixel_spacing_dy_dx_from_metadata({}, (5,))
