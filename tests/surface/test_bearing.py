"""Tests for :mod:`tmd.surface.metrics` (bearing / Abbott)."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.surface.metrics import (
    bearing_analysis,
    material_ratio_curve,
    remove_least_squares_plane,
    rmr_at_depths,
)


def test_remove_plane_exact_linear_surface() -> None:
    h, w = 20, 25
    yy, xx = np.indices((h, w), dtype=np.float64)
    z = 0.1 * xx + 0.05 * yy + 2.0
    leveled, meta = remove_least_squares_plane(z)
    assert meta["n_finite"] == h * w
    np.testing.assert_allclose(leveled, 0.0, atol=1e-9, rtol=0)


def test_material_ratio_plane_with_groove_fraction() -> None:
    """Groove occupies 10% of columns; curve is well-defined and monotone in Rmr."""
    h, w = 50, 50
    z = np.zeros((h, w), dtype=np.float64)
    z[:, 20:25] = -0.3  # 5/50 = 10% columns
    leveled, _ = remove_least_squares_plane(z)
    depths, rmr, info = material_ratio_curve(leveled, n_depth_samples=64)
    assert not info.get("degenerate")
    assert np.all(np.diff(rmr) >= -1e-9)
    assert rmr[-1] == pytest.approx(100.0)
    assert depths[-1] > 0


def test_rmr_at_depths_interpolation() -> None:
    depths = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    rmr = np.array([0.0, 50.0, 100.0], dtype=np.float64)
    q = np.array([0.5, 1.5])
    out = rmr_at_depths(depths, rmr, q)
    np.testing.assert_allclose(out, [25.0, 75.0], rtol=0, atol=1e-9)


def test_bearing_analysis_rmr_query() -> None:
    z = np.array([[0.0, 0.0], [0.0, -0.4]], dtype=np.float64)
    out = bearing_analysis(z, n_depth_samples=16, rmr_query_depths=[0.05, 0.2])
    assert "rmr_at_depth" in out
    assert len(out["rmr_at_depth"]["depths"]) == 2
