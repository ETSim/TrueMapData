"""Optional Surfalize tests for tribology plane removal (skipped if Surfalize is not installed)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("surfalize")

from tmd.surface.metrics import surfalize as ts
from tmd.surface.metrics import bearing_area_curve, _preprocess_plane


def test_level_height_map_surfalize_removes_tilt() -> None:
    """Least-squares leveling should remove a dominant plane better than mean-only offset."""
    h, w = 32, 40
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    z = (0.02 * xx + 0.015 * yy + 0.001 * np.sin(xx)).astype(np.float32)
    meta = {"x_length": 2.0, "y_length": 2.0}
    leveled = ts.level_height_map_surfalize(z, meta)
    assert leveled is not None
    assert leveled.shape == z.shape
    assert float(np.std(leveled)) < float(np.std(z)) * 0.5
    assert abs(float(np.mean(leveled))) < abs(float(np.mean(z))) + 1e-3


def test_preprocess_surfalize_matches_helper() -> None:
    rng = np.random.default_rng(42)
    z = rng.normal(size=(24, 28)).astype(np.float32)
    meta = {"x_length": 1.0, "y_length": 1.0}
    a = _preprocess_plane(z, meta, "surfalize")
    b = ts.level_height_map_surfalize(z, meta)
    assert b is not None
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-4)


def test_bearing_area_curve_with_surfalize_metadata() -> None:
    rng = np.random.default_rng(7)
    z = rng.normal(size=(20, 20)).astype(np.float64)
    meta = {"x_length": 1.0, "y_length": 1.0}
    c = bearing_area_curve(z, n=12, metadata=meta, plane_removal="surfalize")
    assert c["plane_removal"] == "surfalize"
    assert c["area_fraction"][0] >= c["area_fraction"][-1] - 1e-5
