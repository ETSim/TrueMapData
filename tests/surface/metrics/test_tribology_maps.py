"""Tests for normalized tribology proxy map helpers in :mod:`tmd.surface.metrics.tribology_maps`."""

from __future__ import annotations

import numpy as np

from tmd.surface.metrics.tribology_maps import debris_pocket_map_01, shear_hazard_map_01


def test_shear_hazard_map_01_shape_and_range() -> None:
    x = np.linspace(-1, 1, 32)
    xx, yy = np.meshgrid(x, x)
    z = (xx**2 + yy**2).astype(np.float32)
    meta = {"x_length": 1.0, "y_length": 1.0}
    out = shear_hazard_map_01(z, meta, roughness_sigma=3.0)
    assert out.shape == z.shape
    assert out.dtype == np.float32
    assert float(np.min(out)) >= 0.0
    assert float(np.max(out)) <= 1.0


def test_debris_pocket_map_01_shape_and_range() -> None:
    z = np.zeros((24, 24), dtype=np.float32)
    z[8:10, 8:10] = -0.5
    meta = {"x_length": 1.0, "y_length": 1.0}
    out = debris_pocket_map_01(z, meta)
    assert out.shape == z.shape
    assert out.dtype == np.float32
    assert float(np.min(out)) >= 0.0
    assert float(np.max(out)) <= 1.0
