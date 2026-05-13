"""Tests for :mod:`tmd.surface.metrics` (shear proxy)."""

from __future__ import annotations

import numpy as np

from tmd.surface.metrics import local_rms, shear_proxy_map


def test_shear_proxy_constant_height_near_zero() -> None:
    z = np.ones((32, 32), dtype=np.float64)
    p = shear_proxy_map(z, window=5, normalize="none")
    np.testing.assert_allclose(p, 0.0, atol=1e-9)


def test_shear_proxy_shape_and_finite() -> None:
    yy, xx = np.indices((24, 24), dtype=np.float64)
    z = 0.01 * xx + 0.02 * np.sin(yy / 3.0)
    p = shear_proxy_map(z, window=7, normalize="p98")
    assert p.shape == z.shape
    assert np.all(np.isfinite(p))
    assert float(p.max()) <= 1.0 + 1e-6


def test_local_rms_flat() -> None:
    z = np.zeros((16, 16), dtype=np.float64)
    r = local_rms(z, window=5)
    np.testing.assert_allclose(r, 0.0, atol=1e-9)
