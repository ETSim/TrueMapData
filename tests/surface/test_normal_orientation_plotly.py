"""Tests for normal-orientation helpers in ``tmd.surface.metrics.curvature``."""

from __future__ import annotations

import numpy as np

from tmd.surface.metrics.curvature import (
    orientation_histograms,
    pixel_spacing_dy_dx_from_metadata,
)


def _synthetic_heights(n: int = 36) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)
    z0 = 0.12 * np.sin(5 * xx) * np.cos(4 * yy)
    z1 = z0 + 0.03 * np.exp(-4.0 * (xx**2 + yy**2))
    return z0.astype(np.float32), z1.astype(np.float32)


def test_orientation_histograms_shapes_and_finite() -> None:
    b, a = _synthetic_heights(40)
    oh = orientation_histograms(b, a, n_lat=24, n_lon=32)
    assert oh.h_before.shape == oh.h_after.shape == oh.h_delta.shape
    assert np.all(np.isfinite(oh.h_before))
    assert np.all(np.isfinite(oh.h_after))
    assert np.all(np.isfinite(oh.h_delta))


def test_physical_spacing_spreads_sphere_histogram_vs_index_units() -> None:
    """TMD-like heights in mm with small per-pixel delta need mm lateral spacing."""
    n = 80
    j = np.arange(n, dtype=np.float64)
    z = np.tile(1.5 * j / max(n - 1, 1), (n, 1)).astype(np.float32)
    oh_index = orientation_histograms(z, z, n_lat=40, n_lon=40, dy=1.0, dx=1.0)
    pole_frac_index = float(oh_index.h_before[-1].sum() / max(float(oh_index.h_before.sum()), 1e-12))

    mmpp = 0.05
    oh_phys = orientation_histograms(z, z, n_lat=40, n_lon=40, dy=mmpp, dx=mmpp)
    pole_frac_phys = float(oh_phys.h_before[-1].sum() / max(float(oh_phys.h_before.sum()), 1e-12))
    # Index-space gradients are tiny → almost all mass in top latitude bin
    assert pole_frac_index > 0.85
    # Physical spacing steepens ∂z/∂x → normals spread in latitude
    assert pole_frac_phys < 0.75


def test_pixel_spacing_from_prefixed_metadata() -> None:
    meta = {"a_mmpp": 0.002, "noise": 1}
    dy, dx = pixel_spacing_dy_dx_from_metadata(meta, (100, 200), key_prefix="a_")
    assert dy == 0.002 and dx == 0.002

    meta2 = {"a_x_length": 4.0, "a_y_length": 2.0}
    dy2, dx2 = pixel_spacing_dy_dx_from_metadata(meta2, (100, 200), key_prefix="a_")
    assert abs(dx2 - 4.0 / 200) < 1e-9
    assert abs(dy2 - 2.0 / 100) < 1e-9
