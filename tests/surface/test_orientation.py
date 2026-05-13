"""Tests for :mod:`tmd.surface.filters` gradient / texture-direction helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.surface import filters as flt


def test_global_texture_angle_sinusoidal_grating() -> None:
    """Dominant lay along x → gradient in y; doubled angle should align."""
    x = np.linspace(0, 4 * np.pi, 128)
    y = np.linspace(0, 4 * np.pi, 128)
    _, yy = np.meshgrid(x, y)
    z = 0.15 * np.sin(yy)  # ridges along x
    gy, gx = np.gradient(z.astype(np.float64))
    ang = flt.global_texture_angle_rad(gx, gy)
    assert abs(abs(ang) - np.pi / 2) < 0.2


def test_gradient_slant_angle_rad_horizontal_ridge() -> None:
    """Constant gradient in x → angle 0."""
    gx = np.ones((8, 8), dtype=np.float64)
    gy = np.zeros_like(gx)
    ang = flt.gradient_slant_angle_rad(gx, gy)
    np.testing.assert_allclose(ang, 0.0, atol=1e-12)


def test_wrapped_angle_diff_rad_symmetry() -> None:
    a = np.array([0.0, np.pi / 2, -np.pi / 4], dtype=np.float64)
    d = flt.wrapped_angle_diff_rad(a, 0.0)
    assert d.shape == a.shape
    assert np.all((d >= 0.0) & (d <= np.pi + 1e-9))


def test_local_texture_angle_and_coherence_uniform_field() -> None:
    """Constant doubled-angle field → coherence ≈ 1 everywhere."""
    cos2 = np.full((16, 16), 0.6, dtype=np.float64)
    sin2 = np.full((16, 16), 0.8, dtype=np.float64)
    ang, coh = flt.local_texture_angle_and_coherence(cos2, sin2, window=5)
    assert ang.shape == coh.shape == cos2.shape
    np.testing.assert_allclose(coh, 1.0, atol=1e-9)


def test_global_texture_angle_with_explicit_weights() -> None:
    gx = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    gy = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    w_x = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    w_y = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    a_x = flt.global_texture_angle_rad(gx, gy, weights=w_x)
    a_y = flt.global_texture_angle_rad(gx, gy, weights=w_y)
    assert pytest.approx(0.0, abs=1e-9) == a_x
    assert pytest.approx(np.pi / 2, abs=1e-9) == abs(a_y)
