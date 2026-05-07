"""Coverage for KLT, periodicity, FFT denoise, and correlation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.surface.filters import (
    apply_klt_filter,
    calculate_autocorrelation,
    calculate_intercorrelation,
    denoise_by_fft,
    detect_surface_periodicity,
)


def test_apply_klt_filter_whole_map_and_patches() -> None:
    flat = np.zeros((16, 16), dtype=np.float32)
    out0 = apply_klt_filter(flat, retain_components=0.99, patch_size=None)
    assert out0.shape == flat.shape

    rng = np.random.default_rng(42)
    noisy = rng.normal(0, 0.1, (16, 16)).astype(np.float32)
    out1 = apply_klt_filter(noisy, retain_components=0.95, patch_size=(4, 4), stride=2)
    assert out1.shape == noisy.shape


def test_detect_surface_periodicity() -> None:
    flat = np.zeros((24, 24), dtype=np.float32)
    r0 = detect_surface_periodicity(flat, pixel_size=1.0, threshold=0.5)
    assert "is_periodic" in r0

    grid = np.zeros((50, 50), dtype=np.float32)
    r1 = detect_surface_periodicity(grid)
    assert r1.get("is_periodic") is True

    x = np.linspace(0, 4 * np.pi, 40)
    y = np.linspace(0, 4 * np.pi, 40)
    xx, yy = np.meshgrid(x, y)
    periodic = np.sin(xx) * np.cos(yy)
    r2 = detect_surface_periodicity(periodic.astype(np.float32), threshold=0.15)
    assert "periods" in r2


def test_denoise_by_fft_2d() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(0, 0.05, (32, 32)).astype(np.float32)
    base = np.sin(np.linspace(0, 2 * np.pi, 32)).astype(np.float32)
    hm = z + base[:, np.newaxis]
    out = denoise_by_fft(
        hm,
        filter_type="lowpass",
        high_cutoff=0.15,
        apply_windowing=True,
        smooth_transition=False,
    )
    assert out.shape == hm.shape


def test_autocorrelation_and_intercorrelation_2d() -> None:
    rng = np.random.default_rng(1)
    a = rng.random((16, 16)).astype(np.float32)
    ac = calculate_autocorrelation(a, normalize=True)
    assert ac.shape[0] <= a.shape[0]
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    assert ac[cy, cx] == pytest.approx(1.0, abs=0.01)

    xc = calculate_intercorrelation(a, a.copy(), normalize=True)
    assert xc.shape == a.shape
    my, mx = xc.shape[0] // 2, xc.shape[1] // 2
    assert xc[my, mx] == pytest.approx(1.0, abs=0.05)
