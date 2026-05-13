"""Tests for Ausloos–Berman Weierstrass–Mandelbrot fractal terrain."""

import numpy as np
import pytest

from tmd.surface.terrain import (
    TMDTerrain,
    ausloos_berman_wm_grid,
    make_phases_for_seed,
    validate_wm_params,
)


def test_validate_wm_params_rejects_bad_df() -> None:
    with pytest.raises(ValueError, match="D_f"):
        validate_wm_params(2.0, 1.5, 8, 8)
    with pytest.raises(ValueError, match="D_f"):
        validate_wm_params(3.0, 1.5, 8, 8)


def test_validate_wm_params_rejects_bad_gamma() -> None:
    with pytest.raises(ValueError, match="gamma"):
        validate_wm_params(2.5, 1.0, 8, 8)


def test_ausloos_berman_wm_grid_shape_and_finite() -> None:
    x = np.linspace(-1, 1, 12)
    y = np.linspace(-1, 1, 10)
    X, Y = np.meshgrid(x, y)
    M, N_max = 6, 10
    phases = make_phases_for_seed(12345, M, N_max)
    Z = ausloos_berman_wm_grid(
        X, Y, L=2.0, D_f=2.5, gamma=1.5, M=M, N_max=N_max, phases=phases
    )
    assert Z.shape == (10, 12)
    assert np.all(np.isfinite(Z))


def test_create_sample_height_map_wm_ab_reproducible() -> None:
    a = TMDTerrain.create_sample_height_map(
        width=32, height=32, pattern="wm_ab", seed=99, noise_level=0.0
    )
    b = TMDTerrain.create_sample_height_map(
        width=32, height=32, pattern="wm_ab", seed=99, noise_level=0.0
    )
    assert a.shape == (32, 32)
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_create_sample_height_map_wm_ab_normalized_range() -> None:
    z = TMDTerrain.create_sample_height_map(
        width=24, height=24, pattern="wm_ab", seed=7, noise_level=0.0
    )
    assert z.dtype == np.float32
    assert z.min() >= 0.0
    assert z.max() <= 1.0


def test_wm_ab_differs_when_df_or_gamma_changes() -> None:
    kw = dict(width=28, height=28, pattern="wm_ab", seed=42, noise_level=0.0)
    base = TMDTerrain.create_sample_height_map(**kw)
    alt_df = TMDTerrain.create_sample_height_map(**kw, wm_df=2.7)
    alt_g = TMDTerrain.create_sample_height_map(**kw, wm_gamma=1.7)
    assert not np.allclose(base, alt_df)
    assert not np.allclose(base, alt_g)


def test_ausloos_berman_wm_alias_pattern() -> None:
    a = TMDTerrain.create_sample_height_map(
        width=16, height=16, pattern="wm_ab", seed=3, noise_level=0.0
    )
    b = TMDTerrain.create_sample_height_map(
        width=16, height=16, pattern="ausloos_berman_wm", seed=3, noise_level=0.0
    )
    np.testing.assert_allclose(a, b, rtol=0, atol=0)
