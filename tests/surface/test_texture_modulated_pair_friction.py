"""Tests for texture-modulated pair friction maps (tribology + wear_simulation baseline)."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.sequence.wear_simulation import DIRECTION_NAMES, WearParameters, combined_mu
from tmd.surface.metrics import slide_azimuth_rad_from_direction, texture_modulated_pair_friction_maps


def test_slide_azimuth_rad_from_direction() -> None:
    assert slide_azimuth_rad_from_direction(1) == pytest.approx(0.0)
    assert slide_azimuth_rad_from_direction(2) == pytest.approx(np.pi)
    assert slide_azimuth_rad_from_direction(3) == pytest.approx(np.pi / 2)
    assert slide_azimuth_rad_from_direction(4) == pytest.approx(-np.pi / 2)


def test_slide_azimuth_invalid_direction() -> None:
    with pytest.raises(ValueError):
        slide_azimuth_rad_from_direction(99)


def test_texture_modulated_same_shape() -> None:
    ms = np.zeros((8, 8), dtype=np.float64)
    ss = np.zeros((8, 9), dtype=np.float64)
    p = WearParameters()
    with pytest.raises(ValueError, match="same shape"):
        texture_modulated_pair_friction_maps(ms, ss, p)


def test_texture_modulated_invalid_fusion() -> None:
    ms = np.zeros((8, 8), dtype=np.float64)
    p = WearParameters()
    with pytest.raises(ValueError, match="fusion"):
        texture_modulated_pair_friction_maps(ms, ms.copy(), p, fusion="invalid")  # type: ignore[arg-type]


def test_texture_modulated_strength_zero_matches_combined_mu() -> None:
    rng = np.random.default_rng(42)
    ms = rng.standard_normal((24, 24)).astype(np.float64)
    ss = rng.standard_normal((24, 24)).astype(np.float64) * 0.5
    p = WearParameters(friction_clip=0.4)
    maps = texture_modulated_pair_friction_maps(ms, ss, p, strength=0.0, window=7)
    for d in DIRECTION_NAMES:
        expected = combined_mu(ms, ss, p, int(d))
        assert np.allclose(maps[int(d)], expected, rtol=0, atol=1e-12)


def test_texture_modulated_maps_shape_and_finite() -> None:
    ms = np.sin(np.linspace(0, 6.28, 32))[None, :].repeat(32, axis=0)
    ss = np.cos(np.linspace(0, 6.28, 32))[None, :].repeat(32, axis=0) * 0.3
    p = WearParameters(friction_clip=0.35)
    maps = texture_modulated_pair_friction_maps(ms, ss, p, strength=0.25, window=9)
    assert set(maps) == set(DIRECTION_NAMES)
    for d, arr in maps.items():
        assert arr.shape == ms.shape
        assert np.isfinite(arr).all()
        assert float(arr.min()) >= 0.0
        assert float(arr.max()) <= float(p.friction_clip) + 1e-9


def test_x_grating_modulation_favors_slide_x_over_slide_y() -> None:
    """Sinusoidal grating along x: local texture angle aligns with ±x; expect higher w for dir 1 than 3."""
    h = w = 96
    x = np.linspace(0, 2 * np.pi * 10, w, dtype=np.float64)
    ms = np.sin(x)[None, :].repeat(h, axis=0)
    ss = np.zeros_like(ms)
    p = WearParameters(friction_clip=0.5)
    _, aux = texture_modulated_pair_friction_maps(
        ms, ss, p, strength=0.95, fusion="rms", window=9, return_aux=True
    )
    mod = aux["modulation"]
    m1 = float(np.mean(mod[1]))
    m3 = float(np.mean(mod[3]))
    assert m1 > m3 * 1.1


def test_fusion_mean_vs_rms_both_finite() -> None:
    rng = np.random.default_rng(1)
    ms = rng.standard_normal((20, 20))
    ss = rng.standard_normal((20, 20)) * 0.4
    p = WearParameters(friction_clip=0.3)
    rms_maps = texture_modulated_pair_friction_maps(ms, ss, p, strength=0.5, fusion="rms")
    mean_maps = texture_modulated_pair_friction_maps(ms, ss, p, strength=0.5, fusion="mean")
    for d in DIRECTION_NAMES:
        assert np.isfinite(rms_maps[int(d)]).all()
        assert np.isfinite(mean_maps[int(d)]).all()
