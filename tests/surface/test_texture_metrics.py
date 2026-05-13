"""Tests for the FFT-based texture-direction spectrum and metrics summary."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tmd.surface.metrics import surface_metrics_summary, texture_direction_spectrum


def _stripes(n: int, freq: int, axis: str) -> np.ndarray:
    coord = np.linspace(0.0, 2 * np.pi * freq, n, dtype=np.float64)
    if axis == "x":
        return np.tile(np.sin(coord), (n, 1))
    if axis == "y":
        return np.tile(np.sin(coord)[:, None], (1, n))
    raise ValueError(axis)


def test_texture_direction_spectrum_keys_and_shape() -> None:
    Z = _stripes(64, freq=4, axis="x")
    res = texture_direction_spectrum(Z, angular_step=2)
    assert set(res.keys()) == {"angles_deg", "spectrum", "Std_deg"}
    assert res["angles_deg"].shape == res["spectrum"].shape
    assert res["angles_deg"][0] == 0
    assert res["angles_deg"][-1] == 178


def test_texture_direction_spectrum_recovers_signal_along_columns() -> None:
    # Z varies along axis 1 (columns) only → FFT energy along U-axis → theta ~ 0 deg.
    Z = _stripes(64, freq=6, axis="x")
    res = texture_direction_spectrum(Z, angular_step=2)
    assert res["Std_deg"] == 0.0 or abs(res["Std_deg"] - 178.0) <= 2.0


def test_texture_direction_spectrum_recovers_signal_along_rows() -> None:
    # Z varies along axis 0 (rows) only → FFT energy along V-axis → theta ~ 90 deg.
    Z = _stripes(64, freq=6, axis="y")
    res = texture_direction_spectrum(Z, angular_step=2)
    assert abs(res["Std_deg"] - 90.0) <= 2.0


def test_texture_direction_spectrum_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        texture_direction_spectrum(np.zeros(8), angular_step=2)


def test_texture_direction_spectrum_rejects_bad_step() -> None:
    Z = np.zeros((8, 8))
    with pytest.raises(ValueError):
        texture_direction_spectrum(Z, angular_step=0)
    with pytest.raises(ValueError):
        texture_direction_spectrum(Z, angular_step=180)


def test_surface_metrics_summary_keys_and_sq() -> None:
    rng = np.random.default_rng(seed=0)
    Z = rng.standard_normal((48, 48))
    Z -= Z.mean()
    summary = surface_metrics_summary(Z)
    assert set(summary.keys()) == {
        "Sq",
        "Sdq",
        "Sdr",
        "Ssk",
        "Sku",
        "Spc",
        "Spd",
        "Std_deg",
    }
    assert math.isclose(summary["Sq"], float(np.sqrt(np.mean(Z**2))), rel_tol=1e-12)
    assert summary["Sdq"] >= 0.0
    assert summary["Sdr"] >= 0.0
    assert summary["Spd"] >= 0.0


def test_surface_metrics_summary_flat_surface_is_safe() -> None:
    summary = surface_metrics_summary(np.zeros((16, 16)))
    assert summary["Sq"] == 0.0
    assert summary["Ssk"] == 0.0
    assert summary["Sku"] == 0.0
    assert summary["Sdq"] == 0.0
