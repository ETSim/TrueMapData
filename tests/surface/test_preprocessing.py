"""Tests for `tmd.surface.preprocessing` helpers."""

from __future__ import annotations

import math

import numpy as np

from tmd.surface.preprocessing import downsample_to_max_dim, normalize_to_unit_sq


def test_downsample_to_max_dim_no_op_when_already_small() -> None:
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    out = downsample_to_max_dim(arr, max_dim=16)
    assert out.shape == arr.shape
    assert out.dtype == np.float64
    np.testing.assert_allclose(out, arr.astype(np.float64))


def test_downsample_to_max_dim_shrinks_longest_axis() -> None:
    arr = np.linspace(0.0, 1.0, 128 * 64, dtype=np.float64).reshape(128, 64)
    out = downsample_to_max_dim(arr, max_dim=32)
    assert max(out.shape) == 32
    assert out.dtype == np.float64


def test_downsample_to_max_dim_disables_on_non_positive() -> None:
    arr = np.zeros((64, 64), dtype=np.float64)
    assert downsample_to_max_dim(arr, max_dim=0).shape == arr.shape
    assert downsample_to_max_dim(arr, max_dim=-5).shape == arr.shape
    assert downsample_to_max_dim(arr, max_dim=None).shape == arr.shape  # type: ignore[arg-type]


def test_normalize_to_unit_sq_centers_and_scales() -> None:
    rng = np.random.default_rng(seed=1)
    arr = 7.5 + 3.2 * rng.standard_normal((32, 32))
    normalized, original_sq = normalize_to_unit_sq(arr)
    assert math.isclose(float(np.mean(normalized)), 0.0, abs_tol=1e-12)
    assert math.isclose(float(np.sqrt(np.mean(normalized**2))), 1.0, rel_tol=1e-9)
    expected_sq = float(np.sqrt(np.mean((arr - arr.mean()) ** 2)))
    assert math.isclose(original_sq, expected_sq, rel_tol=1e-12)


def test_normalize_to_unit_sq_flat_input_returns_centered_and_unit_scale() -> None:
    arr = np.full((8, 8), 2.0)
    normalized, original_sq = normalize_to_unit_sq(arr)
    np.testing.assert_allclose(normalized, np.zeros_like(arr))
    assert original_sq == 1.0
