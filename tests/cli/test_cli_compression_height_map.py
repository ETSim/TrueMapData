"""Tests for ``compress_height_map`` in ``tmd.cli.compression``."""

from __future__ import annotations

import numpy as np

from tmd.cli.compression import compress_height_map


def test_compress_height_map_quantize_only() -> None:
    hm = np.linspace(0, 1, 100, dtype=np.float64).reshape(10, 10)
    out = compress_height_map(hm, mode="quantize", levels=8)
    assert out.shape == hm.shape


def test_compress_height_map_flat_quantize() -> None:
    hm = np.ones((4, 4), dtype=np.float32)
    out = compress_height_map(hm, mode="quantize", levels=16)
    np.testing.assert_array_equal(out, hm)


def test_compress_height_map_both_modes() -> None:
    hm = np.random.default_rng(0).random((16, 16)).astype(np.float32)
    out = compress_height_map(hm, mode="both", scale=0.5, levels=32, method="nearest")
    assert out.ndim == 2
