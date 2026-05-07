"""Tests for ``tmd.cli.utils.compression``."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.cli.utils import compression as comp


def test_calculate_compression_ratio() -> None:
    assert comp.calculate_compression_ratio(0, 100) == 0.0
    assert comp.calculate_compression_ratio(100, 50) == pytest.approx(50.0)


def test_quantize_height_map_flat_range_returns_same() -> None:
    hm = np.ones((3, 3), dtype=np.float32)
    out = comp.quantize_height_map(hm, levels=8)
    assert out is not None
    np.testing.assert_array_equal(out, hm)


def test_quantize_height_map_levels_clamped() -> None:
    hm = np.array([[0.0, 1.0], [0.25, 0.75]], dtype=np.float64)
    out = comp.quantize_height_map(hm, levels=1)
    assert out is not None


def test_perform_downsampling_nearest() -> None:
    hm = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = comp.perform_downsampling(hm, 2, 2, "nearest")
    assert out is not None
    assert out.shape == (2, 2)


def test_perform_downsampling_unknown_method_defaults_to_bilinear() -> None:
    hm = np.ones((6, 6), dtype=np.float32)
    out = comp.perform_downsampling(hm, 3, 3, "weird_method")
    assert out is not None


def test_perform_downsampling_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):
        raise RuntimeError("no scipy")

    monkeypatch.setattr("scipy.ndimage.zoom", boom)
    hm = np.ones((4, 4), dtype=np.float32)
    assert comp.perform_downsampling(hm, 2, 2, "nearest") is None
