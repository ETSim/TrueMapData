"""Tests for RGBDMapGenerator."""

from __future__ import annotations

import numpy as np

from tmd.image.maps.rgbd import RGBDMapGenerator


def test_rgbd_height_source_combined_channels() -> None:
    h = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    gen = RGBDMapGenerator(color_source="height", depth_scale=1.0)
    out = gen.generate(h)
    assert out.shape == (3, 4, 4)
    np.testing.assert_allclose(out[..., 0], out[..., 1])
    np.testing.assert_allclose(out[..., 1], out[..., 2])
    np.testing.assert_allclose(out[..., 3], h)


def test_rgbd_non_height_hits_exception_fallback() -> None:
    """Non-height placeholder uses 2D rgb; concatenate fails and returns zeros_like(height_map)."""
    h = np.ones((4, 5), dtype=np.float32) * 0.5
    out = RGBDMapGenerator(color_source="normal").generate(h, depth_scale=2.0)
    assert out.shape == h.shape
    assert np.all(out == 0.0)


def test_rgbd_validate_depth_scale() -> None:
    gen = RGBDMapGenerator(depth_scale=0.0)
    p = gen._get_params()
    assert p["depth_scale"] == 1.0
