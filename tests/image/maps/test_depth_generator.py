"""Tests for DepthMapGenerator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from tmd.image.maps.depth import DepthMapGenerator


@pytest.fixture
def ramp() -> np.ndarray:
    rows, cols = 12, 14
    jj, ii = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
    return (ii / max(rows - 1, 1) + jj / max(cols - 1, 1)).astype(np.float32) * 0.5


def test_depth_linear_grayscale(ramp: np.ndarray) -> None:
    gen = DepthMapGenerator(mode="linear", visualization="grayscale")
    out = gen.generate(ramp)
    assert out.shape == ramp.shape
    assert np.isfinite(out).all()


def test_depth_inverse_reverse_smoothing(ramp: np.ndarray) -> None:
    gen = DepthMapGenerator(mode="inverse", reverse=True, smoothing=0.5)
    out = gen.generate(ramp)
    assert out.shape == ramp.shape


def test_depth_focal_mode(ramp: np.ndarray) -> None:
    gen = DepthMapGenerator(mode="focal", focal_plane=0.5, focal_range=0.15)
    out = gen.generate(ramp)
    assert out.shape == ramp.shape


def test_depth_grayscale_enhance_contrast(ramp: np.ndarray) -> None:
    gen = DepthMapGenerator(visualization="grayscale", enhance_contrast=2.0)
    out = gen.generate(ramp)
    assert out.shape == ramp.shape


def test_depth_color_visualization_rgb(ramp: np.ndarray) -> None:
    gen = DepthMapGenerator(visualization="color", colormap="viridis")
    out = gen.generate(ramp)
    assert out.shape == (*ramp.shape, 3)
    assert np.isfinite(out).all()


def test_depth_heatmap_alias(ramp: np.ndarray) -> None:
    gen = DepthMapGenerator(visualization="heatmap", colormap="plasma")
    out = gen.generate(ramp)
    assert out.ndim == 3 and out.shape[-1] == 3


def test_depth_validate_invalid_mode_and_visualization(ramp: np.ndarray) -> None:
    gen = DepthMapGenerator(mode="bogus", visualization="bogus")
    out = gen.generate(ramp)
    assert out.shape == ramp.shape


def test_depth_validate_depth_range_and_focal(ramp: np.ndarray) -> None:
    gen = DepthMapGenerator(
        min_depth=1.0,
        max_depth=0.0,
        focal_plane=2.0,
        focal_range=-1.0,
        smoothing=-0.5,
        enhance_contrast=0.0,
    )
    p = gen._get_params()
    assert p["min_depth"] == 0.0 and p["max_depth"] == 1.0
    assert p["focal_plane"] == 0.5
    assert p["focal_range"] == 0.2
    assert p["smoothing"] == 0.0
    assert p["enhance_contrast"] == 1.0
