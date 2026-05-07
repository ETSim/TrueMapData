"""Tests for CurvatureMapGenerator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from tmd.image.maps.curvature import CurvatureMapGenerator


@pytest.fixture
def height_small() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((24, 24), dtype=np.float32)


@pytest.fixture
def meta() -> dict:
    return {"x_length": 2.4, "y_length": 2.4}


@pytest.mark.parametrize(
    "visualization,expect_rgb",
    [
        ("grayscale", False),
        ("color", True),
        ("classified", True),
        ("edges", False),
        ("features", True),
        ("multi", True),
        ("divergent", True),
    ],
)
def test_curvature_visualization_modes(
    height_small: np.ndarray,
    meta: dict,
    visualization: str,
    expect_rgb: bool,
) -> None:
    gen = CurvatureMapGenerator(
        mode="mean",
        visualization=visualization,
        sigma=0.5,
        multi_scale=False,
    )
    out = gen.generate(height_small, metadata=meta)
    assert np.isfinite(out).all()
    if expect_rgb:
        assert out.ndim == 3 and out.shape[-1] == 3
    else:
        assert out.ndim == 2


def test_curvature_gaussian_multi_scale(height_small: np.ndarray, meta: dict) -> None:
    gen = CurvatureMapGenerator(
        mode="gaussian",
        visualization="grayscale",
        multi_scale=True,
        sigma_levels=[0.5, 1.0],
        sigma=0.5,
    )
    out = gen.generate(height_small, metadata=meta)
    assert out.shape == height_small.shape
    assert np.isfinite(out).all()
