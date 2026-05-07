"""Tests for AngleMapGenerator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from tmd.image.maps.angle import AngleMapGenerator


@pytest.fixture
def height_small() -> np.ndarray:
    rng = np.random.default_rng(7)
    h = rng.random((20, 20), dtype=np.float32)
    return h


@pytest.fixture
def meta() -> dict:
    return {"x_length": 1.0, "y_length": 1.0}


def test_angle_mode_gradient(height_small: np.ndarray, meta: dict) -> None:
    gen = AngleMapGenerator(mode="gradient", smoothing=0.0)
    out = gen.generate(height_small, metadata=meta)
    assert out.shape == height_small.shape
    assert np.isfinite(out).all()


def test_angle_mode_binary(height_small: np.ndarray, meta: dict) -> None:
    gen = AngleMapGenerator(mode="binary", smoothing=0.0)
    out = gen.generate(height_small, metadata=meta, threshold=15.0)
    assert out.shape == height_small.shape


def test_angle_mode_hypsometric(height_small: np.ndarray, meta: dict) -> None:
    gen = AngleMapGenerator(mode="hypsometric", smoothing=0.0)
    out = gen.generate(height_small, metadata=meta)
    assert out.ndim == 3 and out.shape[-1] == 3


def test_angle_mode_aspect(height_small: np.ndarray, meta: dict) -> None:
    gen = AngleMapGenerator(mode="aspect", smoothing=0.0)
    out = gen.generate(height_small, metadata=meta)
    assert out.ndim == 3 and out.shape[-1] == 3


def test_angle_mode_classified(height_small: np.ndarray, meta: dict) -> None:
    gen = AngleMapGenerator(mode="classified", smoothing=0.0)
    out = gen.generate(height_small, metadata=meta)
    assert np.isfinite(out).all()
    assert out.shape == (*height_small.shape, 3)


def test_angle_mode_contour(height_small: np.ndarray, meta: dict) -> None:
    gen = AngleMapGenerator(mode="contour", smoothing=0.0)
    out = gen.generate(height_small, metadata=meta, contour_interval=5.0)
    assert np.isfinite(out).all()


def test_angle_mode_custom_gradient(height_small: np.ndarray, meta: dict) -> None:
    stops = [
        (0.0, (0.0, 0.0, 1.0)),
        (0.5, (0.0, 1.0, 0.0)),
        (1.0, (1.0, 0.0, 0.0)),
    ]
    gen = AngleMapGenerator(mode="custom", custom_gradient=stops, smoothing=0.0)
    out = gen.generate(height_small, metadata=meta)
    assert np.isfinite(out).all()
