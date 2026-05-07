"""Tests for MetallicMapGenerator."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.image.maps.metallic import MetallicMapGenerator


@pytest.fixture
def h() -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.random((20, 22), dtype=np.float32)


def test_metallic_constant(h: np.ndarray) -> None:
    out = MetallicMapGenerator(method="constant", value=0.35).generate(h)
    np.testing.assert_allclose(out, 0.35)


def test_metallic_height_threshold(h: np.ndarray) -> None:
    out = MetallicMapGenerator(method="height_threshold", threshold=0.5).generate(h)
    assert out.shape == h.shape
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_metallic_gradient(h: np.ndarray) -> None:
    out = MetallicMapGenerator(method="gradient").generate(h)
    assert out.shape == h.shape


def test_metallic_pattern_grid_checker_noise(h: np.ndarray) -> None:
    for ptype in ("grid", "checker", "noise"):
        out = MetallicMapGenerator(method="pattern", pattern_type=ptype, pattern_scale=2.0).generate(h)
        assert out.shape == h.shape
        assert np.isfinite(out).all()


def test_metallic_unknown_method_zeros(h: np.ndarray) -> None:
    out = MetallicMapGenerator(method="unknown_xyz").generate(h)
    assert np.all(out == 0.0)


def test_metallic_validate_value_threshold_clamp() -> None:
    gen = MetallicMapGenerator(value=2.0, threshold=-0.5)
    p = gen._get_params()
    assert p["value"] == 1.0
    assert p["threshold"] == 0.0
