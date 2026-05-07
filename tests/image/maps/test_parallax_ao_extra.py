"""Extra coverage for ``ParallaxAOMapGenerator`` branches."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.image.maps.parallax_ao import ParallaxAOMapGenerator


@pytest.fixture
def small_ramp() -> np.ndarray:
    return np.tile(np.linspace(0.0, 1.0, 16, dtype=np.float32), (16, 1))


def test_generate_flat_returns_finite_array() -> None:
    gen = ParallaxAOMapGenerator(samples=8, multi_scale=False)
    out = gen.generate(np.zeros((16, 16), dtype=np.float32))
    assert out.shape == (16, 16)
    assert np.all(np.isfinite(out))


def test_generate_ramp_returns_finite_array(small_ramp: np.ndarray) -> None:
    gen = ParallaxAOMapGenerator(samples=8, multi_scale=False)
    out = gen.generate(small_ramp)
    assert out.shape == small_ramp.shape
    assert np.all(np.isfinite(out))


def test_generate_with_metadata_dimensions(small_ramp: np.ndarray) -> None:
    gen = ParallaxAOMapGenerator(samples=8, multi_scale=False)
    out = gen.generate(small_ramp, metadata={"x_length": 10.0, "y_length": 10.0})
    assert np.all(np.isfinite(out))


def test_generate_with_focus_range(small_ramp: np.ndarray) -> None:
    gen = ParallaxAOMapGenerator(samples=8, multi_scale=False, focus_range=(10.0, 60.0))
    out = gen.generate(small_ramp)
    assert np.all(np.isfinite(out))


def test_generate_with_directional_bias(small_ramp: np.ndarray) -> None:
    gen = ParallaxAOMapGenerator(samples=8, multi_scale=False, directional_bias=0.5, bias_direction=90.0)
    out = gen.generate(small_ramp)
    assert np.all(np.isfinite(out))


def test_generate_with_multi_scale(small_ramp: np.ndarray) -> None:
    gen = ParallaxAOMapGenerator(samples=8, multi_scale=True)
    out = gen.generate(small_ramp)
    assert np.all(np.isfinite(out))


def test_validate_params_clamps_invalid_values() -> None:
    gen = ParallaxAOMapGenerator()
    fixed = gen._validate_params(
        {
            "strength": -1.0,
            "samples": 2,
            "steep_threshold": -10.0,
            "steep_multiplier": 0.0,
            "slope_sensitivity": -1.0,
            "shadow_softness": 0.0,
            "max_distance": 0.0,
            "directional_bias": -0.1,
            "bias_direction": 720.0,
            "focus_range": (50, 30),
            "cavity_emphasis": -1.0,
        }
    )
    assert fixed["strength"] == 1.0
    assert fixed["samples"] == 16
    assert fixed["steep_threshold"] == 0.0
    assert fixed["steep_multiplier"] == 2.0
    assert fixed["slope_sensitivity"] == 0.0
    assert fixed["shadow_softness"] == 1.0
    assert fixed["max_distance"] == 0.05
    assert fixed["directional_bias"] == 0.0
    assert fixed["bias_direction"] == 0.0
    assert fixed["focus_range"] is None
    assert fixed["cavity_emphasis"] == 1.0


def test_validate_params_caps_high_values() -> None:
    gen = ParallaxAOMapGenerator()
    fixed = gen._validate_params(
        {
            "steep_threshold": 200.0,
            "slope_sensitivity": 5.0,
            "max_distance": 0.5,
            "directional_bias": 5.0,
        }
    )
    assert fixed["steep_threshold"] == 90.0
    assert fixed["slope_sensitivity"] == 1.0
    assert fixed["max_distance"] == 0.2
    assert fixed["directional_bias"] == 1.0


def test_validate_params_invalid_focus_range_format() -> None:
    gen = ParallaxAOMapGenerator()
    fixed = gen._validate_params({"focus_range": "not-a-tuple"})
    assert fixed["focus_range"] is None


def test_get_cell_size_xy() -> None:
    gen = ParallaxAOMapGenerator()
    cx, cy = gen._get_cell_size(np.zeros((8, 4), dtype=np.float32), {"x_length": 8.0, "y_length": 16.0})
    assert cx == pytest.approx(8.0 / 4)
    assert cy == pytest.approx(16.0 / 8)


def test_get_cell_size_mmpp() -> None:
    gen = ParallaxAOMapGenerator()
    cx, cy = gen._get_cell_size(np.zeros((4, 4), dtype=np.float32), {"mmpp": 0.05})
    assert cx == cy == 0.05


def test_get_cell_size_default() -> None:
    gen = ParallaxAOMapGenerator()
    cx, cy = gen._get_cell_size(np.zeros((4, 4), dtype=np.float32), {})
    assert cx == 1.0 and cy == 1.0


def test_generate_sample_points_count_and_unit_circle() -> None:
    gen = ParallaxAOMapGenerator()
    pts = gen._generate_sample_points(samples=8, directional_bias=0.0, bias_direction=0.0)
    assert len(pts) == 8
    radii = [np.hypot(x, y) for x, y in pts]
    assert all(abs(r - 1.0) < 1e-6 for r in radii)


def test_generate_sample_points_with_bias() -> None:
    gen = ParallaxAOMapGenerator()
    pts = gen._generate_sample_points(samples=8, directional_bias=0.7, bias_direction=45.0)
    assert len(pts) == 8


def test_sample_heights_returns_same_shape() -> None:
    gen = ParallaxAOMapGenerator()
    hm = np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8)
    radius_map = np.full((8, 8), 2, dtype=np.int32)
    out = gen._sample_heights(hm, dx=1.0, dy=0.0, radius_map=radius_map)
    assert out.shape == hm.shape


def test_calculate_radius_map_shape(small_ramp: np.ndarray) -> None:
    gen = ParallaxAOMapGenerator()
    terrain = gen._analyze_terrain(small_ramp, cell_size_x=1.0, cell_size_y=1.0)
    rm = gen._calculate_radius_map(terrain, dx_sample=1.0, dy_sample=0.0,
                                   base_radius=4, slope_sensitivity=0.5,
                                   focus_range=None)
    assert rm.shape == small_ramp.shape
    assert rm.dtype == np.int32
    assert rm.min() >= 1


def test_calculate_radius_map_with_focus_range(small_ramp: np.ndarray) -> None:
    gen = ParallaxAOMapGenerator()
    terrain = gen._analyze_terrain(small_ramp, cell_size_x=1.0, cell_size_y=1.0)
    rm = gen._calculate_radius_map(
        terrain,
        dx_sample=0.0,
        dy_sample=1.0,
        base_radius=4,
        slope_sensitivity=0.5,
        focus_range=(10.0, 80.0),
    )
    assert rm.shape == small_ramp.shape
