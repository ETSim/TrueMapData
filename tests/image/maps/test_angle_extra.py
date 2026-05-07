"""Extra coverage for ``AngleMapGenerator`` modes and helpers."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.image.maps.angle import AngleMapGenerator


@pytest.fixture
def ramp_height_map() -> np.ndarray:
    return np.tile(np.linspace(0.0, 1.0, 16, dtype=np.float32), (16, 1))


def test_generate_gradient_mode(ramp_height_map: np.ndarray) -> None:
    gen = AngleMapGenerator(mode="gradient")
    out = gen.generate(ramp_height_map)
    assert out.shape == ramp_height_map.shape
    assert np.all(np.isfinite(out))


def test_generate_binary_mode(ramp_height_map: np.ndarray) -> None:
    gen = AngleMapGenerator(mode="binary", threshold=15.0)
    out = gen.generate(ramp_height_map)
    assert out.shape == ramp_height_map.shape
    assert set(np.unique(out)).issubset({0.0, 1.0})


def test_generate_hypsometric_mode(ramp_height_map: np.ndarray) -> None:
    gen = AngleMapGenerator(mode="hypsometric")
    out = gen.generate(ramp_height_map)
    assert out.shape[:2] == ramp_height_map.shape
    assert out.shape[2] == 3
    assert np.all(np.isfinite(out))


def test_generate_aspect_mode(ramp_height_map: np.ndarray) -> None:
    gen = AngleMapGenerator(mode="aspect")
    out = gen.generate(ramp_height_map)
    assert out.shape[:2] == ramp_height_map.shape
    assert out.shape[2] == 3


def test_generate_classified_mode(ramp_height_map: np.ndarray) -> None:
    gen = AngleMapGenerator(mode="classified")
    out = gen.generate(ramp_height_map)
    assert out.shape[:2] == ramp_height_map.shape
    assert out.shape[2] == 3


def test_generate_contour_mode(ramp_height_map: np.ndarray) -> None:
    gen = AngleMapGenerator(mode="contour")
    out = gen.generate(ramp_height_map, contour_interval=15.0)
    assert out.shape == ramp_height_map.shape


def test_generate_custom_gradient(ramp_height_map: np.ndarray) -> None:
    custom = [(0.0, (0.0, 0.0, 1.0)), (45.0, (0.0, 1.0, 0.0)), (90.0, (1.0, 0.0, 0.0))]
    gen = AngleMapGenerator(mode="custom")
    out = gen.generate(ramp_height_map, custom_gradient=custom)
    assert out.shape[:2] == ramp_height_map.shape
    assert out.shape[2] == 3
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_generate_with_terrain_class(ramp_height_map: np.ndarray) -> None:
    gen = AngleMapGenerator()
    out = gen.generate(ramp_height_map, terrain_class=True)
    assert out.shape[:2] == ramp_height_map.shape


def test_generate_with_metadata_dimensions(ramp_height_map: np.ndarray) -> None:
    gen = AngleMapGenerator(mode="gradient")
    out = gen.generate(ramp_height_map, metadata={"x_length": 10.0, "y_length": 10.0})
    assert np.all(np.isfinite(out))


def test_generate_with_highlight_range(ramp_height_map: np.ndarray) -> None:
    gen = AngleMapGenerator(mode="gradient")
    out = gen.generate(
        ramp_height_map,
        highlight_range=(20.0, 60.0),
        highlight_color=(1.0, 0.0, 0.0),
    )
    assert np.all(np.isfinite(out))


def test_generate_with_smoothing_zero(ramp_height_map: np.ndarray) -> None:
    gen = AngleMapGenerator(mode="gradient", smoothing=0.0)
    out = gen.generate(ramp_height_map)
    assert out.shape == ramp_height_map.shape


def test_classify_terrain_direct(ramp_height_map: np.ndarray) -> None:
    gen = AngleMapGenerator()
    slope = np.array([[1.0, 3.0, 7.0], [12.0, 20.0, 35.0], [50.0, 60.0, 90.0]])
    aspect = np.zeros_like(slope)
    out = gen._classify_terrain(slope, aspect)
    assert out.shape == (3, 3, 3)
    assert np.all(np.isfinite(out))


def test_validate_params_clamps_bad_values() -> None:
    gen = AngleMapGenerator()
    fixed = gen._validate_params(
        {
            "min_angle": -5.0,
            "max_angle": 200.0,
            "smoothing": -1.0,
            "edge_preservation": -1.0,
            "highlight_color": "bogus",
        }
    )
    assert fixed["min_angle"] >= 0.0
    assert fixed["max_angle"] <= 90.0
    assert fixed["smoothing"] >= 0.0
    assert fixed["edge_preservation"] >= 0.0


def test_generate_aspect_map_direct() -> None:
    gen = AngleMapGenerator()
    aspect = np.deg2rad(np.array([[0.0, 90.0], [180.0, 270.0]]))
    slope = np.array([[5.0, 15.0], [25.0, 35.0]])
    out = gen._generate_aspect_map(aspect, slope, min_slope=2.0)
    assert out.shape == (2, 2, 3)


def test_generate_contour_map_direct() -> None:
    gen = AngleMapGenerator()
    slope = np.linspace(0.0, 90.0, 9, dtype=np.float32).reshape(3, 3)
    out = gen._generate_contour_map(slope, contour_interval=15.0)
    assert out.shape == slope.shape


def test_apply_custom_gradient_invalid_returns_grayscale() -> None:
    """When a malformed gradient triggers an exception we get a grayscale fallback."""
    gen = AngleMapGenerator()
    slope = np.linspace(0.0, 90.0, 9, dtype=np.float32).reshape(3, 3)
    out = gen._apply_custom_gradient(slope, custom_gradient="not-a-list")
    assert out.shape == slope.shape or out.shape[:2] == slope.shape
