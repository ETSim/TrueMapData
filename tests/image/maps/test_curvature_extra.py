"""Extra coverage for ``CurvatureMapGenerator`` modes/visualizations/validation."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.image.maps.curvature import CurvatureMapGenerator


@pytest.fixture
def smooth_height_map() -> np.ndarray:
    """Smoothly varying height map (cosine bump)."""
    y, x = np.mgrid[0:16, 0:16]
    return (np.cos(x / 4.0) * np.cos(y / 4.0)).astype(np.float32)


def test_generate_grayscale(smooth_height_map: np.ndarray) -> None:
    gen = CurvatureMapGenerator(visualization="grayscale")
    out = gen.generate(smooth_height_map)
    assert out.shape == smooth_height_map.shape
    assert out.dtype == np.float32 or out.dtype == np.float64


def test_generate_color(smooth_height_map: np.ndarray) -> None:
    gen = CurvatureMapGenerator(visualization="color")
    out = gen.generate(smooth_height_map)
    assert out.shape[:2] == smooth_height_map.shape
    assert out.shape[2] == 3


def test_generate_classified(smooth_height_map: np.ndarray) -> None:
    gen = CurvatureMapGenerator(visualization="classified")
    out = gen.generate(smooth_height_map)
    assert out.shape[:2] == smooth_height_map.shape


def test_generate_edges(smooth_height_map: np.ndarray) -> None:
    gen = CurvatureMapGenerator(visualization="edges")
    out = gen.generate(smooth_height_map)
    assert out.shape[:2] == smooth_height_map.shape


def test_generate_features(smooth_height_map: np.ndarray) -> None:
    gen = CurvatureMapGenerator(visualization="features")
    out = gen.generate(smooth_height_map)
    assert out.shape[:2] == smooth_height_map.shape
    assert out.shape[2] == 3


def test_generate_multi(smooth_height_map: np.ndarray) -> None:
    gen = CurvatureMapGenerator(visualization="multi")
    out = gen.generate(smooth_height_map)
    assert out.shape[:2] == smooth_height_map.shape


def test_generate_divergent(smooth_height_map: np.ndarray) -> None:
    gen = CurvatureMapGenerator(visualization="divergent")
    out = gen.generate(smooth_height_map)
    assert out.shape[:2] == smooth_height_map.shape


def test_generate_unknown_visualization_falls_back(smooth_height_map: np.ndarray) -> None:
    gen = CurvatureMapGenerator(visualization="bogus_mode")
    out = gen.generate(smooth_height_map)
    # Falls back to grayscale (2D) array
    assert out.shape == smooth_height_map.shape


def test_generate_with_metadata_dimensions(smooth_height_map: np.ndarray) -> None:
    gen = CurvatureMapGenerator()
    out = gen.generate(smooth_height_map, metadata={"x_length": 10.0, "y_length": 10.0})
    assert out.shape == smooth_height_map.shape


def test_generate_with_multi_scale(smooth_height_map: np.ndarray) -> None:
    gen = CurvatureMapGenerator(multi_scale=True, sigma_levels=[0.5, 1.5])
    out = gen.generate(smooth_height_map)
    assert out.shape == smooth_height_map.shape


@pytest.mark.parametrize("mode", ["mean", "gaussian", "maximal", "minimal", "profile", "planform"])
def test_calculate_curvature_modes(smooth_height_map: np.ndarray, mode: str) -> None:
    gen = CurvatureMapGenerator()
    out = gen._calculate_curvature(smooth_height_map, mode, 1.0, 1.0, 1.0)
    assert out.shape == smooth_height_map.shape
    assert np.all(np.isfinite(out))


def test_create_classified_visualization_direct() -> None:
    gen = CurvatureMapGenerator()
    height = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    curv = np.linspace(-0.5, 0.5, 16, dtype=np.float32).reshape(4, 4)
    out = gen._create_classified_visualization(curv, height, percentile_clip=(2, 98))
    assert out.shape[:2] == curv.shape
    assert out.shape[2] == 3


def test_create_edge_visualization_direct() -> None:
    gen = CurvatureMapGenerator()
    curv = np.zeros((8, 8), dtype=np.float32)
    curv[3, 3] = 5.0
    out = gen._create_edge_visualization(curv, percentile_clip=(2, 98), edge_width=1)
    assert out.shape[:2] == curv.shape


def test_create_feature_visualization_direct() -> None:
    gen = CurvatureMapGenerator()
    curv = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    out = gen._create_feature_visualization(
        curv, feature_threshold=0.1, feature_colors=((0.0, 0.5, 1.0), (1.0, 0.5, 0.0))
    )
    assert out.shape == (4, 4, 3)


def test_validate_params_handles_invalid_values() -> None:
    gen = CurvatureMapGenerator()
    fixed = gen._validate_params(
        {
            "scale": -1.0,
            "sigma": -1.0,
            "feature_threshold": -1.0,
            "edge_width": -1,
            "enhance_contrast": -1.0,
        }
    )
    assert fixed["scale"] >= 0.0
    assert fixed["sigma"] >= 0.0
    assert fixed["feature_threshold"] >= 0.0
    assert fixed["edge_width"] >= 0
    assert fixed["enhance_contrast"] >= 0.0


def test_get_cell_size_branches() -> None:
    gen = CurvatureMapGenerator()
    arr = np.zeros((4, 4), dtype=np.float32)
    cx, cy = gen._get_cell_size(arr, {"x_length": 8.0, "y_length": 16.0})
    assert cx == 2.0 and cy == 4.0
    cx, cy = gen._get_cell_size(arr, {"mmpp": 0.05})
    assert cx == cy == 0.05
    cx, cy = gen._get_cell_size(arr, {})
    assert cx == 1.0 and cy == 1.0


def test_generate_handles_inner_error(monkeypatch: pytest.MonkeyPatch, smooth_height_map: np.ndarray) -> None:
    """If _calculate_curvature raises, generate falls back to gray image."""

    def _boom(*_a, **_kw):
        raise RuntimeError("boom")

    gen = CurvatureMapGenerator(visualization="grayscale")
    monkeypatch.setattr(gen, "_calculate_curvature", _boom)
    out = gen.generate(smooth_height_map)
    assert out.shape == smooth_height_map.shape
    assert np.allclose(out, 0.5)


def test_calculate_curvature_inner_error_returns_zeros(
    monkeypatch: pytest.MonkeyPatch, smooth_height_map: np.ndarray
) -> None:
    """_calculate_curvature catches its own exceptions and returns zeros."""

    def _boom(*_a, **_kw):
        raise RuntimeError("simulated gradient failure")

    monkeypatch.setattr("tmd.image.maps.curvature.np.gradient", _boom)
    gen = CurvatureMapGenerator()
    out = gen._calculate_curvature(smooth_height_map, "mean", 1.0, 1.0, 1.0)
    assert out.shape == smooth_height_map.shape
    assert np.all(out == 0.0)


def test_create_color_visualization_falls_back_to_manual_when_cmap_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If matplotlib's get_cmap raises, the function uses the manual fallback."""
    gen = CurvatureMapGenerator()

    import matplotlib.pyplot as plt

    def _bad_cmap(_name):
        raise ValueError("missing colormap")

    monkeypatch.setattr(plt, "get_cmap", _bad_cmap)

    curv = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    out = gen._create_color_visualization(curv, (2, 98), 1.0, "bogus")
    assert out.shape == (4, 4, 3)


def test_create_grayscale_visualization_with_contrast() -> None:
    gen = CurvatureMapGenerator()
    curv = np.linspace(-1.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    out = gen._create_grayscale_visualization(curv, (2, 98), enhance_contrast=2.0)
    assert out.shape == curv.shape
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_create_edge_visualization_with_thicker_edges() -> None:
    gen = CurvatureMapGenerator()
    curv = np.zeros((8, 8), dtype=np.float32)
    curv[3, :] = 5.0
    curv[:, 3] = -5.0
    out = gen._create_edge_visualization(curv, (2, 98), edge_width=3)
    assert out.shape == curv.shape


def test_create_divergent_with_only_positive_curvature() -> None:
    gen = CurvatureMapGenerator()
    curv = np.abs(np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4))
    out = gen._create_divergent_visualization(curv, (2, 98), 1.5)
    assert out.shape == (4, 4, 3)


def test_create_divergent_with_only_negative_curvature() -> None:
    gen = CurvatureMapGenerator()
    curv = -np.abs(np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4))
    out = gen._create_divergent_visualization(curv, (2, 98), 1.0)
    assert out.shape == (4, 4, 3)


def test_validate_params_invalid_sigma_levels() -> None:
    gen = CurvatureMapGenerator()
    fixed = gen._validate_params({"sigma_levels": [-1, -2]})
    assert fixed["sigma_levels"] == [0.7, 1.5, 3.0]
    fixed = gen._validate_params({"sigma_levels": "abc"})
    assert fixed["sigma_levels"] == [0.7, 1.5, 3.0]


def test_validate_params_invalid_feature_colors() -> None:
    gen = CurvatureMapGenerator()
    fixed = gen._validate_params({"feature_colors": ((2.0, 0.0, 0.0), (0.0, 0.0, 0.0))})
    assert fixed["feature_colors"] == ((0.0, 0.7, 1.0), (1.0, 0.4, 0.0))
    fixed = gen._validate_params({"feature_colors": "garbage"})
    assert fixed["feature_colors"] == ((0.0, 0.7, 1.0), (1.0, 0.4, 0.0))


def test_validate_params_invalid_percentile_clip() -> None:
    gen = CurvatureMapGenerator()
    fixed = gen._validate_params({"percentile_clip": (-5, 200)})
    assert fixed["percentile_clip"] == (2, 98)
    fixed = gen._validate_params({"percentile_clip": "garbage"})
    assert fixed["percentile_clip"] == (2, 98)


def test_validate_params_invalid_modes_default() -> None:
    gen = CurvatureMapGenerator()
    fixed = gen._validate_params({"mode": "weird", "visualization": "weird"})
    assert fixed["mode"] == "mean"
    assert fixed["visualization"] == "grayscale"


def test_validate_params_caps_edge_width() -> None:
    gen = CurvatureMapGenerator()
    fixed = gen._validate_params({"edge_width": 100})
    assert fixed["edge_width"] == 5
