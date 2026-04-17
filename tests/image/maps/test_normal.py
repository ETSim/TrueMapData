"""Tests for normal map generation (edge-safe gradients)."""

import numpy as np
import pytest

from tmd.image.maps.normal import NormalMapGenerator


def _normals_from_gradients(dx_array: np.ndarray, dy_array: np.ndarray, strength: float = 1.0) -> np.ndarray:
    dx_array = dx_array * strength
    dy_array = dy_array * strength
    normal_map = np.empty((*dx_array.shape, 3), dtype=np.float32)
    normal_map[..., 0] = -dx_array
    normal_map[..., 1] = -dy_array
    normal_map[..., 2] = 1.0
    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    norm = np.maximum(norm, 1e-10)
    normal_map = np.divide(normal_map, norm)
    return (normal_map + 1.0) * 0.5


@pytest.fixture
def generator() -> NormalMapGenerator:
    return NormalMapGenerator(strength=1.0, normalize=False)


def test_normal_map_shape_range_finite(generator: NormalMapGenerator) -> None:
    rng = np.random.default_rng(42)
    h = rng.random((32, 48), dtype=np.float32)
    meta = {"x_length": 4.8, "y_length": 3.2}
    out = generator.generate(h, metadata=meta)
    assert out.shape == (32, 48, 3)
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_flat_surface_is_up_normal(generator: NormalMapGenerator) -> None:
    h = np.full((16, 24), 0.25, dtype=np.float32)
    out = generator.generate(h, metadata={"x_length": 1.0, "y_length": 1.0})
    expected = np.full((16, 24, 3), 0.5, dtype=np.float32)
    expected[..., 2] = 1.0
    np.testing.assert_allclose(out, expected, atol=1e-5, rtol=0.0)


def test_matches_manual_sobel_reference(generator: NormalMapGenerator) -> None:
    pytest.importorskip("scipy")
    from scipy import ndimage

    rows, cols = 40, 36
    jj, ii = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
    h = np.sin(0.18 * jj) + np.cos(0.12 * ii)
    h = h.astype(np.float32)

    cell_size_x = 0.11
    cell_size_y = 0.09
    meta = {"x_length": cell_size_x * cols, "y_length": cell_size_y * rows}

    out = generator.generate(h, metadata=meta)

    gx = ndimage.sobel(h, axis=1, mode="reflect") / (8.0 * cell_size_x)
    gy = ndimage.sobel(h, axis=0, mode="reflect") / (8.0 * cell_size_y)
    ref = _normals_from_gradients(gx.astype(np.float32), gy.astype(np.float32), strength=1.0)

    np.testing.assert_allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_rim_not_spuriously_far_from_np_gradient(generator: NormalMapGenerator) -> None:
    """Rim should stay close to a numpy-gradient reference (guards one-sided edge stencil regressions)."""
    rows, cols = 48, 52
    jj, ii = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
    h = 0.08 * np.sin(0.09 * jj) * np.cos(0.11 * ii) + 0.02 * (ii + jj) / (rows + cols)
    h = h.astype(np.float32)

    cell_size_x = 0.05
    cell_size_y = 0.05
    meta = {"x_length": cell_size_x * cols, "y_length": cell_size_y * rows}

    out = generator.generate(h, metadata=meta)

    gy, gx = np.gradient(h, cell_size_y, cell_size_x)
    ref = _normals_from_gradients(gx.astype(np.float32), gy.astype(np.float32), strength=1.0)

    rim = np.zeros_like(h, dtype=bool)
    rim[0, :] = rim[-1, :] = True
    rim[:, 0] = rim[:, -1] = True
    delta = np.linalg.norm(out - ref, axis=2)
    rim_mean = float(delta[rim].mean())
    interior = delta[~rim].mean()
    # Rim used to be much larger than interior under the old hybrid stencil.
    assert rim_mean <= interior * 2.5 + 0.02


def test_tiny_map_returns_flat(generator: NormalMapGenerator) -> None:
    h = np.array([[0.0, 1.0]], dtype=np.float32)
    out = generator.generate(h, metadata={"x_length": 1.0, "y_length": 1.0})
    assert out.shape == (1, 2, 3)
    np.testing.assert_allclose(out[..., 2], 1.0, atol=1e-5)
