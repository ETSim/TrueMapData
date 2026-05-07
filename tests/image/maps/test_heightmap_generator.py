"""Tests for HeightMapGenerator."""

from __future__ import annotations

import numpy as np

from tmd.image.maps.heightmap import HeightMapGenerator


def test_height_map_generator_pass_through() -> None:
    h = np.array([[0.0, 0.5], [0.25, 1.0]], dtype=np.float32)
    out = HeightMapGenerator(invert=False).generate(h)
    assert out.shape == h.shape
    assert np.all(out >= 0) and np.all(out <= 1)


def test_height_map_generator_invert() -> None:
    h = np.array([[0.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    gen = HeightMapGenerator(invert=True)
    out = gen.generate(h)
    base = HeightMapGenerator(invert=False).generate(h)
    np.testing.assert_allclose(out, 1.0 - base)
