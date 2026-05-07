"""Tests for ParallaxAOMapGenerator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from tmd.image.maps.parallax_ao import ParallaxAOMapGenerator


def test_parallax_ao_small_map_happy_path() -> None:
    rows, cols = 32, 32
    jj, ii = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
    h = (ii / rows + jj / cols).astype(np.float32)
    meta = {"x_length": 1.0, "y_length": 1.0}
    gen = ParallaxAOMapGenerator(
        samples=8,
        multi_scale=False,
        cavity_emphasis=0.0,
        directional_bias=0.0,
    )
    out = gen.generate(h, metadata=meta)
    assert out.shape == (rows, cols)
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()
