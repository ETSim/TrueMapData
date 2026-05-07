"""Tests for BumpMapGenerator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from tmd.image.maps.bump import BumpMapGenerator


def test_bump_map_sloped_ramp_shape_and_range() -> None:
    rows, cols = 16, 16
    jj, ii = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
    h = (ii + jj).astype(np.float32)
    gen = BumpMapGenerator(strength=1.0, blur_radius=1.0)
    out = gen.generate(h)
    assert out.shape == (rows, cols)
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_bump_map_flat_surface_all_zeros() -> None:
    h = np.full((12, 14), 0.5, dtype=np.float32)
    out = BumpMapGenerator().generate(h)
    assert out.shape == h.shape
    np.testing.assert_allclose(out, 0.0)


def test_bump_map_validate_params_resets_strength_and_blur() -> None:
    gen = BumpMapGenerator(strength=-1.0, blur_radius=-0.5)
    p = gen._get_params()
    assert p["strength"] == 1.0
    assert p["blur_radius"] == 0.0
