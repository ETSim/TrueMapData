"""Tests for DisplacementMapGenerator."""

from __future__ import annotations

import numpy as np

from tmd.image.maps.displacement import DisplacementMapGenerator


def test_displacement_intensity_and_no_invert() -> None:
    h = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    base = DisplacementMapGenerator(intensity=2.0, invert=False).generate(h)
    assert base.shape == h.shape


def test_displacement_invert_true() -> None:
    h = np.full((3, 3), 0.5, dtype=np.float32)
    out = DisplacementMapGenerator(intensity=1.0, invert=True).generate(h)
    no_inv = DisplacementMapGenerator(intensity=1.0, invert=False).generate(h)
    np.testing.assert_allclose(out, 1.0 - no_inv)


def test_displacement_validate_intensity_zero_resets() -> None:
    gen = DisplacementMapGenerator(intensity=0.0)
    p = gen._get_params()
    assert p["intensity"] == 1.0
