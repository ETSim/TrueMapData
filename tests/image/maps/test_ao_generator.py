"""Tests for AOMapGenerator."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from tmd.image.maps.ao import AOMapGenerator


def test_ao_small_map_full_loop() -> None:
    rng = np.random.default_rng(0)
    h = rng.random((16, 16), dtype=np.float32)
    gen = AOMapGenerator(samples=8, strength=1.0)
    out = gen.generate(h)
    assert out.shape == (16, 16)
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_ao_large_map_uses_optimized_path() -> None:
    h = np.zeros((501, 501), dtype=np.float32)
    gen = AOMapGenerator(samples=4, strength=1.0)
    out = gen.generate(h)
    assert out.shape == h.shape
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_ao_validate_params_samples_and_strength() -> None:
    gen = AOMapGenerator(samples=0, strength=0.0)
    p = gen._get_params()
    assert p["samples"] == 16
    assert p["strength"] == 1.0


def test_ao_validate_params_non_int_samples() -> None:
    gen = AOMapGenerator()
    p = gen._get_params(samples="bad")
    assert p["samples"] == 16
