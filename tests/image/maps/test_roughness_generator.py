"""Tests for RoughnessMapGenerator."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from tmd.image.maps.roughness import RoughnessMapGenerator


def test_roughness_sloped_input_normalized() -> None:
    rows, cols = 16, 16
    jj, ii = np.meshgrid(np.arange(cols, dtype=np.float32), np.arange(rows, dtype=np.float32))
    h = (ii * 0.1 + jj * 0.07).astype(np.float32)
    out = RoughnessMapGenerator(kernel_size=3, scale=1.0).generate(h)
    assert out.shape == (rows, cols)
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_roughness_flat_input_all_zeros() -> None:
    h = np.ones((8, 8), dtype=np.float32) * 0.3
    out = RoughnessMapGenerator().generate(h)
    assert np.all(out == 0.0)


def test_roughness_validate_even_kernel_and_bad_scale() -> None:
    gen = RoughnessMapGenerator(kernel_size=4, scale=0.0)
    p = gen._get_params()
    assert p["kernel_size"] % 2 == 1
    assert p["kernel_size"] >= 3
    assert p["scale"] == 1.0


def test_roughness_opencv_import_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force numpy gradient path when cv2 cannot be imported."""
    orig_import = builtins.__import__

    def guarded_import(name: str, globals_=None, locals_=None, fromlist=(), level: int = 0):
        if name == "cv2":
            raise ImportError("cv2 blocked")
        return orig_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    h = np.random.default_rng(3).random((10, 12), dtype=np.float32)
    out = RoughnessMapGenerator(kernel_size=3, scale=1.5).generate(h)
    assert out.shape == h.shape
