"""Tests for HillshadeMapGenerator."""

from __future__ import annotations

import builtins

import numpy as np
import pytest

pytest.importorskip("scipy")

from tmd.image.maps.hillshade import HillshadeMapGenerator


@pytest.fixture
def sample_height() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.random((24, 32), dtype=np.float32)


def test_hillshade_shape_range_with_metadata(sample_height: np.ndarray) -> None:
    meta = {"x_length": 3.2, "y_length": 2.4}
    out = HillshadeMapGenerator().generate(sample_height, metadata=meta)
    assert out.shape == sample_height.shape
    assert np.isfinite(out).all()
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_hillshade_without_metadata(sample_height: np.ndarray) -> None:
    out = HillshadeMapGenerator(azimuth=90.0, altitude=30.0).generate(sample_height)
    assert out.shape == sample_height.shape
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_generate_multi_hillshade_default_keys(sample_height: np.ndarray) -> None:
    multi = HillshadeMapGenerator.generate_multi_hillshade(sample_height)
    assert set(multi.keys()) == {"nw", "ne", "sw", "se"}
    for _name, hs in multi.items():
        assert hs.shape == sample_height.shape


def test_blend_hillshades_dict_and_list(sample_height: np.ndarray) -> None:
    multi = HillshadeMapGenerator.generate_multi_hillshade(sample_height)
    blended = HillshadeMapGenerator.blend_hillshades(multi)
    assert blended is not None
    assert blended.shape == sample_height.shape
    lst = list(multi.values())
    b2 = HillshadeMapGenerator.blend_hillshades(lst, weights=[0.25, 0.25, 0.25, 0.25])
    assert b2.shape == sample_height.shape


def test_blend_hillshades_empty_returns_none() -> None:
    assert HillshadeMapGenerator.blend_hillshades([]) is None


def test_hillshade_numpy_gradient_fallback(monkeypatch: pytest.MonkeyPatch, sample_height: np.ndarray) -> None:
    orig_import = builtins.__import__

    def guarded_import(name: str, globals_=None, locals_=None, fromlist=(), level: int = 0):
        if name == "scipy":
            raise ImportError("scipy blocked for fallback test")
        return orig_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    out = HillshadeMapGenerator().generate(sample_height, metadata={"x_length": 1.0, "y_length": 1.0})
    assert out.shape == sample_height.shape
    assert np.isfinite(out).all()


def test_validation_azimuth_altitude_zfactor() -> None:
    gen = HillshadeMapGenerator(azimuth=400.0, altitude=-10.0, z_factor=0.0)
    p = gen._get_params()
    assert p["azimuth"] == pytest.approx(40.0)
    assert p["altitude"] == pytest.approx(0.0)
    assert p["z_factor"] == 1.0
