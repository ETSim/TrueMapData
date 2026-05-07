"""Tests for MapGenerator base class."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.image.maps.base_generator import MapGenerator


def test_map_generator_cannot_instantiate_directly() -> None:
    with pytest.raises(TypeError):
        MapGenerator()


def test_map_generator_subclass_get_params_merges_defaults() -> None:
    class DummyGenerator(MapGenerator):
        def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
            return height_map

        def _validate_params(self, params):
            params = dict(params)
            params.setdefault("x", 1)
            return params

    g = DummyGenerator(a=1, b=2)
    merged = g._get_params(b=3, c=4)
    assert merged["a"] == 1 and merged["b"] == 3 and merged["c"] == 4


def test_map_generator_prepare_height_map_delegates_to_core() -> None:
    class DummyGenerator(MapGenerator):
        def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
            return height_map

    g = DummyGenerator()
    h = np.linspace(0, 1, 12, dtype=np.float64).reshape(3, 4)
    out = g._prepare_height_map(h, normalize=False)
    assert out.shape == (3, 4)
    assert out.dtype == np.float32
