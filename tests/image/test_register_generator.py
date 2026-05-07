"""Tests for register_generator decorator."""

from __future__ import annotations

import numpy as np

from tmd.image.export.registry import MapRegistry, register_generator
from tmd.image.maps.base_generator import MapGenerator


def test_register_generator_with_aliases_registers_all_names() -> None:
    primary = "zzz_decorator_primary"
    alias = "zzz_decorator_alias"

    @register_generator(primary, aliases=[alias])
    class _DecoratedGen(MapGenerator):
        def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
            return height_map

    try:
        assert MapRegistry.get(primary) is _DecoratedGen
        assert MapRegistry.get(alias) is _DecoratedGen
    finally:
        MapRegistry._generators.pop(primary, None)
        MapRegistry._generators.pop(alias, None)
