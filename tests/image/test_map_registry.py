"""Tests for MapRegistry after tmd.image package registration."""

from __future__ import annotations

import numpy as np

import tmd.image  # noqa: F401 — triggers MapRegistry.register(...) in package __init__
from tmd.image import MapRegistry
from tmd.image.maps.base_generator import MapGenerator
from tmd.image.maps.normal import NormalMapGenerator


def test_map_registry_resolves_normal() -> None:
    cls = MapRegistry.get("normal")
    assert cls is not None
    assert cls is NormalMapGenerator


def test_map_registry_case_insensitive_lookup() -> None:
    assert MapRegistry.get("NORMAL") is NormalMapGenerator


def test_map_registry_lists_core_generators() -> None:
    names = set(MapRegistry.list())
    expected = {
        "ao",
        "normal",
        "bump",
        "roughness",
        "metallic",
        "displacement",
        "height",
        "hillshade",
        "curvature",
        "angle",
        "parallax_ao",
        "depth",
    }
    assert expected <= names


def test_map_registry_unknown_returns_none() -> None:
    assert MapRegistry.get("not_a_registered_map_type_xyz") is None


class _StubMapGenerator(MapGenerator):
    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        return height_map


def test_register_and_retrieve_custom_generator() -> None:
    key = "zzz_unit_test_map_registry_stub"
    MapRegistry.register(key, _StubMapGenerator)
    try:
        assert MapRegistry.get(key) is _StubMapGenerator
    finally:
        MapRegistry._generators.pop(key, None)
