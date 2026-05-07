"""Tests for :mod:`tmd.model.config` (dataclass configs)."""

from __future__ import annotations

import pytest

from tmd.model.config import ExportConfig, ModelConfig


def test_model_config_valid() -> None:
    c = ModelConfig()
    assert c.z_scale == 1.0


def test_model_config_invalid_z_scale() -> None:
    with pytest.raises(ValueError, match="z_scale"):
        ModelConfig(z_scale=0)


def test_model_config_invalid_coordinate_system() -> None:
    with pytest.raises(ValueError, match="coordinate_system"):
        ModelConfig(coordinate_system="invalid")


def test_model_config_from_dict_roundtrip() -> None:
    c = ModelConfig.from_dict({"x_length": 2.0, "y_length": 3.0, "unknown_key": 99})
    assert c.x_length == 2.0
    assert c.extra.get("unknown_key") == 99
    d = c.as_dict()
    assert d["x_length"] == 2.0


def test_export_config_validate_triangulation() -> None:
    ExportConfig(triangulation_method="quadtree", method="quadtree")


def test_export_config_invalid_error_threshold() -> None:
    with pytest.raises(ValueError, match="error_threshold"):
        ExportConfig(error_threshold=0)


def test_export_config_invalid_quad_sizes() -> None:
    with pytest.raises(ValueError, match="max_quad_size"):
        ExportConfig(min_quad_size=10, max_quad_size=5)
