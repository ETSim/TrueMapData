"""Extra coverage for tmd.model.config (validators + ConfigManager)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tmd.model.config import ConfigManager, ExportConfig, ModelConfig


def test_model_config_invalid_base_height() -> None:
    with pytest.raises(ValueError, match="base_height"):
        ModelConfig(base_height=-0.1)


def test_model_config_invalid_x_length() -> None:
    with pytest.raises(ValueError, match="x_length"):
        ModelConfig(x_length=0)


def test_model_config_invalid_y_length() -> None:
    with pytest.raises(ValueError, match="y_length"):
        ModelConfig(y_length=-1.0)


def test_model_config_as_dict_strips_empty_extra() -> None:
    c = ModelConfig()
    d = c.as_dict()
    assert "extra" not in d


def test_model_config_as_dict_keeps_extra() -> None:
    c = ModelConfig.from_dict({"unknown": 42})
    d = c.as_dict()
    assert d["extra"] == {"unknown": 42}


def test_export_config_invalid_min_quad_size() -> None:
    with pytest.raises(ValueError, match="min_quad_size"):
        ExportConfig(min_quad_size=0)


def test_export_config_invalid_curvature_threshold() -> None:
    with pytest.raises(ValueError, match="curvature_threshold"):
        ExportConfig(curvature_threshold=0)


def test_export_config_invalid_max_triangles() -> None:
    with pytest.raises(ValueError, match="max_triangles"):
        ExportConfig(max_triangles=0)


def test_export_config_invalid_simplify_ratio() -> None:
    with pytest.raises(ValueError, match="simplify_ratio"):
        ExportConfig(simplify_ratio=2.0)


def test_export_config_invalid_smoothing() -> None:
    with pytest.raises(ValueError, match="smoothing"):
        ExportConfig(smoothing=2.0)


def test_export_config_invalid_max_subdivisions() -> None:
    with pytest.raises(ValueError, match="max_subdivisions"):
        ExportConfig(max_subdivisions=0)


def test_export_config_invalid_detail_boost() -> None:
    with pytest.raises(ValueError, match="detail_boost"):
        ExportConfig(detail_boost=-0.5)


def test_export_config_invalid_obj_units() -> None:
    with pytest.raises(ValueError, match="obj_units_to_mm"):
        ExportConfig(obj_units_to_mm=0)


def test_export_config_invalid_tmd_mm_per_pixel() -> None:
    with pytest.raises(ValueError, match="tmd_mm_per_pixel"):
        ExportConfig(tmd_mm_per_pixel=0)


def test_export_config_invalid_template_kind() -> None:
    with pytest.raises(ValueError, match="template_kind"):
        ExportConfig(template_kind="bogus")


def test_export_config_invalid_uv_alignment_mode() -> None:
    with pytest.raises(ValueError, match="uv_alignment_mode"):
        ExportConfig(uv_alignment_mode="bogus")


def test_export_config_invalid_triangulation_method() -> None:
    with pytest.raises(ValueError, match="triangulation_method"):
        ExportConfig(method="weird")


def test_config_manager_get_default_config_known_format() -> None:
    cfg = ConfigManager.get_default_config("stl")
    assert cfg.binary is True


def test_config_manager_get_default_config_unknown_format() -> None:
    cfg = ConfigManager.get_default_config("not_a_format")
    assert isinstance(cfg, ExportConfig)


def test_config_manager_create_config_with_overrides() -> None:
    cfg = ConfigManager.create_config(format_name="obj", x_length=10.0, custom_key="x")
    assert cfg.x_length == 10.0
    assert cfg.extra["custom_key"] == "x"


def test_config_manager_save_and_load_roundtrip(tmp_path: Path) -> None:
    cfg = ExportConfig(x_length=2.0, y_length=3.0)
    cfg.extra["custom"] = 42
    out = tmp_path / "config.json"
    ConfigManager.save_config(cfg, str(out))
    loaded = ConfigManager.load_config(str(out))
    assert loaded.x_length == 2.0
    assert loaded.y_length == 3.0
    assert loaded.extra.get("custom") == 42


def test_config_manager_load_config_invalid_path() -> None:
    with pytest.raises(IOError):
        ConfigManager.load_config("/path/that/does/not/exist.json")


def test_config_manager_save_config_invalid_path(tmp_path: Path) -> None:
    cfg = ExportConfig()
    bogus = tmp_path / "no" / "such" / "dir" / "x.json"
    with pytest.raises(IOError):
        ConfigManager.save_config(cfg, str(bogus))


def test_config_manager_merge_configs() -> None:
    base = ExportConfig(x_length=1.0, y_length=2.0)
    base.extra["custom"] = "base"
    merged = ConfigManager.merge_configs(base, {"y_length": 5.0, "extra_key": "added"})
    assert merged.x_length == 1.0
    assert merged.y_length == 5.0
    assert merged.extra.get("custom") == "base"
    assert merged.extra.get("extra_key") == "added"
