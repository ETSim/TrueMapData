"""Tests for MapExporter material maps and tiling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tmd.image  # noqa: F401 — register MapRegistry entries
from tmd.image.export.exporter import MapExporter


def test_export_material_binding_maps_writes_four_pngs(tmp_path) -> None:
    h = np.linspace(0, 1, 16 * 16, dtype=np.float32).reshape(16, 16)
    out_dir = tmp_path / "maps"
    out_dir.mkdir()
    slots = MapExporter.export_material_binding_maps(
        h, str(out_dir), "stem", compress=75, normalize=True
    )
    assert set(slots.keys()) == {"map_kd", "map_bump", "map_disp", "map_pr"}
    for path in slots.values():
        assert path.endswith(".png")
        assert Path(path).is_file()


def test_export_map_invalid_type_returns_none(tmp_path) -> None:
    h = np.ones((8, 8), dtype=np.float32)
    out = tmp_path / "x.png"
    result = MapExporter.export_map(h, str(out), "not_a_real_map_type_zzz")
    assert result is None


def test_export_material_binding_maps_with_physical_tiling_happy_path(tmp_path) -> None:
    h = np.random.default_rng(0).random((12, 12), dtype=np.float32)
    out_dir = tmp_path / "tiled"
    slots = MapExporter.export_material_binding_maps_with_physical_tiling(
        h,
        str(out_dir),
        "atlas",
        tile_size_px=(6, 6),
        target_size_px=(10, 8),
        compress=75,
        normalize=True,
    )
    assert len(slots) == 4
    for path in slots.values():
        assert Path(path).is_file()


@pytest.mark.parametrize(
    "tile_size,target_size,msg",
    [
        ((0, 8), (10, 10), "tile_size_px"),
        ((8, 8), (0, 10), "target_size_px"),
    ],
)
def test_export_material_binding_maps_tiling_rejects_non_positive(
    tmp_path, tile_size, target_size, msg
) -> None:
    h = np.ones((8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match=msg):
        MapExporter.export_material_binding_maps_with_physical_tiling(
            h,
            str(tmp_path / "o"),
            "s",
            tile_size_px=tile_size,
            target_size_px=target_size,
        )
