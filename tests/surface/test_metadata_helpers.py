"""Additional tests for tmd.surface.metadata (extract, roughness, JSON)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tmd.surface.metadata import (
    analyze_surface_roughness,
    extract_metadata,
    save_metadata_to_json,
)


def test_extract_metadata_includes_dimensions() -> None:
    hm = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    data = {
        "height_map": hm,
        "metadata": {"source": "test"},
    }
    meta = extract_metadata(data)
    assert meta["source"] == "test"
    assert meta["dimensions"]["height"] == 2
    assert meta["dimensions"]["width"] == 2
    assert "height_stats" in meta


def test_analyze_surface_roughness_keys() -> None:
    hm = np.array([[0.0, 0.1], [-0.05, 0.02]], dtype=np.float64)
    r = analyze_surface_roughness(hm)
    for key in ("rq", "ra", "rp", "rv", "rt", "mean", "std_dev"):
        assert key in r
        assert isinstance(r[key], float)


def test_analyze_surface_roughness_with_nan_replacement() -> None:
    hm = np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float64)
    r = analyze_surface_roughness(hm)
    assert np.isfinite(r["rq"])


def test_save_metadata_to_json_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "meta.json"
    payload = {
        "a": 1,
        "b": [1, 2, 3],
        "c": np.float32(0.5),
    }
    save_metadata_to_json(payload, str(path))
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["a"] == 1
    assert loaded["b"] == [1, 2, 3]
    assert loaded["c"] == 0.5
