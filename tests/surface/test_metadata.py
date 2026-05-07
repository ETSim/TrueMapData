"""Tests for tmd.surface.metadata helpers."""

from __future__ import annotations

import numpy as np

from tmd.surface.metadata import compute_stats, export_metadata, export_metadata_txt


def test_compute_stats_normal_array() -> None:
    h = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    stats = compute_stats(h)
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["mean"] == 2.5
    assert stats["shape"] == (2, 2)
    assert stats["non_nan"] == 4
    assert stats["nan_count"] == 0


def test_compute_stats_all_nan() -> None:
    h = np.full((2, 3), np.nan, dtype=np.float64)
    stats = compute_stats(h)
    assert stats["min"] == 0.0
    assert stats["max"] == 0.0
    assert stats["mean"] == 0.0
    assert stats["nan_count"] == 6
    assert stats["non_nan"] == 0
    assert stats["shape"] == (2, 3)


def test_export_metadata_writes_expected_sections(tmp_path) -> None:
    out = tmp_path / "meta.txt"
    meta = {"file_path": str(tmp_path / "in.tmd"), "units": "mm"}
    stats = {"min": 0.0, "max": 1.0}
    export_metadata(meta, stats, str(out))
    text = out.read_text(encoding="utf-8")
    assert "TMD File:" in text
    assert "units: mm" in text
    assert "Height Map Statistics" in text
    assert "min: 0.0" in text


def test_export_metadata_txt_skips_height_map_in_header(tmp_path, capsys) -> None:
    out = tmp_path / "m.txt"
    h = np.ones((3, 4), dtype=np.float32)
    export_metadata_txt({"note": "x", "height_map": h}, str(out))
    captured = capsys.readouterr()
    assert "saved to text file" in captured.out
    text = out.read_text(encoding="utf-8")
    assert "note: x" in text
    assert "Height Map Statistics" in text
    assert "Shape: (3, 4)" in text
