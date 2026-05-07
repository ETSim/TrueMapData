"""Tests for :mod:`tmd.model.formats.ply`."""

from __future__ import annotations

import numpy as np

from tmd.model.base import ExportConfig
from tmd.model.formats.ply import PLYExporter


def test_ply_exporter_writes_ascii(tmp_path) -> None:
    hm = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    out = str(tmp_path / "m.ply")
    res = PLYExporter.export(
        hm,
        out,
        ExportConfig(triangulation_method="quadtree", max_triangles=200, x_length=4, y_length=4),
    )
    assert res is not None
    raw = __import__("pathlib").Path(res).read_bytes()
    assert raw.startswith(b"ply")
    assert b"format" in raw
