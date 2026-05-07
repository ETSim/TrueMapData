"""Tests for :mod:`tmd.model.formats.stl`."""

from __future__ import annotations

import numpy as np

from tmd.model.base import ExportConfig
from tmd.model.formats.stl import STLExporter


def test_stl_exporter_writes_binary(tmp_path) -> None:
    hm = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    out = str(tmp_path / "out.stl")
    res = STLExporter.export(
        hm,
        out,
        ExportConfig(triangulation_method="quadtree", max_triangles=200, x_length=4, y_length=4, binary=True),
    )
    assert res is not None
    data = __import__("pathlib").Path(res).read_bytes()
    assert len(data) > 80 and data[:5] != b"solid"
