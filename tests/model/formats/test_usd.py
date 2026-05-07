"""Tests for :mod:`tmd.model.formats.usd`."""

from __future__ import annotations

import numpy as np
import pytest

pxr = pytest.importorskip("pxr.Usd", reason="USD tests need USD / pxr")

from tmd.model.base import ExportConfig
from tmd.model.formats.usd import USDExporter


def test_usd_export_minimal(tmp_path) -> None:
    hm = np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3)
    out = str(tmp_path / "m.usd")
    res = USDExporter.export(
        hm,
        out,
        ExportConfig(triangulation_method="quadtree", max_triangles=200, x_length=3, y_length=3),
    )
    assert res is not None
    assert __import__("pathlib").Path(res).exists()
