"""Tests for :mod:`tmd.sequence.exporters.powerpoint`."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from tmd.sequence.exporters.powerpoint import PowerPointExporter

pytestmark = pytest.mark.skipif(importlib.util.find_spec("pptx") is None, reason="python-pptx optional")


def test_powerpoint_exporter_writes(tmp_path: Path) -> None:
    frames = [np.ones((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32)]
    exp = PowerPointExporter()
    out = str(tmp_path / "slides.pptx")
    path = exp.export(frames=frames, output_file=out, title="t")
    assert path is not None and Path(path).exists()
