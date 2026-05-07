"""Tests for :class:`tmd.surface.terrain.TMDTerrain` sample generators."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.surface.terrain import TMDTerrain


@pytest.mark.parametrize(
    "pattern",
    ["waves", "peak", "dome", "ramp", "combined", "flat", "random", "perlin", "fbm", "square", "sawtooth"],
)
def test_create_sample_height_map_patterns(pattern: str) -> None:
    h = TMDTerrain.create_sample_height_map(16, 16, pattern=pattern, seed=42)
    assert h.shape == (16, 16)


def test_unknown_pattern_returns_zeros() -> None:
    h = TMDTerrain.create_sample_height_map(8, 8, pattern="___unknown___", seed=1)
    assert h.shape == (8, 8) and float(np.max(h)) == 0.0


def test_generate_synthetic_tmd(tmp_path) -> None:
    p = str(tmp_path / "syn.tmd")
    out = TMDTerrain.generate_synthetic_tmd(output_path=p, width=8, height=8, pattern="flat")
    assert out == p or __import__("os").path.samefile(out, p)
