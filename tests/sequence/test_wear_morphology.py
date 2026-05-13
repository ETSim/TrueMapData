"""Tests for :mod:`tmd.sequence.wear_analysis` (morphology)."""

from __future__ import annotations

import numpy as np

from tmd.sequence.wear_analysis import scratch_evolution_pair, slip_axis_metrics


def test_scratch_evolution_growth() -> None:
    a = np.zeros((20, 20), dtype=bool)
    b = np.zeros_like(a)
    b[5:15, 10] = True
    ev = scratch_evolution_pair(a, b)
    assert ev["growth_pixels"] == int(b.sum())
    assert ev["area_after"] == int(b.sum())


def test_slip_axis_runs() -> None:
    rng = np.random.default_rng(3)
    z = rng.random((32, 32)).astype(np.float64) * 0.02
    m = slip_axis_metrics(z)
    assert "gradient_structure_angle_deg" in m
    assert np.isfinite(m["psd_wedge_asymmetry"])
