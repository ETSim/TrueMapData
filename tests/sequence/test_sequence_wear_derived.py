"""Tests for TMDSequence.sequential_wear_metrics derived outputs and alignment."""

from __future__ import annotations

import numpy as np

from tmd.core.sequence import TMDSequence


def _groove_maps(*, deep: bool) -> np.ndarray:
    size = 32
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(x, y)
    height_map = 0.05 * np.sin(3.0 * np.pi * xx) + 0.05 * np.cos(4.0 * np.pi * yy)
    if deep:
        height_map[14:16, 4:28] -= 0.35
    else:
        height_map[14:16, 4:28] -= 0.05
    return height_map.astype(np.float64)


def test_sequential_wear_slip_axis_series() -> None:
    seq = TMDSequence("s")
    seq.add_frame(_groove_maps(deep=False), "a", {})
    seq.add_frame(_groove_maps(deep=True), "b", {})
    out = seq.sequential_wear_metrics(
        dx_mm=None,
        dy_mm=None,
        include_slip_axis_series=True,
    )
    assert "vs_reference" not in out and "incremental" not in out
    assert "slip_axis_series" in out
    rows = out["slip_axis_series"]
    assert len(rows) == 2
    assert rows[0]["frame_index"] == 0
    assert "psd_wedge_asymmetry" in rows[0]


def test_sequential_wear_scratch_series() -> None:
    seq = TMDSequence("s")
    seq.add_frame(_groove_maps(deep=False), "a", {})
    seq.add_frame(_groove_maps(deep=True), "b", {})
    out = seq.sequential_wear_metrics(
        dx_mm=None,
        dy_mm=None,
        include_scratch_series=True,
    )
    assert "scratch_series" in out
    s = out["scratch_series"]
    assert len(s) == 2
    assert s[0].get("note") == "reference"


def test_sequential_wear_align_phase_fft_then_volumes() -> None:
    rng = np.random.default_rng(7)
    base = rng.random((24, 24)).astype(np.float64)
    base[8:12, 8:12] += 0.5
    shifted = np.roll(np.roll(base, 3, axis=0), -2, axis=1)
    seq = TMDSequence("align-wear")
    seq.add_frame(base.copy(), "0", {})
    seq.add_frame(shifted.copy(), "1", {})
    out = seq.sequential_wear_metrics(
        dx_mm=1.0,
        dy_mm=1.0,
        reference_index=0,
        align_before="phase_fft",
        include_slip_axis_series=False,
    )
    assert "alignment" in out
    assert out["alignment"].get("method") == "phase_fft_numpy"
    assert "vs_reference" in out and "incremental" in out


def test_to_dict_wear_slip_without_pitch() -> None:
    seq = TMDSequence("d")
    seq.add_frame(np.zeros((8, 8), dtype=np.float64), "t0", {})
    seq.add_frame(np.ones((8, 8), dtype=np.float64) * 0.01, "t1", {})
    d = seq.to_dict(
        include_derived=True,
        wear_include_slip_axis_series=True,
    )
    assert "derived" in d and "wear" in d["derived"]
    assert "slip_axis_series" in d["derived"]["wear"]
    assert "vs_reference" not in d["derived"]["wear"]
