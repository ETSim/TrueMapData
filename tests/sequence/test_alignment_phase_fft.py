"""Tests for NumPy phase-correlation sequence alignment."""

from __future__ import annotations

import numpy as np

from tmd.core.sequence import TMDSequence
from tmd.sequence.alignment import align_height_map_sequence_phase_fft, estimate_translation_phase_fft


def test_phase_fft_recovers_roll() -> None:
    rng = np.random.default_rng(42)
    ref = rng.random((48, 64)).astype(np.float64)
    ref[20:28, 30:38] += 2.0
    shifted = np.roll(np.roll(ref, 5, axis=0), -7, axis=1)
    frames = [ref, shifted, np.roll(np.roll(ref, 2, axis=0), 3, axis=1)]
    aligned, info = align_height_map_sequence_phase_fft(frames, reference_index=0)
    assert info["method"] == "phase_fft_numpy"
    assert len(aligned) == 3
    c0 = float(np.corrcoef(ref.ravel(), aligned[1].ravel())[0, 1])
    assert c0 > 0.82, f"expected high correlation after align, got {c0}"


def test_estimate_translation_matches_known_shift() -> None:
    rng = np.random.default_rng(0)
    ref = rng.random((24, 32)).astype(np.float64)
    dy, dx = 3, -4
    mov = np.roll(np.roll(ref, dy, axis=0), dx, axis=1)
    esty, estx = estimate_translation_phase_fft(ref, mov)
    assert int(round(esty)) == dy
    assert int(round(estx)) == dx


def test_tmd_sequence_align_phase_fft_metadata() -> None:
    seq = TMDSequence(name="fft")
    rng = np.random.default_rng(1)
    base = rng.random((28, 28)).astype(np.float64)
    base[8:12, 8:12] += 1.0
    seq.add_frame(base, timestamp="a")
    seq.add_frame(np.roll(np.roll(base, 4, axis=0), -2, axis=1), timestamp="b")
    info = seq.align_height_maps_phase_fft(reference_index=0)
    assert seq.metadata["alignment"] == info
    assert info["method"] == "phase_fft_numpy"
    assert len(seq.frames) == 2


def test_to_dict_derived_includes_wear_when_pitch_given() -> None:
    seq = TMDSequence(name="w")
    z = np.zeros((4, 4), dtype=np.float64)
    seq.add_frame(z)
    seq.add_frame(z)
    d = seq.to_dict(include_derived=True, wear_dx_mm=1.0, wear_dy_mm=1.0)
    assert "derived" in d
    assert "statistics" in d["derived"]
    assert "wear" in d["derived"]
    assert "vs_reference" in d["derived"]["wear"]
    assert "incremental" in d["derived"]["wear"]


def test_to_dict_derived_wear_includes_scratch_series_flag() -> None:
    seq = TMDSequence(name="ws")
    seq.add_frame(_groove_like(0.05), "a")
    seq.add_frame(_groove_like(0.35), "b")
    d = seq.to_dict(
        include_derived=True,
        wear_dx_mm=1.0,
        wear_dy_mm=1.0,
        wear_include_scratch_series=True,
    )
    assert "scratch_series" in d["derived"]["wear"]
    assert len(d["derived"]["wear"]["scratch_series"]) == 2


def _groove_like(depth: float) -> np.ndarray:
    size = 24
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(x, y)
    hm = 0.05 * np.sin(3.0 * np.pi * xx) + 0.05 * np.cos(4.0 * np.pi * yy)
    hm[10:12, 3:21] -= depth
    return hm.astype(np.float64)
