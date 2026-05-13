"""Tests for :mod:`tmd.sequence.wear_analysis` (roughness trajectory)."""

from __future__ import annotations

from tmd.sequence.wear_analysis import append_trajectory_derivatives, ssk_trend_heuristic


def test_append_trajectory_derivatives_valley_share() -> None:
    row = {"Sp": 2.0, "Sv": 2.0, "frame": 0}
    out = append_trajectory_derivatives(row)
    assert out["Sp_Sv_ratio"] == 1.0
    assert out["valley_share"] == 0.5
    assert out["peak_share"] == 0.5


def test_append_trajectory_skips_error_row() -> None:
    row = {"__error__": "x"}
    out = append_trajectory_derivatives(row)
    assert out == row


def test_ssk_trend_heuristic() -> None:
    rows = [{"Ssk": 0.0}, {"Ssk": -0.2}]
    msg = ssk_trend_heuristic(rows)
    assert "negative" in msg.lower() or "valley" in msg.lower()
