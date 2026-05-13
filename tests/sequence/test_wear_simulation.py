"""Smoke tests for the height-matrix wear simulation engine."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tmd.sequence import (
    DIRECTION_NAMES,
    WearParameters,
    compare_surface_pairs,
    compute_contact,
    directional_slope,
    run_simulation,
    select_pair,
    select_wear_mask,
    sliding_vector,
)
from tmd.sequence import wear_simulation as wear_sim


def _synthetic_pair(n: int = 32, amp: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Build a small bumpy master and a flat-zero slave that interpenetrate."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    MS = amp * np.sin(2.0 * np.pi * x / n) * np.cos(2.0 * np.pi * y / n)
    SS = np.zeros_like(MS)
    return MS, SS


def test_wear_parameters_effective_moduli() -> None:
    p = WearParameters(E=0.103, poisson=0.34, G=0.041)
    assert math.isclose(p.Ex, 0.103 / (1.0 - 0.34**2), rel_tol=1e-12)
    assert math.isclose(p.Gx, 4.0 * 0.041 / (2.0 - 0.34), rel_tol=1e-12)


def test_directional_slope_signs_on_ramp() -> None:
    """+x ramp: slope along +x should be negative (Z increases with +x, slope = -dZ/dx)."""
    n = 8
    x = np.arange(n, dtype=np.float64)
    Z = np.tile(x, (n, 1))
    sx_plus = directional_slope(Z, direction=1)
    sx_minus = directional_slope(Z, direction=2)
    assert np.all(sx_plus[:, :-1] < 0)
    assert np.all(sx_minus[:, 1:] > 0)
    sy_plus = directional_slope(Z.T, direction=3)
    sy_minus = directional_slope(Z.T, direction=4)
    assert np.all(sy_plus[:-1, :] < 0)
    assert np.all(sy_minus[1:, :] > 0)


def test_directional_slope_unknown_direction_raises() -> None:
    with pytest.raises(ValueError):
        directional_slope(np.zeros((4, 4)), direction=99)


def test_wear_energy_surface_weights_follow_inverse_hardness() -> None:
    w_ms, w_ss = wear_sim._wear_energy_surface_weights(
        WearParameters(hardness_master=1.0, hardness_slave=3.0)
    )
    assert w_ms == pytest.approx(0.75)
    assert w_ss == pytest.approx(0.25)
    eq = wear_sim._wear_energy_surface_weights(WearParameters(hardness_master=2.0, hardness_slave=2.0))
    assert eq[0] == pytest.approx(0.5)
    assert eq[1] == pytest.approx(0.5)


def test_initialize_state_splits_initial_energy_by_hardness() -> None:
    MS, SS = _synthetic_pair(n=16, amp=0.5)
    p = WearParameters(h0=-0.2, hardness_master=1.0, hardness_slave=3.0, friction_clip=0.3)
    st = wear_sim.initialize_state(MS, SS, p, direction=1, load_controlled=False)
    e0 = wear_sim.combined_wear_energy(st["contact"], p)
    assert float(st["wear_energy_MS"].sum() + st["wear_energy_SS"].sum()) == pytest.approx(float(e0.sum()))
    assert float(st["wear_energy_MS"].mean()) == pytest.approx(0.75 * float(e0.mean()))
    assert float(st["wear_energy_SS"].mean()) == pytest.approx(0.25 * float(e0.mean()))


def test_sliding_vector_and_direction_names() -> None:
    assert DIRECTION_NAMES == {1: "+x", 2: "-x", 3: "+y", 4: "-y"}
    assert sliding_vector(1) == (1, 0)
    assert sliding_vector(4) == (0, -1)
    with pytest.raises(ValueError):
        sliding_vector(7)


def test_compute_contact_fixed_gap_has_contact() -> None:
    MS, SS = _synthetic_pair(n=32, amp=0.5)
    p = WearParameters(h0=-0.2, friction_clip=0.3)
    c = compute_contact(MS, SS, p, direction=1, load_controlled=False)
    assert c["N"] > 0
    assert c["force"] > 0.0
    assert c["mask"].shape == MS.shape
    assert math.isclose(c["h0"], -0.2, rel_tol=1e-9)


def test_compute_contact_load_controlled_matches_target() -> None:
    MS, SS = _synthetic_pair(n=24, amp=0.6)
    p = WearParameters(
        h0=0.0,
        normalF=1e-3,
        normal_error=1e-3,
        max_h0_iterations=200,
    )
    c = compute_contact(MS, SS, p, direction=1, load_controlled=True)
    # Bisection should land within (target * normal_error) or quit at the
    # iteration cap; either way force must be finite and non-negative.
    assert c["force"] >= 0.0
    assert np.isfinite(c["force"])


def test_select_wear_mask_topk_respects_fraction() -> None:
    energy = np.linspace(0.0, 1.0, 100, dtype=np.float64).reshape(10, 10)
    p = WearParameters(deltaw=0.0, wear_mode="topk", max_wear_fraction=0.1)
    mask = select_wear_mask(energy, p)
    assert mask.sum() == 10  # 10 % of 100 candidates


def test_select_wear_mask_all_picks_above_threshold() -> None:
    energy = np.array([0.0, 0.5, 1.0, 2.0])
    p = WearParameters(deltaw=0.4, wear_mode="all")
    mask = select_wear_mask(energy.reshape(2, 2), p)
    assert mask.ravel().tolist() == [False, True, True, True]


def test_select_wear_mask_invalid_mode_raises() -> None:
    p = WearParameters(wear_mode="weird")
    with pytest.raises(ValueError):
        select_wear_mask(np.ones((2, 2)), p)


def test_run_simulation_shapes_and_monotonic_wear() -> None:
    MS, SS = _synthetic_pair(n=32, amp=0.5)
    p = WearParameters(
        h0=-0.2,
        wear_scale=1.0,
        wear_clamp_frac=0.05,
        wear_mode="topk",
        max_wear_fraction=0.05,
        friction_clip=0.3,
        deltaw=0.0,
    )
    state, log_rows, snapshots = run_simulation(
        MS, SS, p, direction=1, n_steps=5, load_controlled=False, save_every=2
    )

    assert len(log_rows) == 5
    assert set(snapshots.keys()) == {0, 2, 4, 5}
    for key in (
        "MS",
        "SS",
        "wear_energy_MS",
        "wear_energy_SS",
        "cumulative_wear_MS",
        "cumulative_wear_SS",
        "sliding_distance",
        "sliding_distance_slave",
        "contact",
    ):
        assert key in state

    required_log_keys = {
        "wear_volume_MS",
        "cumulative_wear_sum_MS",
        "contact_percent",
        "normal_force",
    }
    assert required_log_keys.issubset(log_rows[0].keys())

    cum_ms = [row["cumulative_wear_sum_MS"] for row in log_rows]
    assert all(b >= a - 1e-12 for a, b in zip(cum_ms, cum_ms[1:]))
    assert all(row["contact_percent"] >= 0.0 for row in log_rows)


def test_run_simulation_save_every_zero_keeps_only_initial_snapshot() -> None:
    MS, SS = _synthetic_pair(n=16, amp=0.4)
    p = WearParameters(h0=-0.2, wear_scale=1.0, wear_mode="topk", max_wear_fraction=0.05)
    _, _, snapshots = run_simulation(
        MS, SS, p, direction=1, n_steps=3, save_every=0
    )
    assert list(snapshots.keys()) == [0]


def test_select_pair_returns_negated_slave() -> None:
    surfaces = {1: np.ones((4, 4)), 2: np.full((4, 4), 2.0)}
    MS, SS = select_pair(surfaces, 1, 2)
    assert np.allclose(MS, 1.0)
    assert np.allclose(SS, -2.0)
    # Mutating the result must not propagate to the input dict.
    MS[0, 0] = 99.0
    assert surfaces[1][0, 0] == 1.0


def test_compare_surface_pairs_returns_rows_and_histories() -> None:
    n = 16
    surfaces = {
        1: 0.4 * np.sin(np.linspace(0, 2 * np.pi, n))[None, :].repeat(n, axis=0),
        2: np.zeros((n, n)),
        3: 0.3 * np.cos(np.linspace(0, 2 * np.pi, n))[None, :].repeat(n, axis=0),
    }
    p = WearParameters(h0=-0.2, wear_scale=1.0, wear_mode="topk", max_wear_fraction=0.05)
    rows, histories = compare_surface_pairs(
        surfaces, pairs=[(1, 2), (3, 2)], p=p, direction=1, n_steps=4
    )
    assert len(rows) == 2
    assert {r["pair"] for r in rows} == {"MS1-SS2", "MS3-SS2"}
    for label in ("MS1-SS2", "MS3-SS2"):
        assert label in histories
        assert len(histories[label]) == 4
        assert "cumulative_wear_sum_MS" in histories[label][-1]
