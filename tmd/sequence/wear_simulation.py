"""Height-matrix wear simulation engine.

Self-contained, NumPy-only port of the Sun (2025) ``Surface wear characterized by
height matrix`` workflow. Public entry points live in this module and are re-exported
from :mod:`tmd.sequence` for convenience.

Two height fields slide against each other (**master** ``MS`` / **slave** ``SS``);
local penetration drives a Hertz-like normal force, directional friction, accumulating
wear-energy fields, and material removal. Incoming slip energy is **split between
bodies** in proportion to ``1 / hardness`` so softer surfaces accumulate more
wear-energy per step before the usual ``1 / hardness`` depth removal.
The engine is independent of pandas and any plotting stack so it can be embedded in
pipelines and tests; callers wanting tabular logs can wrap ``run_simulation``'s
``list[dict]`` output with ``pandas.DataFrame``.

Reference:
    Sun, D. (2025). MATLAB code for "Surface wear characterized by height matrix in
    relation to texture direction (Std) and other surface parameters" (Version 1).
    Mendeley Data. https://doi.org/10.17632/v635kysfr2.1 (CC BY 4.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


__all__ = [
    "DIRECTION_NAMES",
    "WearParameters",
    "apply_wear",
    "combined_mu",
    "combined_wear_energy",
    "compare_surface_pairs",
    "compute_contact",
    "contact_at_h0",
    "directional_mu",
    "directional_slope",
    "export_final_state_mat",
    "find_default_input_surface",
    "height_matrix_surface_metrics",
    "initialize_state",
    "load_surfaces_for_simulation",
    "make_snapshot",
    "ms_snapshots_to_sequence",
    "pixel_pitch_mm_from_metadata",
    "roll_with_slide",
    "run_simulation",
    "select_pair",
    "select_wear_mask",
    "simulation_step",
    "simulation_metadata_to_pitch",
    "slide_slave",
    "sliding_energy",
    "sliding_vector",
    "slip_energy",
    "solve_h0_for_load",
    "texture_direction_spectrum",
    "truemap_repo_root",
    "wear_depth_from_energy",
]


# --- Direction primitives -----------------------------------------------------

DIRECTION_NAMES: Dict[int, str] = {1: "+x", 2: "-x", 3: "+y", 4: "-y"}
_SLIDING_VECTORS: Dict[int, Tuple[int, int]] = {1: (1, 0), 2: (-1, 0), 3: (0, 1), 4: (0, -1)}
_ROLL_SHIFTS: Dict[int, Tuple[int, int]] = {1: (0, 1), 2: (0, -1), 3: (1, 0), 4: (-1, 0)}


def sliding_vector(direction: int) -> Tuple[int, int]:
    """Unit ``(du, dv)`` lab-grid offset for an integer direction code."""
    try:
        return _SLIDING_VECTORS[int(direction)]
    except KeyError as exc:
        raise ValueError(f"direction must be one of {list(DIRECTION_NAMES)}") from exc


def roll_with_slide(field: np.ndarray, direction: int) -> np.ndarray:
    """``np.roll`` ``field`` by one texel along the sliding direction."""
    try:
        shift = _ROLL_SHIFTS[int(direction)]
    except KeyError as exc:
        raise ValueError(f"direction must be one of {list(DIRECTION_NAMES)}") from exc
    return np.roll(field, shift, axis=(0, 1))


# --- Parameters ---------------------------------------------------------------


@dataclass
class WearParameters:
    """Material, contact, and solver parameters for the wear simulation."""

    # Contact offset for fixed-gap mode; smaller h0 increases penetration.
    h0: float = 1.3

    # Hertz / Mindlin material constants.
    E: float = 0.103
    poisson: float = 0.34
    f0: float = 0.075
    G: float = 0.041
    R: float = 0.5

    # Adhesive wear-energy threshold (Rabinowicz-style).
    deltaw: float = 3.4e-12

    # Target normal force and tolerance for load-controlled bisection.
    normalF: float = 0.00359
    normal_error: float = 0.0001

    # Sliding distance per simulation step.
    delta_slid: float = 1.0

    # Wear-depth scale; published MATLAB value (5000) is calibrated for tiny
    # load-controlled forces and overshoots in fixed-gap demos.
    wear_scale: float = 5000.0

    # Friction clamp; avoids edge spikes that would create negative traction.
    friction_clip: float = 0.3

    # Vectorized-wear numerical safety net.
    enable_wear_clamp: bool = True
    wear_clamp_frac: float = 0.16

    # ``"all"`` wears every texel above ``deltaw``; ``"topk"`` keeps only the
    # ``max_wear_fraction`` highest-energy texels each step (closer to the
    # MATLAB greedy update).
    wear_mode: str = "all"
    max_wear_fraction: float = 1.0

    # Load-controlled bisection budget.
    max_h0_iterations: int = 400

    # Relative hardness (Archard-style: wear depth scales like 1 / H on each body).
    # Also drives the split of ``combined_wear_energy`` between ``wear_energy_MS`` and
    # ``wear_energy_SS``: share ∝ 1/H so softer material receives more accumulated
    # energy before removal.
    hardness_master: float = 1.0
    hardness_slave: float = 1.0

    @property
    def Ex(self) -> float:
        """Effective Young's modulus ``E / (1 - poisson**2)``."""
        return self.E / (1.0 - self.poisson**2)

    @property
    def Gx(self) -> float:
        """Effective shear modulus ``4 * G / (2 - poisson)``."""
        return 4.0 * self.G / (2.0 - self.poisson)


# --- Friction -----------------------------------------------------------------


def directional_slope(Z: np.ndarray, direction: int, spacing: float = 1.0) -> np.ndarray:
    """Forward / backward slope along ``direction`` with edge replication."""
    Z = np.asarray(Z, dtype=np.float64)
    s = np.empty_like(Z)
    d = int(direction)
    if d == 1:
        s[:, :-1] = (Z[:, :-1] - Z[:, 1:]) / spacing
        s[:, -1] = s[:, -2]
    elif d == 2:
        s[:, 1:] = (Z[:, 1:] - Z[:, :-1]) / spacing
        s[:, 0] = s[:, 1]
    elif d == 3:
        s[:-1, :] = (Z[:-1, :] - Z[1:, :]) / spacing
        s[-1, :] = s[-2, :]
    elif d == 4:
        s[1:, :] = (Z[1:, :] - Z[:-1, :]) / spacing
        s[0, :] = s[1, :]
    else:
        raise ValueError(f"direction must be one of {list(DIRECTION_NAMES)}")
    return s


def directional_mu(
    Z: np.ndarray,
    p: WearParameters,
    direction: int,
    spacing: float = 1.0,
) -> np.ndarray:
    """Per-texel friction along ``direction`` (slope-corrected, single surface)."""
    slope = directional_slope(Z, direction, spacing)
    denom = np.sqrt(1.0 + slope * slope)
    sin_theta = slope / denom
    cos_theta = 1.0 / denom

    normalF = float(p.normalF)
    f0 = float(p.f0)
    normal_component = normalF * cos_theta + normalF * f0 * sin_theta
    shear_component = normalF * f0 * cos_theta - normalF * sin_theta
    tau0 = np.divide(
        shear_component,
        normal_component,
        out=np.zeros_like(shear_component),
        where=np.abs(normal_component) > 1e-15,
    )
    return tau0 + slope


def combined_mu(MS: np.ndarray, SS: np.ndarray, p: WearParameters, direction: int) -> np.ndarray:
    """Sum of per-surface directional friction, clipped to ``[0, friction_clip]``."""
    raw = directional_mu(MS, p, direction) + directional_mu(SS, p, direction)
    return np.clip(raw, 0.0, float(p.friction_clip))


# --- Contact mechanics --------------------------------------------------------


def contact_at_h0(
    MS: np.ndarray,
    SS: np.ndarray,
    h0: float,
    p: WearParameters,
    mu: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate the contact state at a fixed gap ``h0``.

    Penetration ``d = MS - (SS - h0)``; texels with ``d > 0`` are in contact and
    contribute Hertz-like normal force ``dF = 4/3 * Ex * sqrt(R) * d^(3/2)``.
    """
    MS = np.asarray(MS, dtype=np.float64)
    SS = np.asarray(SS, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)

    d = MS - (SS - h0)
    mask = d > 0.0

    deltaF = np.zeros_like(MS)
    deltaA = np.zeros_like(MS)

    pen = d[mask]
    Ex = float(p.Ex)
    R = float(p.R)
    deltaF[mask] = (4.0 / 3.0) * Ex * np.sqrt(R) * np.power(pen, 1.5)
    deltaA[mask] = np.pi * R * pen

    tau = mu * deltaF
    poisson = float(p.poisson)
    G = float(p.G)
    ux = np.pi * (2.0 - poisson) * tau * R / (4.0 * G)

    return {
        "h0": float(h0),
        "d": d,
        "mask": mask,
        "deltaF": deltaF,
        "deltaA": deltaA,
        "tau": tau,
        "ux": ux,
        "N": int(mask.sum()),
        "area": float(deltaA.sum()),
        "force": float(deltaF.sum()),
    }


def solve_h0_for_load(
    MS: np.ndarray,
    SS: np.ndarray,
    p: WearParameters,
    mu: np.ndarray,
) -> Dict[str, Any]:
    """Bisect ``h0`` so total normal force matches ``p.normalF``."""
    target = float(p.normalF)
    tolerance = max(abs(target * float(p.normal_error)), 1e-18)

    extent = float(np.max(np.abs(MS)) + np.max(np.abs(SS)) + abs(float(p.h0)) + 1.0)
    lo, hi = -extent, extent
    best: Optional[Dict[str, Any]] = None

    for _ in range(int(p.max_h0_iterations)):
        mid = 0.5 * (lo + hi)
        c = contact_at_h0(MS, SS, mid, p, mu)
        best = c
        if abs(c["force"] - target) <= tolerance:
            return c
        if c["force"] < target:
            lo = mid
        else:
            hi = mid
    assert best is not None  # noqa: S101 - max_h0_iterations >= 1 by construction
    return best


def compute_contact(
    MS: np.ndarray,
    SS: np.ndarray,
    p: WearParameters,
    direction: int,
    load_controlled: bool = False,
) -> Dict[str, Any]:
    """Combined-friction contact at either fixed gap or target load."""
    mu = combined_mu(MS, SS, p, direction)
    if load_controlled:
        return solve_h0_for_load(MS, SS, p, mu)
    return contact_at_h0(MS, SS, float(p.h0), p, mu)


# --- Wear energy and update --------------------------------------------------


def slip_energy(tau: np.ndarray, ux: np.ndarray) -> np.ndarray:
    """Instantaneous slip energy ``0.5 * tau * ux`` per texel."""
    return 0.5 * np.asarray(tau, dtype=np.float64) * np.asarray(ux, dtype=np.float64)


def sliding_energy(tau: np.ndarray, delta_slid: float) -> np.ndarray:
    """Per-step sliding energy ``tau * delta_slid``."""
    return np.asarray(tau, dtype=np.float64) * float(delta_slid)


def combined_wear_energy(contact: Mapping[str, Any], p: WearParameters) -> np.ndarray:
    """Slip + sliding wear-energy contribution from a contact state."""
    return slip_energy(contact["tau"], contact["ux"]) + sliding_energy(contact["tau"], p.delta_slid)


def _wear_energy_surface_weights(p: WearParameters) -> Tuple[float, float]:
    """Return ``(w_ms, w_ss)`` partitioning one unit of combined wear energy between bodies.

    Weights are proportional to ``1 / hardness`` so a softer body (smaller ``H``)
    receives a larger share of the frictional work rate. Equal hardness yields
    ``(0.5, 0.5)``, recovering the previous symmetric energy bookkeeping.
    """
    inv_hm = 1.0 / max(float(p.hardness_master), 1e-12)
    inv_hs = 1.0 / max(float(p.hardness_slave), 1e-12)
    den = inv_hm + inv_hs
    return inv_hm / den, inv_hs / den


def select_wear_mask(energy: np.ndarray, p: WearParameters) -> np.ndarray:
    """Boolean mask of texels eligible for material removal this step."""
    energy = np.asarray(energy, dtype=np.float64)
    flat = energy.ravel()
    candidates = np.flatnonzero(flat > float(p.deltaw))
    mask = np.zeros(flat.size, dtype=bool)
    if candidates.size == 0:
        return mask.reshape(energy.shape)
    if p.wear_mode == "topk":
        k = max(1, int(np.ceil(float(p.max_wear_fraction) * candidates.size)))
        k = min(k, candidates.size)
        values = flat[candidates]
        chosen = candidates[np.argpartition(values, -k)[-k:]]
        mask[chosen] = True
    elif p.wear_mode == "all":
        mask[candidates] = True
    else:
        raise ValueError("wear_mode must be 'all' or 'topk'")
    return mask.reshape(energy.shape)


def wear_depth_from_energy(
    energy: np.ndarray,
    Z: np.ndarray,
    p: WearParameters,
    inv_hardness: float = 1.0,
) -> np.ndarray:
    """Depth removed from one surface for the supplied energy values."""
    h = np.asarray(energy, dtype=np.float64) * float(p.wear_scale) * float(inv_hardness)
    if p.enable_wear_clamp:
        amp = float(Z.max() - Z.min())
        if amp > 0:
            h = np.minimum(h, float(p.wear_clamp_frac) * amp)
    return h


def apply_wear(
    MS: np.ndarray,
    SS: np.ndarray,
    eMS: np.ndarray,
    eSS: np.ndarray,
    p: WearParameters,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Remove material from both height maps according to per-body wear-energy fields."""
    MS = np.asarray(MS, dtype=np.float64).copy()
    SS = np.asarray(SS, dtype=np.float64).copy()
    eMS = np.asarray(eMS, dtype=np.float64).copy()
    eSS = np.asarray(eSS, dtype=np.float64).copy()

    maskMS = select_wear_mask(eMS, p)
    maskSS = select_wear_mask(eSS, p)

    inv_hm = 1.0 / max(float(p.hardness_master), 1e-12)
    inv_hs = 1.0 / max(float(p.hardness_slave), 1e-12)

    hMS = np.zeros_like(eMS)
    hSS = np.zeros_like(eSS)
    if maskMS.any():
        hMS[maskMS] = wear_depth_from_energy(eMS[maskMS], MS, p, inv_hm)
    if maskSS.any():
        hSS[maskSS] = wear_depth_from_energy(eSS[maskSS], SS, p, inv_hs)

    MS[maskMS] -= hMS[maskMS]
    SS[maskSS] += hSS[maskSS]

    eMS[maskMS] = 0.0
    eSS[maskSS] = 0.0

    info = {
        "wear_points_MS": int(maskMS.sum()),
        "wear_points_SS": int(maskSS.sum()),
        "wear_volume_MS": float(hMS.sum()),
        "wear_volume_SS": float(hSS.sum()),
        "max_wear_MS": float(hMS.max() if maskMS.any() else 0.0),
        "max_wear_SS": float(hSS.max() if maskSS.any() else 0.0),
    }
    return MS, SS, eMS, eSS, hMS, hSS, info


def slide_slave(
    SS: np.ndarray, eSS: np.ndarray, direction: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Shift the slave (``SS``) height map and its energy field one texel along ``direction``."""
    return roll_with_slide(SS, direction), roll_with_slide(eSS, direction)


# --- Simulation loop ----------------------------------------------------------


def initialize_state(
    MS: np.ndarray,
    SS: np.ndarray,
    p: WearParameters,
    direction: int = 1,
    load_controlled: bool = False,
) -> Dict[str, Any]:
    """Build the starting state dict consumed by :func:`simulation_step`."""
    MS = np.asarray(MS, dtype=np.float64)
    SS = np.asarray(SS, dtype=np.float64)
    contact = compute_contact(MS, SS, p, direction, load_controlled)
    e0 = combined_wear_energy(contact, p)
    w_ms, w_ss = _wear_energy_surface_weights(p)
    slip0 = np.where(contact["mask"], p.delta_slid, 0.0)
    du, dv = sliding_vector(direction)
    return {
        "MS": MS.copy(),
        "SS": SS.copy(),
        "wear_energy_MS": (e0 * w_ms).copy(),
        "wear_energy_SS": (e0 * w_ss).copy(),
        "cumulative_wear_MS": np.zeros_like(MS),
        "cumulative_wear_SS": np.zeros_like(SS),
        "last_wear_MS": np.zeros_like(MS),
        "last_wear_SS": np.zeros_like(SS),
        "sliding_distance": slip0.copy(),
        "sliding_distance_slave": slip0.copy(),
        "slide_u": np.full_like(MS, du, dtype=np.float64),
        "slide_v": np.full_like(MS, dv, dtype=np.float64),
        "contact": contact,
    }


def simulation_step(
    state: Mapping[str, Any],
    p: WearParameters,
    direction: int,
    load_controlled: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Advance one wear / sliding step. Returns ``(new_state, log_row)``."""
    MS, SS = state["MS"], state["SS"]
    eMS, eSS = state["wear_energy_MS"], state["wear_energy_SS"]
    cumulative_wear_MS = state["cumulative_wear_MS"]
    cumulative_wear_SS = state["cumulative_wear_SS"]
    sliding_distance_slave = state["sliding_distance_slave"]

    MS, SS, eMS, eSS, hMS, hSS, wear_info = apply_wear(MS, SS, eMS, eSS, p)
    cumulative_wear_MS = cumulative_wear_MS + hMS
    cumulative_wear_SS = cumulative_wear_SS + hSS

    SS, eSS = slide_slave(SS, eSS, direction)
    cumulative_wear_SS = roll_with_slide(cumulative_wear_SS, direction)
    sliding_distance_slave = roll_with_slide(sliding_distance_slave, direction)

    contact = compute_contact(MS, SS, p, direction, load_controlled)
    new_energy = combined_wear_energy(contact, p)
    w_ms, w_ss = _wear_energy_surface_weights(p)
    eMS = eMS + new_energy * w_ms
    eSS = eSS + new_energy * w_ss

    du, dv = sliding_vector(direction)
    slip_inc = np.where(contact["mask"], p.delta_slid, 0.0)
    sliding_distance = state["sliding_distance"] + slip_inc
    sliding_distance_slave = sliding_distance_slave + slip_inc

    new_state = {
        "MS": MS,
        "SS": SS,
        "wear_energy_MS": eMS,
        "wear_energy_SS": eSS,
        "cumulative_wear_MS": cumulative_wear_MS,
        "cumulative_wear_SS": cumulative_wear_SS,
        "last_wear_MS": hMS,
        "last_wear_SS": hSS,
        "sliding_distance": sliding_distance,
        "sliding_distance_slave": sliding_distance_slave,
        "slide_u": np.full_like(MS, du, dtype=np.float64),
        "slide_v": np.full_like(MS, dv, dtype=np.float64),
        "contact": contact,
    }

    log = {
        **wear_info,
        "h0": contact["h0"],
        "contact_texels": contact["N"],
        "contact_percent": 100.0 * contact["N"] / MS.size,
        "normal_force": contact["force"],
        "contact_area": contact["area"],
        "traction_sum": float(contact["tau"].sum()),
        "traction_max": float(contact["tau"].max()),
        "energy_max_MS": float(eMS.max()),
        "energy_max_SS": float(eSS.max()),
        "energy_sum_MS": float(eMS.sum()),
        "energy_sum_SS": float(eSS.sum()),
        "sliding_distance_sum": float(sliding_distance.sum()),
        "sliding_slave_sum": float(sliding_distance_slave.sum()),
        "cumulative_wear_sum_MS": float(cumulative_wear_MS.sum()),
        "cumulative_wear_sum_SS": float(cumulative_wear_SS.sum()),
        "cumulative_wear_max_MS": float(cumulative_wear_MS.max()),
        "cumulative_wear_max_SS": float(cumulative_wear_SS.max()),
    }
    return new_state, log


def make_snapshot(state: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Lightweight per-step snapshot (no contact dict; arrays only) for animations."""
    c = state["contact"]
    return {
        "MS": np.asarray(state["MS"]).copy(),
        "SS": np.asarray(state["SS"]).copy(),
        "mask": c["mask"].astype(float).copy(),
        "penetration": np.where(c["mask"], c["d"], 0.0).copy(),
        "tau": c["tau"].copy(),
        "wear_energy": np.asarray(state["wear_energy_MS"]).copy(),
        "wear_map": np.asarray(state["cumulative_wear_MS"]).copy(),
        "wear_map_SS": np.asarray(state["cumulative_wear_SS"]).copy(),
        "wear_increment": np.asarray(state["last_wear_MS"]).copy(),
        "wear_increment_SS": np.asarray(state["last_wear_SS"]).copy(),
        "sliding_distance": np.asarray(state["sliding_distance"]).copy(),
        "sliding_distance_slave": np.asarray(state["sliding_distance_slave"]).copy(),
    }


def run_simulation(
    MS: np.ndarray,
    SS: np.ndarray,
    p: WearParameters,
    direction: int = 1,
    n_steps: int = 200,
    load_controlled: bool = False,
    save_every: int = 20,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[int, Dict[str, np.ndarray]]]:
    """Run ``n_steps`` of the wear loop and return ``(state, log_rows, snapshots)``.

    ``log_rows`` is a plain ``list[dict]`` (one row per step) so the engine has no
    pandas dependency; wrap it in ``pandas.DataFrame(log_rows).set_index("step")``
    on the caller side when a tabular view is needed.
    """
    state = initialize_state(MS, SS, p, direction, load_controlled)
    log_rows: List[Dict[str, Any]] = []
    snapshots: Dict[int, Dict[str, np.ndarray]] = {0: make_snapshot(state)}

    for step_idx in range(1, int(n_steps) + 1):
        state, row = simulation_step(state, p, direction, load_controlled)
        row["step"] = step_idx
        log_rows.append(row)
        if save_every and (step_idx % int(save_every) == 0 or step_idx == n_steps):
            snapshots[step_idx] = make_snapshot(state)

    return state, log_rows, snapshots


# --- Helpers for multi-surface comparison ------------------------------------


def select_pair(
    surfaces: Mapping[int, np.ndarray],
    ms_idx: int,
    ss_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(MS, -SS)`` copies; matches the notebook master/slave sign convention."""
    MS = np.asarray(surfaces[int(ms_idx)], dtype=np.float64).copy()
    SS = -np.asarray(surfaces[int(ss_idx)], dtype=np.float64).copy()
    return MS, SS


def compare_surface_pairs(
    surfaces: Mapping[int, np.ndarray],
    pairs: Sequence[Tuple[int, int]],
    p: WearParameters,
    direction: int = 1,
    n_steps: int = 60,
    load_controlled: bool = False,
    save_every: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Run the wear loop on several ``(MS_idx, SS_idx)`` pairs and summarize each.

    Returns ``(rows, histories)`` where ``rows`` is one summary dict per pair and
    ``histories[label]`` is the full per-step log list for that pair. Use the
    same negative-sign convention as :func:`select_pair`.
    """
    rows: List[Dict[str, Any]] = []
    histories: Dict[str, List[Dict[str, Any]]] = {}
    for ms_i, ss_i in pairs:
        MS_i, SS_i = select_pair(surfaces, ms_i, ss_i)
        _, log_rows, _ = run_simulation(
            MS_i,
            SS_i,
            p,
            direction=direction,
            n_steps=n_steps,
            load_controlled=load_controlled,
            save_every=save_every,
        )
        label = f"MS{int(ms_i)}-SS{int(ss_i)}"
        histories[label] = log_rows
        last = log_rows[-1] if log_rows else {}
        mean_force = (
            float(np.mean([row["normal_force"] for row in log_rows])) if log_rows else float("nan")
        )
        rows.append(
            {
                "pair": label,
                "final_contact_percent": float(last.get("contact_percent", float("nan"))),
                "cumulative_wear_MS": float(last.get("cumulative_wear_sum_MS", 0.0)),
                "cumulative_wear_SS": float(last.get("cumulative_wear_sum_SS", 0.0)),
                "mean_force": mean_force,
            }
        )
    return rows, histories


# --- Repo root, pitch, lightweight surface metrics -----------------------------


def truemap_repo_root(start: Optional[Path] = None) -> Path:
    """Resolve TrueMapData repository root from *start* or ``Path.cwd()``."""
    p = (start or Path.cwd()).resolve()
    for d in (p, *p.parents):
        if (d / "examples").is_dir() or (d / "tmd").is_dir():
            return d
        if (d / ".git").is_dir():
            return d
    if p.name == "notebooks":
        return p.parent
    return p


def pixel_pitch_mm_from_metadata(meta: Mapping[str, Any], shape: Tuple[int, int]) -> Tuple[float, float]:
    """Pixel pitch (mm) along X and Y from metadata keys ``width``/``height``/``x_length``/``y_length``."""
    h, w = int(shape[0]), int(shape[1])
    width = int(meta.get("width", w))
    height = int(meta.get("height", h))
    xl = float(meta.get("x_length", 10.0))
    yl = float(meta.get("y_length", 10.0))
    dx = xl / max(width, 1)
    dy = yl / max(height, 1)
    return dx, dy


def simulation_metadata_to_pitch(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Map ``a_x_length``-style keys from :func:`load_surfaces_for_simulation` to plain pitch keys."""
    out: Dict[str, Any] = {}
    for plain, prefixed in (
        ("x_length", "a_x_length"),
        ("y_length", "a_y_length"),
        ("width", "a_width"),
        ("height", "a_height"),
    ):
        if prefixed in meta:
            out[plain] = meta[prefixed]
        elif plain in meta:
            out[plain] = meta[plain]
    return out


def texture_direction_spectrum(Z: np.ndarray, angular_step: int = 2) -> Dict[str, Any]:
    """Angular energy in the 2D FFT magnitude; returns dominant texture direction ``Std_deg``."""
    z = np.asarray(Z, dtype=np.float64)
    f = np.fft.fftshift(np.fft.fft2(z))
    mag = np.abs(f)
    rows, cols = z.shape
    u = np.arange(cols) - cols // 2
    v = np.arange(rows) - rows // 2
    uu, vv = np.meshgrid(u, v)
    theta = np.mod(np.arctan2(vv, uu), np.pi)

    angles = np.arange(0, 180, angular_step)
    spectrum = np.zeros_like(angles, dtype=np.float64)
    half = np.deg2rad(angular_step / 2.0)
    for i, deg in enumerate(angles):
        a = np.deg2rad(deg)
        spectrum[i] = float(mag[np.abs(theta - a) <= half].sum())

    dominant = float(angles[int(np.argmax(spectrum))])
    return {"angles_deg": angles, "spectrum": spectrum, "Std_deg": dominant}


def height_matrix_surface_metrics(Z: np.ndarray) -> Dict[str, float]:
    """Notebook-style roughness / texture metrics (NumPy + SciPy ``ndimage`` only)."""
    from scipy import ndimage

    z = np.asarray(Z, dtype=np.float64)
    zy, zx = np.gradient(z)
    sq = float(np.sqrt(np.mean(z**2)))
    sdq = float(np.sqrt(np.mean(zx**2 + zy**2)))
    actual_area = float(np.sum(np.sqrt(1.0 + zx[:-1, :-1] ** 2 + zy[:-1, :-1] ** 2)))
    nominal_area = float((z.shape[0] - 1) * (z.shape[1] - 1))
    sdr = (actual_area - nominal_area) / nominal_area if nominal_area > 0 else 0.0
    ssk = float(np.mean(z**3) / (sq**3)) if sq > 0 else 0.0
    sku = float(np.mean(z**4) / (sq**4)) if sq > 0 else 0.0
    spc = float(-0.5 * np.mean(ndimage.laplace(z, mode="nearest")))
    local_max = ndimage.maximum_filter(z, size=3, mode="nearest")
    peaks = (z == local_max) & (z > z.mean())
    peaks[[0, -1], :] = False
    peaks[:, [0, -1]] = False
    spd = float(peaks.sum() / z.size)
    std = float(texture_direction_spectrum(z)["Std_deg"])
    return {"Sq": sq, "Sdq": sdq, "Sdr": sdr, "Ssk": ssk, "Sku": sku, "Spc": spc, "Spd": spd, "Std_deg": std}


# --- TMD / MAT loading --------------------------------------------------------


_TMD_SUFFIXES = frozenset({".tmd"})
_MAT_SUFFIXES = frozenset({".mat"})


def _maybe_downsample(arr: np.ndarray, max_dim: int) -> np.ndarray:
    from scipy import ndimage

    a = np.asarray(arr, dtype=np.float64)
    if max_dim is None or max_dim <= 0:
        return a
    h, w = a.shape
    longest = max(h, w)
    if longest <= max_dim:
        return a
    factor = max_dim / float(longest)
    return ndimage.zoom(a, zoom=factor, order=1).astype(np.float64)


def _maybe_normalize(arr: np.ndarray, *, normalize_to_unit_sq: bool) -> Tuple[np.ndarray, float]:
    a = np.asarray(arr, dtype=np.float64)
    if not normalize_to_unit_sq:
        return a, 1.0
    centered = a - float(np.mean(a))
    sq = float(np.sqrt(np.mean(centered**2)))
    if sq <= 1e-12:
        return centered, 1.0
    return centered / sq, sq


def _coerce_to_metric_dict(raw: Mapping[str, Any]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for name, value in raw.items():
        if name == "Zp":
            continue
        try:
            cleaned[name] = float(np.asarray(value).flatten()[0])
        except (TypeError, ValueError):
            continue
    return cleaned


def _detect_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _TMD_SUFFIXES:
        return "tmd"
    if suffix in _MAT_SUFFIXES:
        return "mat"
    raise ValueError(f"Unsupported input extension {suffix!r} for {path}")


def find_default_input_surface(repo_root: Optional[Path] = None) -> Tuple[Optional[str], Optional[Path]]:
    """Return ``("tmd"|"mat", path)`` for a built-in example, or ``(None, None)``."""
    root = repo_root or truemap_repo_root()
    nb = root / "notebooks"
    tmd_candidates = [
        root / "examples/gelsight/circle_0mm_100g_heightmap_linear_detrend.tmd",
        Path("examples/gelsight/circle_0mm_100g_heightmap_linear_detrend.tmd"),
    ]
    for p in tmd_candidates:
        if p.exists():
            return "tmd", p.resolve()

    mat_candidates = [
        nb / "derived_surfaces.mat",
        Path("notebooks/derived_surfaces.mat"),
        Path("derived_surfaces.mat"),
    ]
    for p in mat_candidates:
        if p.exists():
            return "mat", p.resolve()
    return None, None


def _surfaces_from_tmd(
    tmd_path: Path,
    *,
    max_grid_dim: int,
    normalize_to_unit_sq: bool,
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict], Dict[str, Any]]:
    from tmd import TMD

    obj = TMD.load(str(tmd_path))
    base = np.asarray(obj.height_map, dtype=np.float64)
    if base.ndim != 2:
        raise ValueError(f"Expected 2D height map in {tmd_path}, got shape={base.shape}")
    original_shape = base.shape
    base = _maybe_downsample(base, max_grid_dim)
    base, original_sq = _maybe_normalize(base, normalize_to_unit_sq=normalize_to_unit_sq)
    raw_meta = obj.metadata if isinstance(obj.metadata, dict) else dict(obj.metadata or {})
    metadata: Dict[str, Any] = {"source": str(tmd_path), "kind": "tmd"}
    for key in ("mmpp", "x_length", "y_length", "x_offset", "y_offset", "width", "height"):
        if key in raw_meta:
            try:
                metadata[key] = float(raw_meta[key])
            except (TypeError, ValueError):
                metadata[key] = raw_meta[key]
    if base.shape != original_shape:
        metadata["downsampled_from"] = f"{original_shape[0]}x{original_shape[1]}"
        metadata["downsampled_to"] = f"{base.shape[0]}x{base.shape[1]}"
    if normalize_to_unit_sq:
        metadata["original_sq"] = original_sq
    surfaces = {1: base.copy()}
    stored_params = {1: {"source": "tmd", "path": str(tmd_path)}}
    return surfaces, stored_params, metadata


def _surfaces_from_mat(
    mat_path: Path,
    *,
    max_grid_dim: int,
    normalize_to_unit_sq: bool,
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict], Dict[str, Any]]:
    from tmd.compression import MATImporter

    payload = MATImporter().load(str(mat_path))
    raw_surfaces = payload.get("surfaces") or {}
    surfaces: Dict[int, np.ndarray] = {}
    downsample_info: Dict[int, str] = {}
    sq_info: Dict[int, float] = {}
    for i, arr in raw_surfaces.items():
        original = np.asarray(arr, dtype=np.float64)
        downsampled = _maybe_downsample(original, max_grid_dim)
        if downsampled.shape != original.shape:
            downsample_info[int(i)] = f"{original.shape[0]}x{original.shape[1]}->{downsampled.shape[0]}x{downsampled.shape[1]}"
        normalized, original_sq = _maybe_normalize(downsampled, normalize_to_unit_sq=normalize_to_unit_sq)
        surfaces[int(i)] = normalized
        if normalize_to_unit_sq and abs(original_sq - 1.0) > 1e-3:
            sq_info[int(i)] = original_sq
    if not surfaces:
        raise KeyError(f"No Surface_simulation_* arrays found in {mat_path}")
    raw_params = payload.get("parameters", {}) or {}
    stored_params: Dict[int, dict] = {}
    for i, params in raw_params.items():
        cleaned = _coerce_to_metric_dict(dict(params))
        if cleaned:
            stored_params[int(i)] = cleaned
    raw_meta = payload.get("metadata") or {}
    metadata: Dict[str, Any] = {"source": str(mat_path), "kind": "mat"}
    if isinstance(raw_meta, dict):
        for key, value in raw_meta.items():
            try:
                metadata[key] = float(value)
            except (TypeError, ValueError):
                metadata[key] = value
    if "tmd_format" in payload:
        metadata["tmd_format"] = payload["tmd_format"]
    if "tmd_version" in payload:
        metadata["tmd_version"] = payload["tmd_version"]
    if downsample_info:
        metadata["downsampled"] = downsample_info
    if sq_info:
        metadata["original_sq"] = sq_info
    return surfaces, stored_params, metadata


def _load_one_surface_file(
    path: Path,
    *,
    max_grid_dim: int,
    normalize_to_unit_sq: bool,
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict], Dict[str, Any]]:
    kind = _detect_kind(path)
    if kind == "tmd":
        return _surfaces_from_tmd(path, max_grid_dim=max_grid_dim, normalize_to_unit_sq=normalize_to_unit_sq)
    return _surfaces_from_mat(path, max_grid_dim=max_grid_dim, normalize_to_unit_sq=normalize_to_unit_sq)


def load_surfaces_for_simulation(
    input_a: Union[str, Path, None] = None,
    input_b: Union[str, Path, None] = None,
    *,
    second_surface_mode: str = "flat",
    max_grid_dim: int = 256,
    normalize_to_unit_sq: bool = True,
    repo_root: Optional[Path] = None,
) -> Tuple[Dict[int, np.ndarray], Dict[int, dict], Dict[str, Any], str]:
    """Load master/slave height maps for :func:`run_simulation`.

    When ``input_a`` is ``None``, searches default TMD/MAT paths under *repo_root*
    (see :func:`find_default_input_surface`). ``second_surface_mode`` when
    ``input_b`` is ``None``: ``flat`` | ``mirror`` | ``same_file`` (see notebook).

    Returns ``(surfaces, stored_surface_params, metadata, resolved_mode)`` where
    ``surfaces[1]`` is the master grid; use :func:`select_pair` for the ``MS, SS``
    sign convention.
    """
    root = repo_root or truemap_repo_root()

    if input_a is None:
        a_kind, found = find_default_input_surface(root)
        if found is None:
            raise FileNotFoundError(
                "No input found. Set input_a to a .tmd or .mat path, or place a default file in the repo."
            )
        input_a = found
        metadata: Dict[str, Any] = {"input_a": str(input_a), "input_a_kind": a_kind}
    else:
        input_a = Path(input_a)
        metadata = {"input_a": str(input_a.resolve()), "input_a_kind": _detect_kind(input_a)}

    surfaces_a, params_a, meta_a = _load_one_surface_file(
        Path(input_a), max_grid_dim=max_grid_dim, normalize_to_unit_sq=normalize_to_unit_sq
    )
    a_first = min(surfaces_a)
    base = surfaces_a[a_first]

    surfaces: Dict[int, np.ndarray] = {1: base.copy()}
    stored_params: Dict[int, Any] = {}
    if a_first in params_a:
        stored_params[1] = dict(params_a[a_first])

    metadata["second_surface_mode"] = second_surface_mode
    for key, value in meta_a.items():
        if key not in {"source", "kind"}:
            metadata[f"a_{key}"] = value

    if input_b is not None:
        input_b = Path(input_b)
        b_kind = _detect_kind(input_b)
        surfaces_b, params_b, meta_b = _load_one_surface_file(
            Path(input_b), max_grid_dim=max_grid_dim, normalize_to_unit_sq=normalize_to_unit_sq
        )
        b_first = min(surfaces_b)
        b_base = surfaces_b[b_first]
        if b_base.shape != base.shape:
            raise ValueError(f"Slave shape {b_base.shape} from {input_b} must match master {base.shape}")
        surfaces[2] = b_base.copy()
        if b_first in params_b:
            stored_params[2] = dict(params_b[b_first])
        metadata["input_b"] = str(input_b.resolve())
        metadata["input_b_kind"] = b_kind
        for key, value in meta_b.items():
            if key not in {"source", "kind"}:
                metadata[f"b_{key}"] = value
        return surfaces, stored_params, metadata, "two_files"

    mode = (second_surface_mode or "flat").lower()
    if mode == "flat":
        surfaces[2] = np.zeros_like(base, dtype=np.float64)
        stored_params[2] = {"source": "flat"}
        metadata["slave"] = "flat_zero"
        return surfaces, stored_params, metadata, "flat"
    if mode == "mirror":
        surfaces[2] = np.flipud(np.fliplr(base)).copy()
        stored_params[2] = {"source": "mirrored_master"}
        metadata["slave"] = "mirror"
        return surfaces, stored_params, metadata, "mirror"
    if mode == "same_file":
        for idx, arr in surfaces_a.items():
            if int(idx) == a_first:
                continue
            new_idx = int(idx) if int(idx) != 1 else max(surfaces_a) + 1
            surfaces[new_idx] = arr.copy()
            if int(idx) in params_a:
                stored_params[new_idx] = dict(params_a[int(idx)])
        if 2 not in surfaces:
            raise ValueError(
                f"second_surface_mode='same_file' requires INPUT_A to contain >=2 surfaces; found {len(surfaces_a)}."
            )
        metadata["slave"] = "same_file"
        return surfaces, stored_params, metadata, "same_file"

    raise ValueError(f"Unknown second_surface_mode={second_surface_mode!r}; expected flat|mirror|same_file")


def export_final_state_mat(
    path: Union[str, Path],
    *,
    ms_initial: np.ndarray,
    ss_initial: np.ndarray,
    final_state: Mapping[str, Any],
    surface_metrics_fn: Optional[Callable[[np.ndarray], Mapping[str, float]]] = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Export canonical ``Surface_simulation_*`` bundle via :class:`tmd.compression.MATExporter`."""
    from tmd.compression import MATExporter

    fn = surface_metrics_fn or height_matrix_surface_metrics
    out = Path(path)
    surfaces: Dict[int, np.ndarray] = {
        1: np.asarray(ms_initial, dtype=np.float64),
        2: np.asarray(ss_initial, dtype=np.float64),
        3: np.asarray(final_state["MS"], dtype=np.float64),
        4: np.asarray(final_state["SS"], dtype=np.float64),
        5: np.asarray(final_state["cumulative_wear_MS"], dtype=np.float64),
    }
    p1 = dict(fn(np.asarray(ms_initial, dtype=np.float64)))
    p3 = dict(fn(np.asarray(final_state["MS"], dtype=np.float64)))
    parameters: Dict[int, Dict[str, Any]] = {1: p1, 3: p3}
    meta: Dict[str, Any] = {k: (v if isinstance(v, (str, int, float, bool)) else str(v)) for k, v in (extra_metadata or {}).items()}
    MATExporter().export({"surfaces": surfaces, "parameters": parameters, "metadata": meta}, str(out))
    return out


def ms_snapshots_to_sequence(
    snapshots: Mapping[int, Mapping[str, np.ndarray]],
    *,
    name: str = "wear-ms",
    field: str = "MS",
    pitch_metadata: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Build a :class:`tmd.core.sequence.TMDSequence` from snapshot dicts (e.g. :func:`make_snapshot`).

    Each frame's ``frame_metadata`` includes ``step`` plus optional keys from
    *pitch_metadata* (typically ``x_length``, ``y_length``, ``width``, ``height``)
    so :func:`pixel_pitch_mm_from_metadata` can derive mm pitch for
    :meth:`TMDSequence.sequential_wear_metrics`.

    Example::

        seq = ms_snapshots_to_sequence(snapshots, pitch_metadata=truemap_metadata)
        fr0 = seq.get_frame(0)
        meta0 = seq.get_frame_metadata(0) or {}
        dx, dy = pixel_pitch_mm_from_metadata(simulation_metadata_to_pitch(meta0), fr0.shape)
        wear = seq.sequential_wear_metrics(dx_mm=dx, dy_mm=dy, reference_index=0)
    """
    from tmd.core.sequence import TMDSequence

    seq = TMDSequence(name=name)
    pitch_clean = simulation_metadata_to_pitch(pitch_metadata or {})
    for step in sorted(int(s) for s in snapshots.keys()):
        snap = snapshots[step]
        arr = np.asarray(snap[field], dtype=np.float64)
        meta: Dict[str, Any] = {"step": step, **pitch_clean}
        seq.add_frame(arr, timestamp=f"step_{step}", metadata=meta)
    return seq
