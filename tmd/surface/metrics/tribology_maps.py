"""Normalized 0–1 grayscale proxy maps built on :mod:`tmd.surface.metrics.tribology` primitives."""

from __future__ import annotations

import numpy as np

from .tribology import debris_pocket_map, interfacial_shear_proxy_map


def _percentile_norm01(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo_v, hi_v = np.percentile(x, (lo, hi))
    if hi_v <= lo_v:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo_v) / (hi_v - lo_v)
    return np.clip(y, 0.0, 1.0).astype(np.float32, copy=False)


def shear_hazard_map_01(
    height_map: np.ndarray,
    metadata: dict,
    *,
    roughness_sigma: float = 10.0,
    sq_window: int = 7,
    gaussian_sigma: float = 0.0,
    plane_removal: str = "none",
) -> np.ndarray:
    """Interfacial shear hazard proxy, percentile-normalized to ``float32`` in ``[0, 1]``."""
    raw = interfacial_shear_proxy_map(
        height_map,
        metadata,
        roughness_sigma=float(roughness_sigma),
        sq_window=int(sq_window),
        gaussian_sigma=float(gaussian_sigma),
        plane_removal=str(plane_removal),
    )
    return _percentile_norm01(raw)


def debris_pocket_map_01(
    height_map: np.ndarray,
    metadata: dict,
    *,
    valley_percentile: float = 10.0,
    slope_percentile: float = 40.0,
    pit_dilate: int = 3,
    plane_removal: str = "none",
) -> np.ndarray:
    """Debris pocket heuristic score, percentile-normalized to ``float32`` in ``[0, 1]``."""
    out = debris_pocket_map(
        height_map,
        metadata,
        valley_percentile=float(valley_percentile),
        slope_percentile=float(slope_percentile),
        pit_dilate=int(pit_dilate),
        plane_removal=str(plane_removal),
    )["pocket_score"]
    return _percentile_norm01(np.asarray(out, dtype=np.float64))
