"""Texture-direction spectrum and compact areal metrics summary.

Adds two areal helpers that sit alongside the bearing-curve and proxy-map utilities
in :mod:`tmd.surface.metrics.tribology`:

- :func:`texture_direction_spectrum` — FFT angular bin returning the dominant
  texture direction ``Std_deg`` (Sun, 2025 convention) along with the full
  per-angle magnitude spectrum.
- :func:`surface_metrics_summary` — single-call ISO 25178-flavoured bundle of
  Sq, Sdq, Sdr, Ssk, Sku, Spc, Spd, and Std_deg suitable for tabular before /
  after comparisons.

Reference:
    Sun, D. (2025). MATLAB code for "Surface wear characterized by height matrix in
    relation to texture direction (Std) and other surface parameters" (Version 1).
    Mendeley Data. https://doi.org/10.17632/v635kysfr2.1 (CC BY 4.0).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy import ndimage


__all__ = [
    "surface_metrics_summary",
    "texture_direction_spectrum",
]


def texture_direction_spectrum(Z: np.ndarray, angular_step: int = 2) -> Dict[str, Any]:
    """Sum the centered FFT magnitude into angular bins of width ``angular_step``.

    Parameters
    ----------
    Z:
        2D height field. NaNs are not tolerated; pass a finite array.
    angular_step:
        Bin width in degrees over ``[0, 180)``. Smaller bins give finer angular
        resolution at the cost of noisier per-bin magnitude sums.

    Returns
    -------
    dict
        ``{"angles_deg": ndarray, "spectrum": ndarray, "Std_deg": float}`` where
        ``Std_deg`` is the bin centre with the largest summed magnitude (dominant
        texture direction).
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D, got shape={Z.shape}")
    step = int(angular_step)
    if step <= 0 or step >= 180:
        raise ValueError("angular_step must be an integer in (0, 180)")

    F = np.fft.fftshift(np.fft.fft2(Z))
    mag = np.abs(F)
    rows, cols = Z.shape
    u = np.arange(cols) - cols // 2
    v = np.arange(rows) - rows // 2
    U, V = np.meshgrid(u, v)
    theta = np.mod(np.arctan2(V, U), np.pi)

    angles = np.arange(0, 180, step)
    spectrum = np.zeros_like(angles, dtype=np.float64)
    half = np.deg2rad(step / 2.0)
    for i, deg in enumerate(angles):
        a = np.deg2rad(deg)
        spectrum[i] = mag[np.abs(theta - a) <= half].sum()

    dominant = float(angles[int(np.argmax(spectrum))])
    return {"angles_deg": angles, "spectrum": spectrum, "Std_deg": dominant}


def surface_metrics_summary(Z: np.ndarray) -> Dict[str, float]:
    """Single-call areal metrics bundle (Sq, Sdq, Sdr, Ssk, Sku, Spc, Spd, Std_deg).

    Notes
    -----
    These values follow the Sun (2025) reference for the wear-simulation
    notebooks and are independent of Surfalize. They are not licensed-tied
    ISO 25178 values from a metrology vendor; treat them as fast research
    proxies suitable for before / after comparisons.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2D, got shape={Z.shape}")
    Zy, Zx = np.gradient(Z)
    Sq = float(np.sqrt(np.mean(Z**2)))
    Sdq = float(np.sqrt(np.mean(Zx**2 + Zy**2)))
    actual_area = float(np.sum(np.sqrt(1.0 + Zx[:-1, :-1] ** 2 + Zy[:-1, :-1] ** 2)))
    nominal_area = float((Z.shape[0] - 1) * (Z.shape[1] - 1))
    Sdr = (actual_area - nominal_area) / nominal_area if nominal_area > 0 else 0.0
    Ssk = float(np.mean(Z**3) / (Sq**3)) if Sq > 0 else 0.0
    Sku = float(np.mean(Z**4) / (Sq**4)) if Sq > 0 else 0.0
    Spc = float(-0.5 * np.mean(ndimage.laplace(Z, mode="nearest")))
    local_max = ndimage.maximum_filter(Z, size=3, mode="nearest")
    peaks = (Z == local_max) & (Z > Z.mean())
    peaks[[0, -1], :] = False
    peaks[:, [0, -1]] = False
    Spd = float(peaks.sum() / Z.size)
    Std = texture_direction_spectrum(Z)["Std_deg"]
    return {
        "Sq": Sq,
        "Sdq": Sdq,
        "Sdr": Sdr,
        "Ssk": Ssk,
        "Sku": Sku,
        "Spc": Spc,
        "Spd": Spd,
        "Std_deg": Std,
    }
