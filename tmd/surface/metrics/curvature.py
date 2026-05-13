"""
Areal differential geometry from height fields:

- Mean curvature ``H`` (same convention as ``CurvatureMapGenerator`` mean mode).
- Per-pixel surface unit normals and lateral pixel-spacing inference.
- Sphere-binned orientation histograms (and before/after triptych) used by the
  Plotly normal-orientation visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

import numpy as np


def mean_curvature_from_derivatives(
    fx: np.ndarray,
    fy: np.ndarray,
    fxx: np.ndarray,
    fxy: np.ndarray,
    fyy: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """Mean curvature H from first/second partials of z = h(x, y)."""
    p = fx**2 + fy**2
    q = p + 1.0
    h = 0.5 * ((1 + fy**2) * fxx - 2 * fx * fy * fxy + (1 + fx**2) * fyy) / (q ** (3 / 2))
    return np.nan_to_num(h * scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)


def mean_curvature(
    height_map: np.ndarray,
    cell_size_x: float = 1.0,
    cell_size_y: float = 1.0,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Mean curvature H of graph z = h(x, y), consistent with ``CurvatureMapGenerator`` mean mode.

    Uses ``np.gradient`` with physical cell sizes ``cell_size_x``, ``cell_size_y``.
    """
    fx, fy = np.gradient(height_map, cell_size_x, cell_size_y)
    fxx, fxy = np.gradient(fx, cell_size_x, cell_size_y)
    _fyx, fyy = np.gradient(fy, cell_size_x, cell_size_y)
    return mean_curvature_from_derivatives(fx, fy, fxx, fxy, fyy, scale)


# --- Surface unit normals and orientation histograms ------------------------


def pixel_spacing_dy_dx_from_metadata(
    metadata: Mapping[str, Any] | None,
    shape: tuple[int, ...],
    *,
    key_prefix: str = "a_",
) -> tuple[float, float]:
    """Lateral spacing in **mm** for ``numpy.gradient`` on ``Z[row, col]`` (see normal map generators)."""
    if len(shape) < 2:
        raise ValueError("shape must be at least 2D (rows, cols)")
    rows, cols = int(shape[0]), int(shape[1])
    meta: dict[str, Any] = dict(metadata or {})
    p = key_prefix

    def pick(*keys: str) -> Any:
        for k in keys:
            if k in meta and meta[k] is not None:
                return meta[k]
        return None

    mmpp = pick(f"{p}mmpp", "mmpp")
    if mmpp is not None:
        m = float(mmpp)
        if m > 0.0:
            return (m, m)

    xl = pick(f"{p}x_length", "x_length")
    yl = pick(f"{p}y_length", "y_length")
    if xl is not None and yl is not None:
        dx = float(xl) / max(cols, 1)
        dy = float(yl) / max(rows, 1)
        if dx > 0.0 and dy > 0.0:
            return (dy, dx)
    return (1.0, 1.0)


def _unit_normals_stack(z: np.ndarray, dy: float, dx: float) -> np.ndarray:
    if z.ndim != 2:
        raise ValueError("Z must be 2D")
    if dy <= 0.0 or dx <= 0.0:
        raise ValueError("dy and dx must be positive")
    zy, zx = np.gradient(z, float(dy), float(dx))
    nx, ny, nz = -zx, -zy, np.ones_like(zx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / norm, ny / norm, nz / norm], axis=-1)


def surface_normals_flat(
    Z: np.ndarray,
    *,
    dy: float = 1.0,
    dx: float = 1.0,
) -> np.ndarray:
    """Per-pixel unit normals ``(N, 3)`` with ``n proportional to (-dZ/dx, -dZ/dy, 1)`` normalized."""
    z = np.asarray(Z, dtype=np.float64)
    if not np.any(np.isfinite(z)):
        raise ValueError("Z has no finite values")
    return _unit_normals_stack(z, dy, dx).reshape(-1, 3)


def surface_normals_grid(
    Z: np.ndarray,
    *,
    dy: float = 1.0,
    dx: float = 1.0,
) -> np.ndarray:
    """Unit normals ``(H, W, 3)``."""
    return _unit_normals_stack(np.asarray(Z, dtype=np.float64), dy, dx)


def normal_density_on_sphere(
    normals: np.ndarray,
    n_lat: int = 64,
    n_lon: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = normals[:, 0]
    y = normals[:, 1]
    z = normals[:, 2]
    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    lon = np.arctan2(y, x)
    lat_edges = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 1)
    lon_edges = np.linspace(-np.pi, np.pi, n_lon + 1)
    h, _, _ = np.histogram2d(lat, lon, bins=[lat_edges, lon_edges], density=False)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    cos_lat = np.cos(np.clip(lat_centers, -np.pi / 2 + 1e-6, np.pi / 2 - 1e-6))
    h = h / np.maximum(cos_lat[:, None], 0.02)
    return lat_centers, lon_centers, h.astype(np.float64)


@dataclass(frozen=True)
class OrientationHistograms:
    lat_c: np.ndarray
    lon_c: np.ndarray
    h_before: np.ndarray
    h_after: np.ndarray
    h_delta: np.ndarray
    common_max: float
    delta_lim: float


def orientation_histograms(
    before: np.ndarray,
    after: np.ndarray,
    *,
    n_lat: int = 64,
    n_lon: int = 128,
    dy: float = 1.0,
    dx: float = 1.0,
) -> OrientationHistograms:
    n_b = surface_normals_flat(before, dy=dy, dx=dx)
    n_a = surface_normals_flat(after, dy=dy, dx=dx)
    lat_c, lon_c, h_b = normal_density_on_sphere(n_b, n_lat, n_lon)
    _, _, h_a = normal_density_on_sphere(n_a, n_lat, n_lon)
    h_b = h_b / max(float(h_b.sum()), 1e-12)
    h_a = h_a / max(float(h_a.sum()), 1e-12)
    h_d = h_a - h_b
    common_max = max(float(h_b.max()), float(h_a.max()), 1e-12)
    delta_lim = max(abs(float(h_d.min())), abs(float(h_d.max())), 1e-12)
    return OrientationHistograms(lat_c, lon_c, h_b, h_a, h_d, common_max, delta_lim)


def normal_orientation_summary(
    Z: np.ndarray,
    *,
    dy: float = 1.0,
    dx: float = 1.0,
) -> Dict[str, float]:
    n = surface_normals_flat(Z, dy=dy, dx=dx)
    mean = n.mean(axis=0)
    mean_len = float(np.linalg.norm(mean))
    mean = mean / max(mean_len, 1e-12)
    azim = float(np.degrees(np.arctan2(mean[1], mean[0])))
    elev = float(np.degrees(np.arcsin(np.clip(mean[2], -1.0, 1.0))))
    return {
        "mean_resultant_length": mean_len,
        "mean_normal_azim_deg": azim,
        "mean_normal_elev_deg": elev,
    }


def per_pixel_normal_change_deg(
    Z_before: np.ndarray,
    Z_after: np.ndarray,
    *,
    stride: int = 1,
    dy: float = 1.0,
    dx: float = 1.0,
) -> np.ndarray:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    zb = np.asarray(Z_before, dtype=np.float64)[::stride, ::stride]
    za = np.asarray(Z_after, dtype=np.float64)[::stride, ::stride]
    if zb.shape != za.shape:
        raise ValueError("Z_before and Z_after must match shape (after stride)")
    nb = surface_normals_grid(zb, dy=dy * stride, dx=dx * stride)
    na = surface_normals_grid(za, dy=dy * stride, dx=dx * stride)
    dot = np.sum(nb * na, axis=-1)
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))
