"""
Tribology-oriented metrics from height maps (NumPy core; optional Surfalize plane removal).

Includes bearing / material ratio helpers, debris and shear proxy maps, preferred slip axis,
bearing-area curves, summit maps, optional Surfalize ``Surface.level()`` plane removal (GPL-3.0
when installed), and a matplotlib PNG dashboard export.

Use ``plane_removal="surfalize"`` for least-squares leveling via Surfalize; defaults preserve
prior behaviour (no extra plane pass).

Sign convention for ``bearing_area_curve``: heights are leveled to zero mean;
``separation`` is a cut height in the same units as ``height_map``; the rigid
counterface lies at that height; **contact** is counted where ``z_leveled >= separation``.
Then ``area_fraction`` decreases as ``separation`` increases (fewer peaks reach high planes).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy import ndimage

from tmd.surface.defects import detect_surface_defects
from tmd.surface.filters import (
    calculate_surface_gradient,
    calculate_surface_isotropy,
    extract_roughness,
    global_texture_angle_rad,
    gradient_slant_angle_rad,
    local_texture_angle_and_coherence,
)
from tmd.surface.types import DefectDetectionConfig

from .curvature import mean_curvature

# ---------------------------------------------------------------------------
# Optional Surfalize (GPL-3.0 when installed); lazy import inside helpers.
# ---------------------------------------------------------------------------


def surfalize_available() -> bool:
    try:
        import surfalize  # noqa: F401

        return True
    except ImportError:
        return False


def _step_micrometers(metadata: Optional[Dict[str, Any]], shape: Tuple[int, int]) -> Tuple[float, float]:
    """Surfalize ``Surface`` expects step in µm/pixel (same as roughness_common)."""
    h, w = shape
    meta = metadata or {}
    if "x_length" in meta and "y_length" in meta and w > 0 and h > 0:
        xl = float(meta["x_length"])
        yl = float(meta["y_length"])
        return (xl / max(w, 1)) * 1000.0, (yl / max(h, 1)) * 1000.0
    if "mmpp" in meta:
        m = float(meta["mmpp"]) * 1000.0
        return m, m
    return 1.0, 1.0


def _slim_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    slim: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            slim[str(k)] = v
    return slim


def level_height_map_surfalize(
    height_map: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[np.ndarray]:
    """
    Return least-squares leveled heights as ``float32``, or ``None`` if Surfalize is missing.

    Uses ``Surface(...).level().data`` — same plane model as ISO roughness workflows.
    """
    try:
        from surfalize import Surface
    except ImportError:
        return None

    hm = np.ascontiguousarray(np.asarray(height_map, dtype=np.float32))
    if hm.ndim != 2:
        raise ValueError("height_map must be 2D")
    step_x, step_y = _step_micrometers(metadata, hm.shape)
    meta = metadata or {}
    slim = _slim_metadata(meta) if meta else {}
    surf = Surface(hm, step_x, step_y, metadata=slim or None)
    leveled = surf.level()
    return np.asarray(leveled.data, dtype=np.float32)


def level_height_map_surfalize_or_warn(
    height_map: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Like ``level_height_map_surfalize`` but falls back to mean removal and warns once path."""
    out = level_height_map_surfalize(height_map, metadata)
    if out is not None:
        return out
    warnings.warn(
        "Surfalize is not installed; falling back to mean plane removal for plane_removal='surfalize'. "
        'Install with: pip install "truemapdata[roughness]"',
        stacklevel=3,
    )
    z = np.asarray(height_map, dtype=np.float32)
    return (z - float(np.mean(z))).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Abbott / material ratio, debris-pocket score, shear proxy (NumPy)
# ---------------------------------------------------------------------------

NormalizeMode = Literal["none", "minmax", "p98"]


def remove_least_squares_plane(z: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("height map must be 2D")
    h, w = z.shape
    valid = np.isfinite(z)
    if not np.any(valid):
        raise ValueError("no finite height samples")

    yy, xx = np.indices((h, w), dtype=np.float64)
    xv = xx[valid]
    yv = yy[valid]
    zv = z[valid]
    X = np.column_stack([xv, yv, np.ones(len(zv))])
    coef, _, rank, _ = np.linalg.lstsq(X, zv, rcond=None)
    if rank < 3:
        plane = np.full_like(z, np.nan, dtype=np.float64)
        plane[valid] = float(np.mean(zv))
    else:
        plane = coef[0] * xx + coef[1] * yy + coef[2]
    leveled = z.copy()
    leveled[valid] = z[valid] - plane[valid]
    meta = {
        "plane_coef_ax_by_c": [float(coef[0]), float(coef[1]), float(coef[2])],
        "n_finite": int(valid.sum()),
    }
    return leveled, meta


def material_ratio_curve(
    z_leveled: np.ndarray,
    *,
    n_depth_samples: int = 256,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    zl = np.asarray(z_leveled, dtype=np.float64)
    flat = zl[np.isfinite(zl)].ravel()
    if flat.size == 0:
        raise ValueError("no finite heights for bearing curve")

    z_max = float(flat.max())
    z_min = float(flat.min())
    span = z_max - z_min
    if span <= 0:
        depths = np.array([0.0], dtype=np.float64)
        rmr = np.array([100.0], dtype=np.float64)
        return depths, rmr, {"z_max": z_max, "z_min": z_min, "span": span, "degenerate": True}

    n = int(max(2, n_depth_samples))
    thresholds = np.linspace(z_max, z_min, n, dtype=np.float64)
    rmr = np.array([100.0 * float(np.mean(flat >= t)) for t in thresholds], dtype=np.float64)
    depths = (z_max - thresholds).astype(np.float64)
    info = {"z_max": z_max, "z_min": z_min, "span": span, "degenerate": False}
    return depths, rmr, info


def rmr_at_depths(
    depths: np.ndarray,
    rmr_percent: np.ndarray,
    query_depths: np.ndarray,
) -> np.ndarray:
    d = np.asarray(depths, dtype=np.float64)
    r = np.asarray(rmr_percent, dtype=np.float64)
    q = np.asarray(query_depths, dtype=np.float64)
    if d.size < 2:
        return np.full_like(q, r[0] if r.size else np.nan, dtype=np.float64)
    return np.interp(q, d, r, left=float(r[0]), right=float(r[-1]))


def depth_at_rmr(
    depths: np.ndarray,
    rmr_percent: np.ndarray,
    query_rmr: np.ndarray,
) -> np.ndarray:
    d = np.asarray(depths, dtype=np.float64)
    r = np.asarray(rmr_percent, dtype=np.float64)
    q = np.asarray(query_rmr, dtype=np.float64)
    if d.size < 2:
        return np.full_like(q, np.nan, dtype=np.float64)
    if r[-1] < r[0]:
        r_s = r[::-1]
        d_s = d[::-1]
    else:
        r_s, d_s = r, d
    return np.interp(q, r_s, d_s, left=float(d_s[0]), right=float(d_s[-1]))


def bearing_analysis(
    z: np.ndarray,
    *,
    n_depth_samples: int = 256,
    rmr_query_depths: Optional[List[float]] = None,
) -> Dict[str, Any]:
    leveled, plane_meta = remove_least_squares_plane(z)
    depths, rmr, curve_info = material_ratio_curve(leveled, n_depth_samples=n_depth_samples)
    out: Dict[str, Any] = {
        "plane": plane_meta,
        "curve": curve_info,
        "depths": depths.tolist(),
        "rmr_percent": rmr.tolist(),
    }
    if rmr_query_depths:
        q = np.asarray(rmr_query_depths, dtype=np.float64)
        out["rmr_at_depth"] = {
            "depths": rmr_query_depths,
            "rmr_percent": rmr_at_depths(depths, rmr, q).tolist(),
        }
    return out


def _mad_std(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad <= 1e-12:
        return max(float(np.std(x)), 1e-12)
    return max(1.4826 * mad, 1e-12)


def debris_pocket_score(
    height_map: np.ndarray,
    *,
    gaussian_sigma: float = 1.0,
    zscore_threshold: float = 1.8,
    min_area: int = 6,
    gradient_percentile: float = 35.0,
    valley_zscore: float = 1.2,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    z = np.asarray(height_map, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("height_map must be 2D")

    cfg = DefectDetectionConfig(
        gaussian_sigma=gaussian_sigma,
        zscore_threshold=zscore_threshold,
        min_area=min_area,
        output_mode="standard",
    )
    defects = detect_surface_defects(z.astype(np.float32, copy=False), cfg)
    pits_mask = np.asarray(defects["defects"]["pits"].get("mask", np.zeros_like(z, dtype=bool)), dtype=bool)

    smooth = ndimage.gaussian_filter(z, sigma=max(0.0, gaussian_sigma))
    residual = smooth - np.median(smooth)
    sig = _mad_std(residual.astype(np.float64))
    zscore = residual / sig
    valley_mask = zscore < -float(valley_zscore)

    gy, gx = np.gradient(smooth.astype(np.float64))
    gmag = np.hypot(gx, gy)
    thresh = float(np.percentile(gmag[np.isfinite(gmag)], gradient_percentile))
    low_grad = gmag <= thresh

    pocket = pits_mask.astype(np.float64) * 0.45 + valley_mask.astype(np.float64) * 0.35 + low_grad.astype(
        np.float64
    ) * 0.20
    pocket = np.clip(pocket, 0.0, 1.0)
    pocket = np.where(np.isfinite(z), pocket, 0.0)

    meta: Dict[str, Any] = {
        "mean_score": float(np.mean(pocket)),
        "p90_score": float(np.percentile(pocket[np.isfinite(pocket)], 90)),
        "fraction_high_risk": float(np.mean(pocket >= 0.65)),
    }
    return pocket.astype(np.float32), meta


def local_rms(height: np.ndarray, window: int) -> np.ndarray:
    z = np.asarray(height, dtype=np.float64)
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    size = int(window)
    z2 = ndimage.uniform_filter(z * z, size=size, mode="nearest")
    zm = ndimage.uniform_filter(z, size=size, mode="nearest")
    var = np.clip(z2 - zm * zm, 0.0, None)
    return np.sqrt(var)


def shear_proxy_map(
    height: np.ndarray,
    *,
    window: int = 7,
    normalize: NormalizeMode = "p98",
) -> np.ndarray:
    z = np.asarray(height, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("height must be 2D")
    gy, gx = np.gradient(z)
    gmag = np.hypot(gx, gy)
    lap = ndimage.laplace(z)
    rms = local_rms(z, window)
    proxy = gmag * np.abs(lap) * rms
    proxy = np.where(np.isfinite(proxy), proxy, 0.0)

    if normalize == "none":
        return proxy.astype(np.float64, copy=False)
    if normalize == "minmax":
        pmin = float(np.nanmin(proxy))
        pmax = float(np.nanmax(proxy))
        if pmax <= pmin:
            return np.zeros_like(proxy)
        return (proxy - pmin) / (pmax - pmin)
    if normalize == "p98":
        flat = proxy[np.isfinite(proxy)]
        if flat.size == 0:
            return np.zeros_like(proxy)
        hi = float(np.percentile(flat, 98))
        if hi <= 0:
            return np.zeros_like(proxy)
        return np.clip(proxy / hi, 0.0, 1.0)
    raise ValueError(f"unknown normalize mode: {normalize!r}")


def shear_proxy_uint8(
    height: np.ndarray,
    *,
    window: int = 7,
    normalize: NormalizeMode = "p98",
) -> Tuple[np.ndarray, np.ndarray]:
    raw = shear_proxy_map(height, window=window, normalize=normalize)
    u8 = (np.clip(raw, 0.0, 1.0) * 255.0).astype(np.uint8)
    return raw, u8


# ---------------------------------------------------------------------------
# Core tribology maps and curves
# ---------------------------------------------------------------------------


def _preprocess_plane(
    height_map: np.ndarray,
    metadata: Optional[Dict[str, Any]],
    plane_removal: str,
) -> np.ndarray:
    """
    Optional large-scale plane handling before local tribology metrics.

    ``plane_removal``:
        - ``"none"``: use heights as-is (default, matches earlier releases).
        - ``"mean"``: subtract global mean.
        - ``"median"``: subtract global median.
        - ``"surfalize"``: Surfalize ``Surface.level()`` (falls back to ``"mean"`` with a warning if unavailable).
    """
    z = np.asarray(height_map, dtype=np.float32)
    pr = (plane_removal or "none").strip().lower()
    if pr in ("none", ""):
        return z.copy()
    if pr == "mean":
        return (z - float(np.mean(z))).astype(np.float32, copy=False)
    if pr == "median":
        return (z - float(np.median(z))).astype(np.float32, copy=False)
    if pr == "surfalize":
        leveled = level_height_map_surfalize(z, metadata)
        if leveled is not None:
            return leveled
        warnings.warn(
            "Surfalize not installed; using mean plane removal instead of plane_removal='surfalize'. "
            'Install with: pip install "truemapdata[roughness]"',
            UserWarning,
            stacklevel=2,
        )
        return (z - float(np.mean(z))).astype(np.float32, copy=False)
    raise ValueError("plane_removal must be one of: none, mean, median, surfalize")


def cell_sizes_from_metadata(metadata: Optional[Dict[str, Any]], shape: Tuple[int, int]) -> Tuple[float, float]:
    """Return (dx, dy) physical step per pixel (same convention as map generators)."""
    h, w = shape
    meta = metadata or {}
    if "x_length" in meta and "y_length" in meta and w > 0 and h > 0:
        return float(meta["x_length"]) / float(w), float(meta["y_length"]) / float(h)
    if "mmpp" in meta:
        m = float(meta["mmpp"])
        return m, m
    return 1.0, 1.0


def _circular_mean_angle_weighted(angle: np.ndarray, weight: np.ndarray) -> Optional[float]:
    w = np.maximum(np.asarray(weight, dtype=np.float64).ravel(), 0.0)
    a = np.asarray(angle, dtype=np.float64).ravel()
    if w.size == 0 or np.sum(w) < 1e-20:
        return None
    c = float(np.sum(w * np.cos(2.0 * a)))
    s = float(np.sum(w * np.sin(2.0 * a)))
    if c * c + s * s < 1e-24:
        return None
    return float(0.5 * np.arctan2(s, c))


def _asymmetry_from_three_angles(theta: List[Optional[float]]) -> float:
    """1 - |mean exp(i 2θ)| over defined angles; undefined angles skipped."""
    vecs: List[Tuple[float, float]] = []
    for t in theta:
        if t is None:
            continue
        vecs.append((float(np.cos(2.0 * t)), float(np.sin(2.0 * t))))
    if not vecs:
        return 1.0
    mx = sum(v[0] for v in vecs) / len(vecs)
    my = sum(v[1] for v in vecs) / len(vecs)
    return float(np.clip(1.0 - np.hypot(mx, my), 0.0, 1.0))


def preferred_slip_axis(
    height_map: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    gaussian_sigma: float = 0.0,
    defect_config: Optional[DefectDetectionConfig] = None,
    include_anomaly_angle: bool = True,
    plane_removal: str = "none",
) -> Dict[str, Any]:
    """
    Fuse gradient-based global lay, PSD dominant angle, and optional anomaly-weighted gradient angle.

    Returns axis in radians (image x toward y); ``axis_deg`` in degrees; ``asymmetry`` in [0, 1]
    (higher = more disagreement between available direction cues).
    """
    z = np.asarray(height_map, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError("height_map must be 2D")
    z = _preprocess_plane(z, metadata, plane_removal)
    dx, dy = cell_sizes_from_metadata(metadata, z.shape)
    smooth = ndimage.gaussian_filter(z, sigma=max(0.0, float(gaussian_sigma))).astype(np.float32, copy=False)

    gx, gy = calculate_surface_gradient(smooth, dx=dx, dy=dy, scale=1.0)
    theta_grad = global_texture_angle_rad(gx, gy)

    pixel_size = float(0.5 * (abs(dx) + abs(dy))) if (dx or dy) else 1.0
    iso = calculate_surface_isotropy(smooth, pixel_size=pixel_size)
    theta_psd = float(iso["dominant_angle"])
    psd_directionality = float(iso["directionality"])

    theta_anom: Optional[float] = None
    dir_response: Optional[np.ndarray] = None
    if include_anomaly_angle:
        cfg = defect_config or DefectDetectionConfig()
        defects = detect_surface_defects(smooth, cfg, output_mode="standard", include_responses=True)
        entry = defects["defects"].get("directionality_anomalies")
        if entry and "response" in entry:
            dir_response = np.asarray(entry["response"], dtype=np.float64)
            ang = gradient_slant_angle_rad(
                np.asarray(gx, dtype=np.float64),
                np.asarray(gy, dtype=np.float64),
            )
            theta_anom = _circular_mean_angle_weighted(ang, dir_response)

    asymmetry = _asymmetry_from_three_angles([theta_grad, theta_psd, theta_anom])

    # Fused axis: PSD direction weighted by directionality, gradient otherwise
    blend_w = np.clip(psd_directionality, 0.0, 1.0)
    fused_sin = (1.0 - blend_w) * np.sin(2.0 * theta_grad) + blend_w * np.sin(2.0 * theta_psd)
    fused_cos = (1.0 - blend_w) * np.cos(2.0 * theta_grad) + blend_w * np.cos(2.0 * theta_psd)
    if theta_anom is not None and dir_response is not None and float(np.mean(dir_response)) > 0.05:
        aw = min(0.35, float(np.mean(dir_response)))
        fused_sin = (1.0 - aw) * fused_sin + aw * np.sin(2.0 * theta_anom)
        fused_cos = (1.0 - aw) * fused_cos + aw * np.cos(2.0 * theta_anom)
    axis_rad = float(0.5 * np.arctan2(fused_sin, fused_cos))

    return {
        "axis_rad": axis_rad,
        "axis_deg": float(np.degrees(axis_rad)),
        "asymmetry": asymmetry,
        "gradient_angle_rad": float(theta_grad),
        "psd_dominant_angle_rad": theta_psd,
        "psd_directionality": psd_directionality,
        "anomaly_weighted_angle_rad": theta_anom,
        "isotropy_index": float(iso["isotropy_index"]),
    }


def interfacial_shear_proxy_map(
    height_map: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    roughness_sigma: float = 10.0,
    sq_window: int = 7,
    gaussian_sigma: float = 0.0,
    plane_removal: str = "none",
) -> np.ndarray:
    """
    Relative hazard map: |∇z| × |H| × local Sq (high-pass RMS in a window).

    Not a physical coefficient of friction; comparable across one material pair / workflow.
    """
    z = np.asarray(height_map, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError("height_map must be 2D")
    z = _preprocess_plane(z, metadata, plane_removal)
    dx, dy = cell_sizes_from_metadata(metadata, z.shape)
    smooth = ndimage.gaussian_filter(z, sigma=max(0.0, float(gaussian_sigma))).astype(np.float32, copy=False)

    gx, gy = calculate_surface_gradient(smooth, dx=dx, dy=dy, scale=1.0)
    slope = np.sqrt(gx * gx + gy * gy)
    h_curv = np.abs(mean_curvature(smooth, cell_size_x=dx, cell_size_y=dy, scale=1.0)).astype(np.float32)

    rough = extract_roughness(smooth, sigma=float(roughness_sigma)).astype(np.float32)
    win = int(max(3, sq_window))
    if win % 2 == 0:
        win += 1
    rough_sq = ndimage.uniform_filter(rough * rough, size=win)
    sq_local = np.sqrt(np.maximum(rough_sq, 0.0))

    hazard = slope.astype(np.float64) * h_curv.astype(np.float64) * sq_local.astype(np.float64)
    return np.nan_to_num(hazard, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Pair friction maps (texture-modulated, analysis / visualization)
# ---------------------------------------------------------------------------


def slide_azimuth_rad_from_direction(direction: int) -> float:
    """
    Image-plane azimuth of the lab slide vector ``(du, dv)`` in radians.

    Uses the same convention as :func:`tmd.sequence.wear_simulation.sliding_vector`:
    ``+x`` → ``0``, ``-x`` → ``π``, ``+y`` → ``π/2``, ``-y`` → ``-π/2``.
    """
    from tmd.sequence.wear_simulation import sliding_vector

    du, dv = sliding_vector(int(direction))
    return float(np.arctan2(float(dv), float(du)))


def _local_texture_angle_rad(
    height_map: np.ndarray,
    metadata: Optional[Dict[str, Any]],
    *,
    plane_removal: str,
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Smoothed local steepest-ascent angle (rad) and coherence in ``[0, 1]``."""
    z = _preprocess_plane(height_map, metadata, plane_removal)
    dx, dy = cell_sizes_from_metadata(metadata, z.shape)
    gx, gy = calculate_surface_gradient(z, dx=dx, dy=dy, scale=1.0)
    ang = gradient_slant_angle_rad(np.asarray(gx, dtype=np.float64), np.asarray(gy, dtype=np.float64))
    cos2 = np.cos(2.0 * ang)
    sin2 = np.sin(2.0 * ang)
    local_angle, coherence = local_texture_angle_and_coherence(cos2, sin2, window)
    return local_angle.astype(np.float64, copy=False), coherence.astype(np.float64, copy=False)


def texture_modulated_pair_friction_maps(
    ms: np.ndarray,
    ss: np.ndarray,
    wear_params: Any,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    strength: float = 0.35,
    fusion: Literal["rms", "mean"] = "rms",
    window: int = 9,
    plane_removal: str = "none",
    return_aux: bool = False,
) -> Union[Dict[int, np.ndarray], Tuple[Dict[int, np.ndarray], Dict[str, Any]]]:
    """
    Four direction-dependent effective friction maps for a master–slave height pair.

    Starts from :func:`tmd.sequence.wear_simulation.combined_mu` (slope-based baseline),
    then multiplies by a **per-pixel** texture-alignment factor derived from **both**
    surfaces using locally averaged gradient directions (see
    :func:`local_texture_angle_and_coherence`).

    For slide direction ``d`` with azimuth ``φ_d`` from :func:`slide_azimuth_rad_from_direction`
    and local texture angles ``θ_MS``, ``θ_SS`` (steepest-ascent direction on **master** /
    **slave** grids, rad):

    - Per-surface alignment ``a_k = clip((1 + cos(2(θ_k − φ_d))) / 2, 0, 1)``.
    - Fused weight ``w``: ``sqrt(a_MS * a_SS)`` (``fusion="rms"``) or mean of the two.
    - Effective map ``μ_d = clip(μ_d^0 * ((1 − s) + s · w), 0, friction_clip)`` with strength ``s``.

    This is a **phenomenological** visualization / analysis aid, not a calibrated COF model
    and not wired into :func:`tmd.sequence.wear_simulation.run_simulation` by default.

    Args:
        ms: Master height map (2D), same shape as ``ss``.
        ss: Slave height map (2D).
        wear_params: :class:`tmd.sequence.wear_simulation.WearParameters` instance.
        metadata: Optional dict for ``cell_sizes_from_metadata`` / plane removal.
        strength: Modulation strength in ``[0, 1]``; ``0`` recovers baseline ``combined_mu``.
        fusion: How to fuse per-surface alignments into ``w``.
        window: Odd window size for local texture angle (see ``local_texture_angle_and_coherence``).
        plane_removal: Passed to :func:`_preprocess_plane`.
        return_aux: If True, return ``(maps, aux)`` with modulation arrays and angles in ``aux``.

    Returns:
        Mapping ``direction_code -> HxW`` array (same keys as ``DIRECTION_NAMES`` in
        ``wear_simulation``), or a tuple with an ``aux`` dict when ``return_aux`` is True.
    """
    from tmd.sequence.wear_simulation import DIRECTION_NAMES, combined_mu

    a_ms = np.asarray(ms, dtype=np.float64)
    a_ss = np.asarray(ss, dtype=np.float64)
    if a_ms.ndim != 2 or a_ss.ndim != 2 or a_ms.shape != a_ss.shape:
        raise ValueError("ms and ss must be 2D arrays with the same shape")

    s_strength = float(np.clip(strength, 0.0, 1.0))
    fu = fusion.strip().lower()
    if fu not in ("rms", "mean"):
        raise ValueError('fusion must be "rms" or "mean"')

    theta_ms, coh_ms = _local_texture_angle_rad(
        a_ms, metadata, plane_removal=plane_removal, window=window
    )
    theta_ss, coh_ss = _local_texture_angle_rad(
        a_ss, metadata, plane_removal=plane_removal, window=window
    )

    maps: Dict[int, np.ndarray] = {}
    mod_by_dir: Dict[int, np.ndarray] = {}
    for d in sorted(DIRECTION_NAMES):
        phi = slide_azimuth_rad_from_direction(int(d))
        a_m = 0.5 * (1.0 + np.cos(2.0 * (theta_ms - phi)))
        a_s = 0.5 * (1.0 + np.cos(2.0 * (theta_ss - phi)))
        if fu == "rms":
            w = np.sqrt(np.maximum(a_m * a_s, 0.0))
        else:
            w = 0.5 * (a_m + a_s)
        w = np.clip(w, 0.0, 1.0)
        base = np.asarray(combined_mu(a_ms, a_ss, wear_params, int(d)), dtype=np.float64)
        mu = base * ((1.0 - s_strength) + s_strength * w)
        clip_v = float(getattr(wear_params, "friction_clip", 0.3))
        maps[int(d)] = np.clip(mu, 0.0, clip_v).astype(np.float64, copy=False)
        mod_by_dir[int(d)] = w.astype(np.float64, copy=False)

    if not return_aux:
        return maps
    aux: Dict[str, Any] = {
        "modulation": mod_by_dir,
        "theta_ms": theta_ms,
        "theta_ss": theta_ss,
        "coherence_ms": coh_ms,
        "coherence_ss": coh_ss,
    }
    return maps, aux


def bearing_area_curve(
    height_map: np.ndarray,
    separations: Optional[np.ndarray] = None,
    *,
    n: int = 50,
    z_reference: str = "mean",
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
    plane_removal: str = "none",
) -> Dict[str, Any]:
    """
    Geometric bearing / contact fraction vs separation (leveled height).

    ``area_fraction[i]`` = mean(z_leveled >= separations[i]).
    """
    z = np.asarray(height_map, dtype=np.float64)
    z = np.nan_to_num(z)
    if z.ndim != 2:
        raise ValueError("height_map must be 2D")
    z = _preprocess_plane(z.astype(np.float32), metadata, plane_removal).astype(np.float64, copy=False)
    if z_reference == "median":
        z0 = z - float(np.median(z))
    else:
        z0 = z - float(np.mean(z))

    if separations is None:
        hi = float(np.max(z0)) if d_max is None else float(d_max)
        lo = float(np.min(z0)) if d_min is None else float(d_min)
        separations = np.linspace(lo, hi, int(max(2, n)))

    s = np.asarray(separations, dtype=np.float64).ravel()
    area_fraction = np.array([float(np.mean(z0 >= t)) for t in s], dtype=np.float64)
    if s.size >= 2:
        dA_dd = np.gradient(area_fraction, s)
    else:
        dA_dd = np.zeros_like(area_fraction)

    return {
        "separations": s,
        "area_fraction": area_fraction,
        "dA_dd": dA_dd.astype(np.float64, copy=False),
        "z_reference": z_reference,
        "plane_removal": plane_removal,
    }


def summit_curvature_map(
    height_map: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    smooth_sigma: float = 0.8,
    min_mean_curvature: float = 0.0,
    plane_removal: str = "none",
) -> Dict[str, Any]:
    """
    Local summits on a smoothed surface; ``inv_radius`` ≈ |H| at summits (1/m in map units).

    ``min_mean_curvature`` gates on **|H|**. Graph peaks have **negative** mean curvature in
    the ``mean_curvature`` convention used here (same as ``CurvatureMapGenerator`` mean mode),
    so requiring ``H >= 0`` would mark no summits; the gate uses magnitude instead.
    """
    z = np.asarray(height_map, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError("height_map must be 2D")
    z = _preprocess_plane(z, metadata, plane_removal)
    dx, dy = cell_sizes_from_metadata(metadata, z.shape)
    zs = ndimage.gaussian_filter(z, sigma=float(max(0.0, smooth_sigma))).astype(np.float32, copy=False)
    h = mean_curvature(zs, cell_size_x=dx, cell_size_y=dy, scale=1.0).astype(np.float32)
    neighborhood = np.ones((3, 3), dtype=bool)
    max_z = ndimage.maximum_filter(zs, footprint=neighborhood)
    h_abs = np.abs(h)
    summit = (zs >= max_z) & (h_abs >= float(min_mean_curvature))
    inv_r = np.abs(h)
    out = np.zeros_like(zs, dtype=np.float32)
    out[summit] = inv_r[summit]
    density = ndimage.uniform_filter(summit.astype(np.float32), size=15)
    return {
        "summit_mask": summit,
        "inv_radius": out,
        "summit_density": density,
    }


def debris_pocket_map(
    height_map: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    *,
    defect_config: Optional[DefectDetectionConfig] = None,
    valley_percentile: float = 10.0,
    slope_percentile: float = 40.0,
    pit_dilate: int = 3,
    plane_removal: str = "none",
) -> Dict[str, Any]:
    """
    Heuristic third-body pocket mask: (pit ∪ deep valley) ∩ low-slope, with pit dilation.
    """
    z = np.asarray(height_map, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError("height_map must be 2D")
    z = _preprocess_plane(z, metadata, plane_removal)
    dx, dy = cell_sizes_from_metadata(metadata, z.shape)
    z0 = z - float(np.median(z))

    cfg = defect_config or DefectDetectionConfig()
    defects = detect_surface_defects(z, cfg, output_mode="standard", include_responses=False)
    pits = np.zeros(z.shape, dtype=bool)
    pentry = defects["defects"].get("pits")
    if pentry and "mask" in pentry:
        pits = np.asarray(pentry["mask"], dtype=bool)

    struct = np.ones((max(1, pit_dilate), max(1, pit_dilate)), dtype=bool)
    pits_d = ndimage.binary_dilation(pits, structure=struct)

    v_thr = float(np.percentile(z0, valley_percentile))
    valleys = z0 <= v_thr

    gx, gy = calculate_surface_gradient(z, dx=dx, dy=dy, scale=1.0)
    slope = np.sqrt(gx * gx + gy * gy)
    s_thr = float(np.percentile(slope, slope_percentile))
    low_slope = slope <= s_thr

    pocket = (pits_d | valleys) & low_slope
    score = pits_d.astype(np.float32) * 0.45 + valleys.astype(np.float32) * 0.35
    score = score * low_slope.astype(np.float32)
    return {"pocket_mask": pocket, "pocket_score": score.astype(np.float32, copy=False)}


# ---------------------------------------------------------------------------
# Matplotlib dashboard (Agg)
# ---------------------------------------------------------------------------


def save_tribology_dashboard_png(
    height_map: np.ndarray,
    metadata: Optional[Dict[str, Any]],
    *,
    title: str,
    output_path: Path,
    plane_removal: str = "none",
    z_reference: str = "mean",
    curve_n: int = 64,
    gaussian_sigma: float = 0.0,
    dpi: float = 150.0,
    include_proxy_maps: bool = True,
    include_anomaly_angle: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hm = np.asarray(height_map, dtype=np.float32)
    meta = metadata or {}
    z = np.nan_to_num(np.asarray(hm, dtype=np.float64))
    z = _preprocess_plane(z.astype(np.float32), meta, plane_removal).astype(np.float64, copy=False)
    z0 = z - float(np.median(z)) if (z_reference or "mean").strip().lower() == "median" else z - float(np.mean(z))

    axis = preferred_slip_axis(
        hm,
        meta,
        gaussian_sigma=float(gaussian_sigma),
        plane_removal=plane_removal,
        include_anomaly_angle=include_anomaly_angle,
    )
    curve = bearing_area_curve(
        hm,
        n=int(max(2, curve_n)),
        z_reference=z_reference,
        metadata=meta,
        plane_removal=plane_removal,
    )

    nrows = 2 if include_proxy_maps else 1
    fig_h = 8.5 if include_proxy_maps else 4.2
    fig, axes = plt.subplots(nrows, 3, figsize=(12.5, fig_h), squeeze=False)
    fig.suptitle(title, fontsize=11)

    lo, hi = float(np.percentile(z0, 2.0)), float(np.percentile(z0, 98.0))
    if hi <= lo:
        lo, hi = float(z0.min()), float(z0.max())
        if hi <= lo:
            hi = lo + 1e-9
    ax0 = axes[0, 0]
    im = ax0.imshow(z0, cmap="viridis", aspect="auto", vmin=lo, vmax=hi)
    ax0.set_title("Leveled height (bearing ref)")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)

    ax1 = axes[0, 1]
    s = np.asarray(curve["separations"], dtype=np.float64)
    a = np.asarray(curve["area_fraction"], dtype=np.float64)
    d = np.asarray(curve["dA_dd"], dtype=np.float64)
    ax1.plot(s, a, color="C0", lw=1.6, label="Area fraction")
    ax1.set_xlabel("Separation")
    ax1.set_ylabel("Area fraction")
    ax1.grid(True, alpha=0.3)
    ax1b = ax1.twinx()
    ax1b.plot(s, d, color="C3", lw=1.0, alpha=0.85, label="|dA/dd| (proxy)")
    ax1b.set_ylabel("|dA/dd|")
    ax1.set_title("Bearing-style curve")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right", fontsize=8)

    ax2 = axes[0, 2]
    ax2.set_title("Preferred slip axis")
    ax2.set_aspect("equal")
    ax2.axhline(0, color="0.5", lw=0.8)
    ax2.axvline(0, color="0.5", lw=0.8)
    rad = float(axis["axis_rad"])
    L = 1.0
    ax2.arrow(0.0, 0.0, L * np.cos(rad), L * np.sin(rad), width=0.04, head_width=0.12, head_length=0.1, fc="C0", ec="C0", length_includes_head=True)
    ax2.set_xlim(-1.15, 1.15)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.text(
        0.0,
        -0.92,
        f"fused: {float(axis['axis_deg']):.1f} deg\nasymmetry: {float(axis['asymmetry']):.3f}",
        ha="center",
        va="top",
        fontsize=9,
        family="monospace",
    )

    if include_proxy_maps:
        from tmd.image.maps.curvature import SummitCurvatureMapGenerator
        from tmd.surface.metrics.tribology_maps import debris_pocket_map_01, shear_hazard_map_01

        sh = shear_hazard_map_01(hm, meta, plane_removal=plane_removal)
        dp = debris_pocket_map_01(hm, meta, plane_removal=plane_removal)
        sm = SummitCurvatureMapGenerator(plane_removal=plane_removal).generate(hm, metadata=meta)
        for ax, arr, ttl in (
            (axes[1, 0], sh, "Shear hazard (norm)"),
            (axes[1, 1], dp, "Debris pocket (norm)"),
            (axes[1, 2], sm, "Summit |H| (norm)"),
        ):
            ax.imshow(np.asarray(arr), cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
            ax.set_title(ttl)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=float(dpi), bbox_inches="tight")
    plt.close(fig)
