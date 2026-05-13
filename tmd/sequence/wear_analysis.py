"""Sequence-oriented wear analysis: scratch morphology, roughness trajectory columns, volume series."""

from __future__ import annotations

import math
from typing import Any, Dict, List, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage


# --- Scratch / slip morphology -------------------------------------------------


def _skeletonize_bool(mask: np.ndarray) -> np.ndarray:
    """Binary skeleton; prefers scikit-image, falls back to a distance-ridge proxy."""
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return np.zeros_like(m, dtype=bool)
    try:
        from skimage.morphology import skeletonize  # type: ignore[import-untyped]

        return skeletonize(m)
    except ImportError:
        dt = ndimage.distance_transform_edt(m)
        inner = dt > 1
        sk = m & ~ndimage.binary_dilation(~inner, iterations=1)
        return sk & m


def skeleton_pixel_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(_skeletonize_bool(np.asarray(mask, dtype=bool))))


def scratch_evolution_pair(mask_before: np.ndarray, mask_after: np.ndarray) -> Dict[str, Any]:
    """Compare two binary scratch (or groove) masks on the same grid."""
    a = np.asarray(mask_before, dtype=bool)
    b = np.asarray(mask_after, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("masks must have the same shape")
    growth = b & ~a
    shrink = a & ~b
    stable = a & b
    sk_before = _skeletonize_bool(a)
    sk_after = _skeletonize_bool(b)
    return {
        "area_before": int(a.sum()),
        "area_after": int(b.sum()),
        "growth_pixels": int(growth.sum()),
        "shrink_pixels": int(shrink.sum()),
        "stable_pixels": int(stable.sum()),
        "skeleton_pixels_before": int(sk_before.sum()),
        "skeleton_pixels_after": int(sk_after.sum()),
        "skeleton_branch_proxy": int(_branch_proxy(sk_after)),
    }


def _branch_proxy(skel: np.ndarray) -> int:
    s = np.asarray(skel, dtype=bool)
    if not np.any(s):
        return 0
    neigh = ndimage.convolve(s.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
    neigh = neigh - s.astype(np.uint8)
    junction = s & (neigh >= 3)
    return int(junction.sum())


def scratch_series_metrics(masks: List[np.ndarray]) -> List[Dict[str, Any]]:
    if not masks:
        return []
    out: List[Dict[str, Any]] = []
    for i, m in enumerate(masks):
        if i == 0:
            out.append({"frame_index": i, "note": "reference"})
            continue
        ev = scratch_evolution_pair(masks[i - 1], m)
        ev["frame_index"] = i
        out.append(ev)
    return out


def _gradient_structure_angle_deg(gx: np.ndarray, gy: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    if mask is not None:
        w = np.asarray(mask, dtype=bool)
        if not np.any(w):
            return float("nan")
        gx = gx[w]
        gy = gy[w]
    else:
        gx = gx.ravel()
        gy = gy.ravel()
    jxx = float(np.mean(gx * gx))
    jxy = float(np.mean(gx * gy))
    jyy = float(np.mean(gy * gy))
    theta = 0.5 * np.arctan2(2.0 * jxy, (jxx - jyy))
    return float(np.degrees(theta))


def _psd_dominant_angle_deg(z: np.ndarray, n_bins: int = 36) -> Tuple[float, float]:
    z0 = np.asarray(z, dtype=np.float64)
    z0 = z0 - np.nanmean(z0)
    z0 = np.where(np.isfinite(z0), z0, 0.0)
    h, w = z0.shape
    spec = np.fft.fftshift(np.fft.fft2(z0))
    mag2 = (spec.real**2 + spec.imag**2).astype(np.float64)
    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w), dtype=np.float64)
    yy = yy - cy
    xx = xx - cx
    ang = np.mod(np.degrees(np.arctan2(yy, xx)), 360.0)
    rad = np.hypot(xx, yy)
    inner = max(2, min(h, w) // 8)
    outer = max(inner + 1, min(h, w) // 2 - 1)
    ring = (rad >= inner) & (rad <= outer)
    if not np.any(ring):
        return float("nan"), float("nan")

    edges = np.linspace(0.0, 360.0, n_bins + 1)
    energy = np.zeros(n_bins, dtype=np.float64)
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        if k == n_bins - 1:
            sel = ring & (ang >= lo) & (ang <= hi)
        else:
            sel = ring & (ang >= lo) & (ang < hi)
        energy[k] = float(np.sum(mag2[sel]))
    total = float(np.sum(energy))
    if total <= 0:
        return float("nan"), float("nan")
    e = energy / total
    kmax = int(np.argmax(e))
    dom = 0.5 * (edges[kmax] + edges[kmax + 1])
    emax = float(np.max(e))
    emin = float(np.min(e[e > 0]) if np.any(e > 0) else 0.0)
    asym = (emax - emin) / (emax + emin + 1e-12)
    return dom, float(asym)


def slip_axis_metrics(
    height_map: np.ndarray,
    *,
    direction_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fuse gradient structure-tensor angle, optional directionality mask, and PSD wedge."""
    z = np.asarray(height_map, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("height_map must be 2D")
    smooth = ndimage.gaussian_filter(z, sigma=1.0)
    gy, gx = np.gradient(smooth)
    ang_grad = _gradient_structure_angle_deg(gx, gy, mask=None)
    ang_grad_masked = (
        _gradient_structure_angle_deg(gx, gy, mask=direction_mask) if direction_mask is not None else float("nan")
    )
    ang_psd, asym = _psd_dominant_angle_deg(smooth)
    return {
        "gradient_structure_angle_deg": ang_grad,
        "gradient_structure_angle_masked_deg": ang_grad_masked,
        "psd_dominant_angle_deg": ang_psd,
        "psd_wedge_asymmetry": asym,
    }


# --- Roughness trajectory helpers --------------------------------------------


def append_trajectory_derivatives(row: MutableMapping[str, Any]) -> Dict[str, Any]:
    """Add ``Sp_Sv_ratio``, ``valley_share``, ``peak_share`` to one roughness row."""
    out: Dict[str, Any] = dict(row)
    if "__error__" in out:
        return out

    def _f(key: str) -> float:
        v = out.get(key)
        if v is None or isinstance(v, str):
            return float("nan")
        try:
            x = float(v)
        except (TypeError, ValueError):
            return float("nan")
        if math.isnan(x) or math.isinf(x):
            return float("nan")
        return x

    sp = _f("Sp")
    sv = _f("Sv")
    if math.isnan(sp) or math.isnan(sv):
        out["Sp_Sv_ratio"] = None
        out["valley_share"] = None
        out["peak_share"] = None
        return out

    denom = sp + sv
    if abs(sv) < 1e-30:
        out["Sp_Sv_ratio"] = None
    else:
        out["Sp_Sv_ratio"] = sp / sv

    if denom > 1e-30 and sp >= 0 and sv >= 0:
        out["valley_share"] = sv / denom
        out["peak_share"] = sp / denom
    else:
        out["valley_share"] = None
        out["peak_share"] = None

    return out


def append_trajectory_derivatives_batch(rows: Sequence[MutableMapping[str, Any]]) -> List[Dict[str, Any]]:
    return [append_trajectory_derivatives(dict(r)) for r in rows]


def ssk_trend_heuristic(rows: Sequence[MutableMapping[str, Any]]) -> str:
    vals: List[float] = []
    for r in rows:
        if "__error__" in r:
            continue
        v = r.get("Ssk")
        try:
            f = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if not math.isnan(f):
            vals.append(f)
    if len(vals) < 2:
        return "Not enough finite Ssk values across frames for a trend."
    delta = vals[-1] - vals[0]
    if delta < -0.05:
        return "Ssk trend: more negative over frames (often more valley-dominated asymmetry)."
    if delta > 0.05:
        return "Ssk trend: more positive over frames (often more peak-dominated asymmetry)."
    return "Ssk trend: roughly flat across frames."


# --- Volume series ------------------------------------------------------------


def _validate_stack(frames: Sequence[np.ndarray], reference_index: int) -> None:
    if not frames:
        raise ValueError("frames must be non-empty")
    if not (0 <= reference_index < len(frames)):
        raise ValueError("reference_index out of range")
    ref = np.asarray(frames[reference_index])
    if ref.ndim != 2:
        raise ValueError("each frame must be 2D")
    for i, fr in enumerate(frames):
        a = np.asarray(fr)
        if a.shape != ref.shape:
            raise ValueError(f"frame {i} shape {a.shape} != reference shape {ref.shape}")


def positive_loss_volume(
    z_ref: np.ndarray,
    z_i: np.ndarray,
    dx: float,
    dy: float,
    *,
    valid_mask: Optional[np.ndarray] = None,
) -> Tuple[float, np.ndarray]:
    ref = np.asarray(z_ref, dtype=np.float64)
    cur = np.asarray(z_i, dtype=np.float64)
    if ref.shape != cur.shape:
        raise ValueError("z_ref and z_i must have the same shape")
    loss = ref - cur
    finite = np.isfinite(loss) & np.isfinite(ref) & np.isfinite(cur)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    pos = np.clip(loss, 0.0, None)
    pos = np.where(finite, pos, 0.0)
    volume = float(np.sum(pos) * float(dx) * float(dy))
    out_loss = np.where(finite, loss, np.nan)
    return volume, out_loss


def localization_index_top_fraction(loss: np.ndarray, *, top_fraction: float = 0.10) -> float:
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1]")
    vals = loss[np.isfinite(loss) & (loss > 0)].ravel()
    if vals.size == 0:
        return float("nan")
    total = float(np.sum(vals))
    if total <= 0:
        return float("nan")
    k = max(1, int(np.ceil(top_fraction * vals.size)))
    idx = np.argpartition(-vals, k - 1)[:k]
    return float(np.sum(vals[idx]) / total)


def wear_series_vs_reference(
    frames: Sequence[np.ndarray],
    *,
    reference_index: int = 0,
    dx: float,
    dy: float,
    top_fraction: float = 0.10,
    signed: bool = False,
) -> List[Dict[str, Any]]:
    _validate_stack(frames, reference_index)
    z_ref = np.asarray(frames[reference_index], dtype=np.float64)
    rows: List[Dict[str, Any]] = []
    for i, fr in enumerate(frames):
        z_i = np.asarray(fr, dtype=np.float64)
        vol_pos, loss = positive_loss_volume(z_ref, z_i, dx, dy)
        loc = localization_index_top_fraction(loss, top_fraction=top_fraction)
        row: Dict[str, Any] = {
            "frame_index": i,
            "volume_positive_loss": vol_pos,
            "localization_index": loc,
        }
        if signed:
            finite = np.isfinite(loss)
            row["signed_volume"] = float(np.nansum(np.where(finite, loss, 0.0)) * dx * dy)
        rows.append(row)
    return rows


def wear_incremental_series(
    frames: Sequence[np.ndarray],
    *,
    dx: float,
    dy: float,
    top_fraction: float = 0.10,
) -> List[Dict[str, Any]]:
    if len(frames) < 1:
        raise ValueError("frames must be non-empty")
    _validate_stack(frames, 0)
    rows: List[Dict[str, Any]] = []
    cum = 0.0
    for i, fr in enumerate(frames):
        if i == 0:
            rows.append(
                {
                    "frame_index": i,
                    "volume_positive_incremental": 0.0,
                    "localization_index_incremental": float("nan"),
                    "cumulative_incremental_volume": 0.0,
                }
            )
            continue
        z_prev = np.asarray(frames[i - 1], dtype=np.float64)
        z_i = np.asarray(fr, dtype=np.float64)
        vol_pos, loss = positive_loss_volume(z_prev, z_i, dx, dy)
        loc = localization_index_top_fraction(loss, top_fraction=top_fraction)
        cum += vol_pos
        rows.append(
            {
                "frame_index": i,
                "volume_positive_incremental": vol_pos,
                "localization_index_incremental": loc,
                "cumulative_incremental_volume": cum,
            }
        )
    return rows
