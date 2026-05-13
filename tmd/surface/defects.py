"""Surface defect detection utilities."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import numpy as np
from scipy import ndimage

from tmd.surface import filters as _filters
from tmd.surface.types import (
    DefectAnalysisResult,
    DefectClassResult,
    DefectDetectionConfig,
    DefectOutputMode,
    DefectSummary,
)

_DEFECT_ORDER = (
    "pits",
    "peaks",
    "scratches",
    "cracks",
    "directionality_anomalies",
)

_DEFECT_COLORS: Dict[str, Tuple[float, float, float]] = {
    "pits": (0.08, 0.75, 0.95),
    "peaks": (0.96, 0.78, 0.12),
    "scratches": (0.90, 0.38, 0.18),
    "cracks": (0.62, 0.23, 0.84),
    "directionality_anomalies": (0.22, 0.85, 0.36),
}


def _mad_std(x: np.ndarray) -> float:
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad <= 1e-12:
        std = float(np.std(x))
        return max(std, 1e-12)
    return max(1.4826 * mad, 1e-12)


def _filter_components(mask: np.ndarray, min_area: int) -> Tuple[np.ndarray, int, list[int]]:
    labeled, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool), 0, []

    label_sizes = np.bincount(labeled.ravel(), minlength=count + 1)
    keep_labels = np.flatnonzero(label_sizes >= min_area)
    keep_labels = keep_labels[keep_labels != 0]
    if keep_labels.size == 0:
        return np.zeros_like(mask, dtype=bool), 0, []

    filtered = np.isin(labeled, keep_labels)
    areas = label_sizes[keep_labels].astype(int).tolist()
    areas.sort(reverse=True)
    return filtered, int(keep_labels.size), areas


def _normalize_response(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo = float(np.percentile(x, 2))
    hi = float(np.percentile(x, 98))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    norm = (x - lo) / (hi - lo)
    return np.clip(norm, 0.0, 1.0).astype(np.float32, copy=False)


def _confidence(mask: np.ndarray, response: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    mean_response = float(np.mean(response[mask]))
    coverage = float(mask.mean())
    # Prefer strong compact detections; cap to [0, 1].
    score = 0.75 * mean_response + 0.25 * min(1.0, 12.0 * coverage)
    return float(np.clip(score, 0.0, 1.0))


def _make_entry(
    mask: np.ndarray,
    response: np.ndarray,
    config: DefectDetectionConfig,
    *,
    include_mask: bool,
    include_response: bool,
    include_areas: bool,
) -> Tuple[DefectClassResult, np.ndarray]:
    filtered, count, areas = _filter_components(mask, min_area=max(1, config.min_area))
    conf = _confidence(filtered, response)
    if conf < config.min_confidence:
        filtered = np.zeros_like(filtered, dtype=bool)
        count = 0
        areas = []
        conf = 0.0

    result: DefectClassResult = {
        "count": int(count),
        "confidence": float(conf),
    }
    if include_areas:
        result["areas"] = areas
    if include_mask:
        result["mask"] = filtered
    if include_response:
        result["response"] = response
    return result, filtered


def _compute_directionality_anomalies(
    gradient_x: np.ndarray,
    gradient_y: np.ndarray,
    magnitude_norm: np.ndarray,
    config: DefectDetectionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    angle = _filters.gradient_slant_angle_rad(gradient_x, gradient_y)
    cos2, sin2 = np.cos(2.0 * angle), np.sin(2.0 * angle)
    global_angle = _filters.global_texture_angle_rad(gradient_x, gradient_y)

    win = int(max(3, config.directionality_window))
    local_angle, local_coherence = _filters.local_texture_angle_and_coherence(
        cos2, sin2, win
    )

    diff = _filters.wrapped_angle_diff_rad(local_angle, global_angle)

    angle_threshold = np.deg2rad(config.directionality_angle_threshold_deg)
    response = _normalize_response(local_coherence * (diff / np.pi) * magnitude_norm)
    mask = (local_coherence > 0.35) & (diff > angle_threshold) & (response > np.percentile(response, 82))
    return mask, response


def detect_surface_defects(
    height_map: np.ndarray,
    config: DefectDetectionConfig | None = None,
    *,
    output_mode: DefectOutputMode | None = None,
    include_responses: bool | None = None,
) -> DefectAnalysisResult:
    """Detect pits, peaks, scratches, cracks, and directionality anomalies."""
    if height_map.ndim != 2:
        raise ValueError("Surface defect detection requires a 2D height map")

    cfg = config or DefectDetectionConfig()
    cfg = replace(cfg, min_confidence=float(np.clip(cfg.min_confidence, 0.0, 1.0)))
    mode = output_mode or cfg.output_mode
    if mode not in {"summary", "standard", "full"}:
        raise ValueError("output_mode must be one of: summary, standard, full")
    include_class_responses = (include_responses if include_responses is not None else cfg.include_responses) and mode in {
        "standard",
        "full",
    }
    include_masks = mode in {"standard", "full"}
    include_areas = mode in {"standard", "full"}
    include_combined_outputs = mode == "full"

    base = np.asarray(height_map, dtype=np.float32)
    smooth = ndimage.gaussian_filter(base, sigma=max(0.0, cfg.gaussian_sigma)).astype(np.float32, copy=False)
    residual = smooth - np.median(smooth)
    robust_sigma = _mad_std(residual)
    zscore = residual / robust_sigma

    gradient_y, gradient_x = np.gradient(smooth)
    gradient_mag = np.hypot(gradient_x, gradient_y)
    gradient_mag_norm = _normalize_response(gradient_mag)
    laplace = ndimage.laplace(smooth)
    abs_laplace = np.abs(laplace)

    pit_response = _normalize_response(np.maximum(0.0, -zscore))
    peak_response = _normalize_response(np.maximum(0.0, zscore))
    pits_mask = zscore < -cfg.zscore_threshold
    peaks_mask = zscore > cfg.zscore_threshold

    # Scratch response: oriented dark-line extraction plus gradient support.
    close_h = ndimage.grey_closing(smooth, size=(1, 11))
    close_v = ndimage.grey_closing(smooth, size=(11, 1))
    scratch_line = np.maximum(close_h - smooth, close_v - smooth)
    scratch_response = _normalize_response(0.65 * scratch_line + 0.35 * gradient_mag_norm)
    scratch_line_threshold = np.percentile(scratch_line, 90)
    scratch_response_threshold = np.percentile(scratch_response, 82)
    scratches_mask = (scratch_line > scratch_line_threshold) & (scratch_response > scratch_response_threshold)

    # Crack response: sharp high-curvature structures with thin morphology.
    crack_raw = _normalize_response(abs_laplace * gradient_mag_norm)
    abs_laplace_threshold = np.percentile(abs_laplace, 93)
    crack_raw_threshold = np.percentile(crack_raw, 86)
    crack_seed = (abs_laplace > abs_laplace_threshold) & (crack_raw > crack_raw_threshold)
    crack_seed = ndimage.binary_opening(crack_seed, structure=np.ones((2, 2)))
    cracks_mask = ndimage.binary_dilation(crack_seed, structure=np.ones((2, 2)))

    dir_mask, dir_response = _compute_directionality_anomalies(gradient_x, gradient_y, gradient_mag_norm, cfg)

    defect_inputs = {
        "pits": (pits_mask, pit_response),
        "peaks": (peaks_mask, peak_response),
        "scratches": (scratches_mask, scratch_response),
        "cracks": (cracks_mask, crack_raw),
        "directionality_anomalies": (dir_mask, dir_response),
    }
    defects: Dict[str, DefectClassResult] = {}
    filtered_masks: Dict[str, np.ndarray] = {}
    for name in _DEFECT_ORDER:
        defect_mask, response = defect_inputs[name]
        defect_entry, filtered_mask = _make_entry(
            defect_mask,
            response,
            cfg,
            include_mask=include_masks,
            include_response=include_class_responses,
            include_areas=include_areas,
        )
        defects[name] = defect_entry
        filtered_masks[name] = filtered_mask

    class_counts = {name: int(defects[name]["count"]) for name in _DEFECT_ORDER}
    total_count = int(sum(class_counts.values()))
    global_confidence = float(np.mean([defects[name]["confidence"] for name in _DEFECT_ORDER]))
    summary: DefectSummary = {
        "total_count": total_count,
        "global_confidence": global_confidence,
        "class_counts": class_counts,
    }

    result: DefectAnalysisResult = {
        "defects": defects,
        "summary": summary,
    }
    if include_combined_outputs:
        combined_mask = np.zeros_like(base, dtype=bool)
        labels = np.zeros_like(base, dtype=np.int32)
        overlay = np.zeros((base.shape[0], base.shape[1], 3), dtype=np.float32)
        for idx, name in enumerate(_DEFECT_ORDER, start=1):
            class_mask = filtered_masks[name]
            combined_mask |= class_mask
            labels[class_mask] = idx
            color = _DEFECT_COLORS[name]
            for channel in range(3):
                overlay[..., channel][class_mask] = color[channel]
        result["mask"] = combined_mask
        result["labels"] = labels
        result["overlay_rgb"] = np.clip(overlay, 0.0, 1.0).astype(np.float32, copy=False)
    return result
