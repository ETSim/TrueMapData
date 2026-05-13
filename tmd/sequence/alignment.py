"""
Sequence alignment in one module:

- **Phase FFT** — translation-only alignment of 2D height stacks (:func:`align_height_map_sequence_phase_fft`).
- **OpenCV SIFT / ECC** — BGR normal maps with tangent-space rotation (TextureFriction ``align.ipynb``),
  or the same feature pipeline on **2D height** maps with scalar warps (:func:`align_normal_map_sequence`,
  :func:`align_height_map_sequence_sift`).
- **ICP on TMD-style height clouds** — subsampled ``(x, y, z)`` points from height maps, rigid ICP in 3D,
  then a partial 2D affine fit in the image plane for warping (:func:`align_height_map_sequence_icp`,
  :func:`icp_rigid_point_to_point`, :func:`point_cloud_from_height_map`).

Design reference for the SIFT / normal pipeline: TextureFriction ``notebook/real/align.ipynb`` (PBR normal
sequence, two-pass SIFT/ECC, shared affine + crop, scalar height warp).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from scipy.linalg import svd
from scipy.spatial import cKDTree

try:
    import cv2

    _has_cv2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment, misc]
    _has_cv2 = False


# ---------------------------------------------------------------------------
# Phase correlation (height, NumPy only)
# ---------------------------------------------------------------------------


def estimate_translation_phase_fft(ref: np.ndarray, mov: np.ndarray) -> Tuple[float, float]:
    """
    Estimate integer-periodic translation (dy, dx) so that rolling ``mov`` by ``(-dy, -dx)``
    aligns it to ``ref`` (same shape, 2D).

    Uses normalized cross-power spectrum (phase correlation). Assumes dominant error is
    global shift; periodic boundaries (``np.roll``), so prefer already similarly cropped stacks.
    """
    r = np.asarray(ref, dtype=np.float64)
    m = np.asarray(mov, dtype=np.float64)
    if r.shape != m.shape or r.ndim != 2:
        raise ValueError("ref and mov must be 2D arrays with the same shape")
    r = r - np.nanmean(r)
    m = m - np.nanmean(m)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    g1 = np.fft.fft2(r)
    g2 = np.fft.fft2(m)
    cross = g2 * np.conj(g1)
    cross /= np.abs(cross) + 1e-15
    cc = np.real(np.fft.ifft2(cross))
    cc = np.fft.fftshift(cc)
    peak = np.unravel_index(int(np.argmax(cc)), cc.shape)
    h, w = r.shape
    dy = float(peak[0] - h // 2)
    dx = float(peak[1] - w // 2)
    return dy, dx


def _roll_align(mov: np.ndarray, dy: float, dx: float) -> np.ndarray:
    a = np.asarray(mov, dtype=np.float64).copy()
    iy = int(round(dy))
    ix = int(round(dx))
    a = np.roll(a, -iy, axis=0)
    a = np.roll(a, -ix, axis=1)
    return a


def align_height_map_sequence_phase_fft(
    frames: List[np.ndarray],
    reference_index: int = 0,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Align each frame to the reference using phase correlation and ``np.roll``.

    All frames must match the reference shape. For differing sizes or affine drift,
    use :func:`tmd.surface.transformations.align_height_map_sequence_opencv` or
    :func:`align_height_map_sequence_sift` / :func:`align_height_map_sequence_icp`.
    """
    if not frames:
        return [], {"reference_index": reference_index, "method": "phase_fft_numpy", "per_frame": []}
    n = len(frames)
    if not (0 <= reference_index < n):
        raise ValueError(f"reference_index must be in [0, {n - 1}], got {reference_index}")
    ref = np.asarray(frames[reference_index], dtype=np.float64)
    if ref.ndim != 2:
        raise ValueError("each frame must be 2D")
    for i, fr in enumerate(frames):
        a = np.asarray(fr)
        if a.shape != ref.shape:
            raise ValueError(
                f"frame {i} shape {a.shape} != reference shape {ref.shape}; "
                "resize or use OpenCV alignment first."
            )

    aligned: List[np.ndarray] = []
    per_frame: List[Dict[str, Any]] = []
    for i, fr in enumerate(frames):
        if i == reference_index:
            aligned.append(np.asarray(fr, dtype=np.float64).copy())
            per_frame.append(
                {
                    "frame_index": i,
                    "dy": 0.0,
                    "dx": 0.0,
                    "roll_dy": 0,
                    "roll_dx": 0,
                    "reference": True,
                }
            )
            continue
        mov = np.asarray(fr, dtype=np.float64)
        dy, dx = estimate_translation_phase_fft(ref, mov)
        warped = _roll_align(mov, dy, dx)
        aligned.append(warped)
        per_frame.append(
            {
                "frame_index": i,
                "dy": dy,
                "dx": dx,
                "roll_dy": int(round(dy)),
                "roll_dx": int(round(dx)),
                "reference": False,
            }
        )

    info: Dict[str, Any] = {
        "reference_index": reference_index,
        "method": "phase_fft_numpy",
        "per_frame": per_frame,
    }
    return aligned, info


# ---------------------------------------------------------------------------
# OpenCV: config, normal warps, masks, crop, SIFT (normals + height)
# ---------------------------------------------------------------------------


@dataclass
class NormalMapSequenceAlignmentConfig:
    """Hyperparameters aligned with TextureFriction ``align.ipynb`` ``Config``."""

    sift_features: int = 8000
    lowe_ratio: float = 0.7
    ransac_thresh: float = 5.0
    ransac_iters: int = 3000
    ecc_iterations: int = 5000
    ecc_epsilon: float = 1e-7
    erosion_iters: int = 1
    safety_margin: int = 8
    border_color_bgr: Tuple[int, int, int] = (128, 128, 255)


@dataclass
class ICPHeightAlignmentConfig:
    """Subsampling and ICP iteration limits for height-derived point clouds."""

    stride: int = 2
    max_iterations: int = 30
    tolerance: float = 1e-4
    max_points: int = 20000
    xy_scale: float = 1.0
    z_scale: float = 1.0
    ransac_reproj_threshold: float = 3.0


def height_maps_to_normal_bgr_uint8(
    frames: List[np.ndarray],
    *,
    strength: float = 1.0,
    normalize: bool = False,
) -> List[np.ndarray]:
    """
    Convert each 2D height map to a ``uint8`` **BGR** normal image for
    :func:`align_normal_map_sequence` (same encoding as OpenCV ``imread`` normals).

    Uses :class:`~tmd.image.maps.normal.NormalMapGenerator` (RGB tangent normals in
    ``[0, 1]``), then scales to ``uint8`` and swaps to BGR for OpenCV/SIFT paths.
    """
    from tmd.image.maps.normal import NormalMapGenerator

    gen = NormalMapGenerator(strength=float(strength), normalize=bool(normalize), debug=False)
    out: List[np.ndarray] = []
    for z in frames:
        zf = np.asarray(z, dtype=np.float32)
        rgb01 = gen.generate(zf, metadata={})
        rgb_u8 = np.clip(np.asarray(rgb01, dtype=np.float64) * 255.0, 0.0, 255.0).astype(np.uint8)
        out.append(rgb_u8[..., ::-1].copy())
    return out


def _require_cv2() -> None:
    if not _has_cv2 or cv2 is None:
        raise ImportError("OpenCV is required for this alignment path")


def rotate_normal_vectors_bgr(normal_img: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """Rotate XY of a BGR normal map in tangent space; Z unchanged. uint8 in/out."""
    normals = normal_img.astype(np.float32)
    normals_rgb = normals[..., [2, 1, 0]]
    normals_normalized = (normals_rgb / 127.5) - 1.0
    h, w = normals.shape[:2]
    xy = normals_normalized[..., :2].reshape(-1, 2)
    xy_rotated = xy @ rotation_matrix.T
    normals_normalized[..., :2] = xy_rotated.reshape(h, w, 2)
    norm = np.linalg.norm(normals_normalized, axis=2, keepdims=True)
    normals_normalized = normals_normalized / (norm + 1e-8)
    normals_rgb = (normals_normalized + 1.0) * 127.5
    normals_bgr = normals_rgb[..., [2, 1, 0]]
    return np.clip(normals_bgr, 0, 255).astype(np.uint8)


def warp_normal_map_bgr(
    normal_img: np.ndarray,
    warp_matrix: np.ndarray,
    size: Tuple[int, int],
    border_value: Tuple[int, int, int] = (128, 128, 255),
) -> np.ndarray:
    """Affine warp with tangent-space rotation from the upper ``2×2`` of ``warp_matrix``."""
    _require_cv2()
    assert cv2 is not None
    rotation_matrix = warp_matrix[:2, :2]
    rotated_normals = rotate_normal_vectors_bgr(normal_img, rotation_matrix)
    return cv2.warpAffine(
        rotated_normals,
        warp_matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def warp_valid_mask(
    shape: Tuple[int, int],
    warp_matrix: np.ndarray,
    size: Tuple[int, int],
) -> np.ndarray:
    """Warp a binary in-bounds mask with nearest-neighbor sampling."""
    _require_cv2()
    assert cv2 is not None
    h, w = shape
    ones = np.ones((h, w), dtype=np.uint8)
    return cv2.warpAffine(
        ones,
        warp_matrix,
        size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def warp_scalar_map(
    img: np.ndarray,
    warp_matrix: np.ndarray,
    size: Tuple[int, int],
    border_value: Union[float, int] = 0,
) -> np.ndarray:
    """Affine warp for scalar maps (height, roughness); dtype preserved where OpenCV allows."""
    _require_cv2()
    assert cv2 is not None
    return cv2.warpAffine(
        img,
        warp_matrix,
        size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def largest_all_ones_rectangle(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Return ``(x, y, w, h)`` for the largest axis-aligned rectangle of ones."""
    h, w = mask.shape
    heights = np.zeros(w, dtype=np.int32)
    best_area = 0
    best_rect: Optional[Tuple[int, int, int, int]] = None

    for y in range(h):
        row = mask[y] > 0
        heights = np.where(row, heights + 1, 0)

        stack: List[int] = []
        for x in range(w + 1):
            cur = int(heights[x]) if x < w else 0
            while stack and cur < int(heights[stack[-1]]):
                top = stack.pop()
                height = int(heights[top])
                left = stack[-1] + 1 if stack else 0
                width = x - left
                area = height * width
                if area > best_area:
                    best_area = area
                    best_rect = (left, y - height + 1, width, height)
            stack.append(x)

    return best_rect


def find_valid_crop_region(
    valid_masks: List[np.ndarray],
    config: NormalMapSequenceAlignmentConfig,
) -> Tuple[int, int, int, int]:
    """Largest all-valid rectangle where every warped frame is in-bounds; eroded and inset."""
    _require_cv2()
    assert cv2 is not None
    h, w = valid_masks[0].shape[:2]
    combined_mask = np.ones((h, w), dtype=bool)
    for mask in valid_masks:
        combined_mask &= mask > 0

    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.erode(
        combined_mask.astype(np.uint8), kernel, iterations=config.erosion_iters
    )

    rect = largest_all_ones_rectangle(combined_mask)
    if rect is not None:
        x, y, w_crop, h_crop = rect
        margin = config.safety_margin
        x += margin
        y += margin
        w_crop -= 2 * margin
        h_crop -= 2 * margin
        x = max(0, x)
        y = max(0, y)
        w_crop = min(w_crop, w - x)
        h_crop = min(h_crop, h - y)
        w_crop = max(w_crop, 1)
        h_crop = max(h_crop, 1)
        return x, y, w_crop, h_crop

    margin = 50
    return margin, margin, w - 2 * margin, h - 2 * margin


def _gray_u8_for_features(img: np.ndarray) -> np.ndarray:
    """Grayscale uint8 for SIFT / ECC from BGR normals or 2D / single-channel height."""
    _require_cv2()
    assert cv2 is not None
    if img.ndim == 2:
        return cv2.normalize(img.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.normalize(img[..., 0].astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _align_sequence_sift_once(
    images: List[np.ndarray],
    config: NormalMapSequenceAlignmentConfig,
    *,
    use_ecc_primary: bool,
    mode: Literal["normal_bgr", "height_2d"],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """One SIFT/ECC pass: cumulative ``2×3`` transforms to frame 0; warp normals or height."""
    _require_cv2()
    assert cv2 is not None
    sift = cv2.SIFT_create(nfeatures=config.sift_features)
    matcher = cv2.FlannBasedMatcher()
    n_imgs = len(images)
    h, w = images[0].shape[:2]
    aligned: List[np.ndarray] = [np.asarray(images[0]).copy()]
    cumulative_transforms: List[np.ndarray] = [np.eye(2, 3, dtype=np.float32)]
    grays = [_gray_u8_for_features(im) for im in images]

    for i in range(1, n_imgs):
        best_transform: Optional[np.ndarray] = None
        best_score: float = -1.0
        best_ref_idx = -1

        if use_ecc_primary and i == 1:
            try:
                warp = np.eye(2, 3, dtype=np.float32)
                _, cc = cv2.findTransformECC(
                    grays[i - 1],
                    grays[i],
                    warp,
                    motionType=cv2.MOTION_EUCLIDEAN,
                    criteria=(
                        cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
                        config.ecc_iterations,
                        config.ecc_epsilon,
                    ),
                )
                if cc > 0.9:
                    best_transform = warp
                    best_score = float(cc)
                    best_ref_idx = i - 1
            except Exception:
                pass

        if best_transform is None:
            for ref_idx in range(max(0, i - 3), i):
                kp1, des1 = sift.detectAndCompute(grays[i], None)
                kp2, des2 = sift.detectAndCompute(grays[ref_idx], None)
                if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
                    continue
                try:
                    matches = matcher.knnMatch(des1, des2, k=2)
                except Exception:
                    continue
                good = []
                for pair in matches:
                    if len(pair) == 2:
                        m, n_other = pair
                        if m.distance < config.lowe_ratio * n_other.distance:
                            good.append(m)
                if len(good) >= 10:
                    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
                    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
                    M, inliers = cv2.estimateAffinePartial2D(
                        pts1,
                        pts2,
                        method=cv2.RANSAC,
                        ransacReprojThreshold=config.ransac_thresh,
                        maxIters=config.ransac_iters,
                    )
                    if M is not None and np.sum(inliers) > best_score:
                        best_transform = M
                        best_score = float(np.sum(inliers))
                        best_ref_idx = ref_idx

        if best_transform is not None:
            ref_t = cumulative_transforms[best_ref_idx]
            ref_3x3 = np.vstack([ref_t, [0, 0, 1]])
            M_3x3 = np.vstack([best_transform, [0, 0, 1]])
            cumulative = ref_3x3 @ M_3x3
            cumulative_transform = cumulative[:2]
        else:
            cumulative_transform = np.eye(2, 3, dtype=np.float32)
        cumulative_transforms.append(cumulative_transform)

        if mode == "normal_bgr":
            aligned_img = warp_normal_map_bgr(
                images[i], cumulative_transform, (w, h), border_value=config.border_color_bgr
            )
        else:
            mov_f = np.asarray(images[i], dtype=np.float32)
            aligned_img = warp_scalar_map(mov_f, cumulative_transform, (w, h), border_value=0.0)
        aligned.append(aligned_img)
    return aligned, cumulative_transforms


def _compose_two_pass_transforms(
    transforms1: List[np.ndarray], transforms2: List[np.ndarray]
) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for i in range(len(transforms1)):
        t1_3x3 = np.vstack([transforms1[i], [0, 0, 1]])
        t2_3x3 = np.vstack([transforms2[i], [0, 0, 1]])
        combined = t2_3x3 @ t1_3x3
        out.append(combined[:2])
    return out


def align_normal_map_sequence_two_pass_core(
    images: List[np.ndarray],
    config: NormalMapSequenceAlignmentConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    aligned1, transforms1 = _align_sequence_sift_once(
        images, config, use_ecc_primary=True, mode="normal_bgr"
    )
    aligned2, transforms2 = _align_sequence_sift_once(
        aligned1, config, use_ecc_primary=False, mode="normal_bgr"
    )
    return aligned2, _compose_two_pass_transforms(transforms1, transforms2)


def align_height_map_sequence_sift_two_pass_core(
    images: List[np.ndarray],
    config: NormalMapSequenceAlignmentConfig,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Two-pass SIFT alignment for 2D height stacks (scalar warp, no tangent rotation)."""
    aligned1, transforms1 = _align_sequence_sift_once(
        images, config, use_ecc_primary=True, mode="height_2d"
    )
    aligned2, transforms2 = _align_sequence_sift_once(
        aligned1, config, use_ecc_primary=False, mode="height_2d"
    )
    return aligned2, _compose_two_pass_transforms(transforms1, transforms2)


def warp_scalar_sequence_with_affine_crop(
    frames: List[np.ndarray],
    transforms: List[np.ndarray],
    full_size_wh: Tuple[int, int],
    crop_xywh: Tuple[int, int, int, int],
    border_value: Union[float, int] = 0,
) -> List[np.ndarray]:
    """
    Apply the same per-frame ``2×3`` affines as a normal/height sequence, then crop.

    ``len(frames)`` may be shorter than ``len(transforms)``; only ``min`` pairs are used.
    """
    w_full, h_full = full_size_wh
    x, y, cw, ch = crop_xywh
    n = min(len(frames), len(transforms))
    out: List[np.ndarray] = []
    for i in range(n):
        warped = warp_scalar_map(frames[i], transforms[i], (w_full, h_full), border_value=border_value)
        out.append(warped[y : y + ch, x : x + cw])
    return out


def align_normal_map_sequence(
    frames: List[np.ndarray],
    *,
    two_pass: bool = True,
    crop: bool = True,
    config: Optional[NormalMapSequenceAlignmentConfig] = None,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Align BGR normal maps (``H×W×3`` ``uint8``) to a common frame (frame ``0``), optionally crop.

    Mirrors TextureFriction ``align.ipynb``: SIFT + partial affine, ECC option, tangent-space
    rotation on warp, two-pass refinement, valid-mask crop.
    """
    _require_cv2()
    if not frames:
        return [], {
            "method": "opencv_sift_normal_bgr",
            "two_pass": two_pass,
            "crop": crop,
            "crop_region": None,
            "per_frame": [],
        }
    cfg = config or NormalMapSequenceAlignmentConfig()
    ref = np.asarray(frames[0])
    if ref.ndim != 3 or ref.shape[2] < 3:
        raise ValueError("each normal frame must be HxWx3 BGR")
    h, w = ref.shape[:2]
    for i, fr in enumerate(frames):
        a = np.asarray(fr)
        if a.shape != (h, w, 3):
            raise ValueError(f"frame {i} shape {a.shape} != {(h, w, 3)}")

    images = [np.asarray(f, dtype=np.uint8).copy() for f in frames]

    if two_pass:
        aligned_full, transforms = align_normal_map_sequence_two_pass_core(images, cfg)
        passes = 2
        method = "opencv_sift_normal_bgr_two_pass"
    else:
        aligned_full, transforms = _align_sequence_sift_once(
            images, cfg, use_ecc_primary=True, mode="normal_bgr"
        )
        passes = 1
        method = "opencv_sift_normal_bgr"

    valid_masks = [warp_valid_mask((h, w), t, (w, h)) for t in transforms]
    crop_region: Optional[Dict[str, int]] = None
    if crop:
        x0, y0, cw, ch = find_valid_crop_region(valid_masks, cfg)
        crop_region = {"x": int(x0), "y": int(y0), "width": int(cw), "height": int(ch)}
        aligned = [img[y0 : y0 + ch, x0 : x0 + cw] for img in aligned_full]
    else:
        aligned = aligned_full

    per_frame: List[Dict[str, Any]] = []
    for i, t in enumerate(transforms):
        per_frame.append(
            {
                "frame_index": i,
                "affine_2x3": t.astype(float).tolist(),
                "reference": i == 0,
            }
        )

    info: Dict[str, Any] = {
        "reference_index": 0,
        "method": method,
        "alignment_passes": passes,
        "two_pass": two_pass,
        "crop": crop,
        "crop_region": crop_region,
        "full_size": {"width": int(w), "height": int(h)},
        "per_frame": per_frame,
    }
    return aligned, info


def align_height_map_sequence_sift(
    frames: List[np.ndarray],
    *,
    two_pass: bool = True,
    crop: bool = True,
    config: Optional[NormalMapSequenceAlignmentConfig] = None,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Align 2D height maps with the same SIFT/ECC + cumulative affine pipeline as normals,
    but **scalar** ``warpAffine`` (no tangent-space normal rotation).

    Input frames must share shape ``H×W``; values are warped as float32 internally and cast
    back to the reference frame's dtype when integer.
    """
    _require_cv2()
    if not frames:
        return [], {
            "method": "opencv_sift_height_2d",
            "two_pass": two_pass,
            "crop": crop,
            "crop_region": None,
            "per_frame": [],
        }
    cfg = config or NormalMapSequenceAlignmentConfig()
    ref0 = np.asarray(frames[0])
    if ref0.ndim != 2:
        raise ValueError("each height frame must be 2D")
    h, w = ref0.shape
    dtypes = [np.asarray(f).dtype for f in frames]
    for i, fr in enumerate(frames):
        a = np.asarray(fr)
        if a.ndim != 2 or a.shape != (h, w):
            raise ValueError(f"frame {i} must have shape {(h, w)}, got {a.shape}")

    images_f = [np.asarray(f, dtype=np.float32) for f in frames]

    if two_pass:
        aligned_full, transforms = align_height_map_sequence_sift_two_pass_core(images_f, cfg)
        passes = 2
        method = "opencv_sift_height_2d_two_pass"
    else:
        aligned_full, transforms = _align_sequence_sift_once(
            images_f, cfg, use_ecc_primary=True, mode="height_2d"
        )
        passes = 1
        method = "opencv_sift_height_2d"

    valid_masks = [warp_valid_mask((h, w), t, (w, h)) for t in transforms]
    crop_region: Optional[Dict[str, int]] = None
    if crop:
        x0, y0, cw, ch = find_valid_crop_region(valid_masks, cfg)
        crop_region = {"x": int(x0), "y": int(y0), "width": int(cw), "height": int(ch)}
        aligned_full = [img[y0 : y0 + ch, x0 : x0 + cw] for img in aligned_full]
    else:
        x0, y0, cw, ch = 0, 0, w, h

    aligned: List[np.ndarray] = []
    for i, img in enumerate(aligned_full):
        dt = dtypes[i]
        if np.issubdtype(dt, np.integer):
            aligned.append(np.clip(np.round(img), np.iinfo(dt).min, np.iinfo(dt).max).astype(dt))
        else:
            aligned.append(img.astype(dt, copy=False))

    per_frame: List[Dict[str, Any]] = []
    for i, t in enumerate(transforms):
        per_frame.append(
            {
                "frame_index": i,
                "affine_2x3": t.astype(float).tolist(),
                "reference": i == 0,
            }
        )

    info: Dict[str, Any] = {
        "reference_index": 0,
        "method": method,
        "alignment_passes": passes,
        "two_pass": two_pass,
        "crop": crop,
        "crop_region": crop_region,
        "full_size": {"width": int(w), "height": int(h)},
        "per_frame": per_frame,
    }
    return aligned, info


# ---------------------------------------------------------------------------
# ICP on height-derived (x, y, z) clouds → partial 2D affine warp
# ---------------------------------------------------------------------------


def point_cloud_from_height_map(
    z: np.ndarray,
    *,
    stride: int = 2,
    xy_scale: float = 1.0,
    z_scale: float = 1.0,
    valid_mask: Optional[np.ndarray] = None,
    max_points: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Build a TMD-style point cloud ``(N, 3)`` with columns ``(x, y, z)`` from a 2D height map.

    Here ``x`` = column index × ``xy_scale``, ``y`` = row index × ``xy_scale``, ``z`` = height × ``z_scale``.
    """
    z = np.asarray(z)
    if z.ndim != 2:
        raise ValueError("z must be a 2D height map")
    h, w = z.shape
    rows = np.arange(0, h, stride, dtype=np.int64)
    cols = np.arange(0, w, stride, dtype=np.int64)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    zz = z[yy, xx]
    ok = np.isfinite(zz)
    if valid_mask is not None:
        ok &= valid_mask[yy, xx].astype(bool)
    pts = np.column_stack(
        (
            xx.ravel()[ok.ravel()].astype(np.float64) * xy_scale,
            yy.ravel()[ok.ravel()].astype(np.float64) * xy_scale,
            zz.ravel()[ok.ravel()].astype(np.float64) * z_scale,
        )
    )
    if pts.shape[0] == 0:
        return pts
    if max_points is not None and pts.shape[0] > max_points:
        g = rng or np.random.default_rng(0)
        idx = g.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
    return pts


def _kabsch_rigid_columns(p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``p``, ``q`` are ``3×N``. Returns ``R`` ``(3,3)``, ``t`` ``(3,1)`` with ``q ≈ R @ p + t``."""
    mu_p = p.mean(axis=1, keepdims=True)
    mu_q = q.mean(axis=1, keepdims=True)
    pc = p - mu_p
    qc = q - mu_q
    h_mat = pc @ qc.T
    u, _, vt = svd(h_mat)
    r_mat = vt.T @ u.T
    if np.linalg.det(r_mat) < 0:
        vt[-1, :] *= -1
        r_mat = vt.T @ u.T
    t_vec = mu_q - r_mat @ mu_p
    return r_mat.astype(np.float64), t_vec.astype(np.float64)


def icp_rigid_point_to_point(
    source: np.ndarray,
    target: np.ndarray,
    *,
    max_iterations: int = 30,
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """
    Iterative closest point with rigid transforms (maps **source** toward **target**).

    ``source`` and ``target`` are ``(N, 3)`` and ``(M, 3)`` row-wise points. Returns ``R`` ``(3, 3)``,
    ``t`` ``(3,)`` such that ``target ≈ (R @ source.T + t.reshape(3, 1)).T`` in the least-squares ICP sense.
    """
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("source must be (N, 3)")
    if tgt.ndim != 2 or tgt.shape[1] != 3:
        raise ValueError("target must be (M, 3)")
    if len(src) < 3 or len(tgt) < 3:
        raise ValueError("ICP needs at least 3 points in source and target")

    p = src.T
    tree = cKDTree(tgt)
    r_acc = np.eye(3, dtype=np.float64)
    t_acc = np.zeros((3, 1), dtype=np.float64)
    prev_err = np.inf
    last_err = 0.0
    it_done = 0
    for it in range(max_iterations):
        y = r_acc @ p + t_acc
        _, idx = tree.query(y.T)
        q = tgt[idx].T
        r_step, t_step = _kabsch_rigid_columns(y, q)
        r_acc = r_step @ r_acc
        t_acc = r_step @ t_acc + t_step
        last_err = float(np.linalg.norm((q - y).T, axis=1).mean())
        if abs(prev_err - last_err) < tolerance:
            it_done = it + 1
            break
        prev_err = last_err
        it_done = it + 1
    else:
        it_done = max_iterations

    return {
        "R": r_acc,
        "t": t_acc.reshape(3),
        "mean_error": last_err,
        "iterations": it_done,
    }


def _icp_affine_warp_from_height_pair(
    ref_z: np.ndarray,
    mov_z: np.ndarray,
    icp_cfg: ICPHeightAlignmentConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Run ICP between clouds from ``mov_z`` and ``ref_z``, then fit a partial 2D affine
    (image ``x``, ``y``) from last correspondences; return OpenCV ``2×3`` that maps **reference**
    pixel coordinates to **moving** image coordinates for ``warpAffine`` sampling.
    """
    _require_cv2()
    assert cv2 is not None
    pts_ref = point_cloud_from_height_map(
        ref_z,
        stride=icp_cfg.stride,
        xy_scale=icp_cfg.xy_scale,
        z_scale=icp_cfg.z_scale,
        max_points=icp_cfg.max_points,
    )
    pts_mov = point_cloud_from_height_map(
        mov_z,
        stride=icp_cfg.stride,
        xy_scale=icp_cfg.xy_scale,
        z_scale=icp_cfg.z_scale,
        max_points=icp_cfg.max_points,
    )
    if pts_ref.shape[0] < 8 or pts_mov.shape[0] < 8:
        return np.eye(2, 3, dtype=np.float32), {
            "icp_skipped": True,
            "reason": "too_few_points",
            "mean_error": None,
            "iterations": 0,
        }

    icp = icp_rigid_point_to_point(
        pts_mov, pts_ref, max_iterations=icp_cfg.max_iterations, tolerance=icp_cfg.tolerance
    )
    r_acc = icp["R"]
    t_col = np.asarray(icp["t"], dtype=np.float64).reshape(3, 1)
    moved = (r_acc @ pts_mov.T + t_col).T
    tree = cKDTree(pts_ref)
    _, idx = tree.query(moved)
    mov_xy = pts_mov[:, :2].astype(np.float32)
    ref_xy = pts_ref[idx, :2].astype(np.float32)
    # ref ≈ A @ mov (OpenCV src=mov, dst=ref)
    a_fwd, _ = cv2.estimateAffinePartial2D(
        mov_xy.reshape(-1, 1, 2),
        ref_xy.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=icp_cfg.ransac_reproj_threshold,
    )
    if a_fwd is None:
        return np.eye(2, 3, dtype=np.float32), {**icp, "affine_skipped": True}
    # warpAffine: dst(ref pixel) = mov(M @ [x,y,1]) → M = inv(A_fwd)
    m_warp = cv2.invertAffineTransform(a_fwd.astype(np.float32))
    meta = {**icp, "affine_skipped": False}
    return m_warp.astype(np.float32), meta


def align_height_map_sequence_icp(
    frames: List[np.ndarray],
    reference_index: int = 0,
    *,
    icp_config: Optional[ICPHeightAlignmentConfig] = None,
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Align each 2D height map to the reference by ICP on subsampled ``(x, y, z)`` clouds, then a
    partial 2D affine warp in the image plane fitted from ICP correspondences.

    Each non-reference frame is registered independently to the reference. Requires OpenCV.
    """
    _require_cv2()
    cfg = icp_config or ICPHeightAlignmentConfig()
    if not frames:
        return [], {"reference_index": reference_index, "method": "icp_height_cloud", "per_frame": []}
    n = len(frames)
    if not (0 <= reference_index < n):
        raise ValueError(f"reference_index must be in [0, {n - 1}], got {reference_index}")
    ref = np.asarray(frames[reference_index], dtype=np.float32)
    if ref.ndim != 2:
        raise ValueError("each frame must be 2D")
    h, w = ref.shape
    for i, fr in enumerate(frames):
        a = np.asarray(fr)
        if a.ndim != 2 or a.shape != (h, w):
            raise ValueError(f"frame {i} must have shape {(h, w)}, got {getattr(a, 'shape', None)}")

    dtypes = [np.asarray(f).dtype for f in frames]
    aligned: List[np.ndarray] = []
    per_frame: List[Dict[str, Any]] = []

    for i, fr in enumerate(frames):
        if i == reference_index:
            aligned.append(np.asarray(fr, dtype=np.float32).copy())
            per_frame.append(
                {
                    "frame_index": i,
                    "affine_2x3": np.eye(2, 3).astype(float).tolist(),
                    "reference": True,
                    "icp": None,
                }
            )
            continue
        mov = np.asarray(fr, dtype=np.float32)
        m_warp, icp_meta = _icp_affine_warp_from_height_pair(ref, mov, cfg)
        warped = warp_scalar_map(mov, m_warp, (w, h), border_value=0.0)
        dt = dtypes[i]
        if np.issubdtype(dt, np.integer):
            warped = np.clip(np.round(warped), np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)
        else:
            warped = warped.astype(dt, copy=False)
        aligned.append(warped)
        per_frame.append(
            {
                "frame_index": i,
                "affine_2x3": m_warp.astype(float).tolist(),
                "reference": False,
                "icp": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in icp_meta.items()},
            }
        )

    info: Dict[str, Any] = {
        "reference_index": reference_index,
        "method": "icp_height_cloud",
        "icp_config": {
            "stride": cfg.stride,
            "max_iterations": cfg.max_iterations,
            "tolerance": cfg.tolerance,
            "max_points": cfg.max_points,
            "xy_scale": cfg.xy_scale,
            "z_scale": cfg.z_scale,
        },
        "crop_region": None,
        "per_frame": per_frame,
    }
    return aligned, info
