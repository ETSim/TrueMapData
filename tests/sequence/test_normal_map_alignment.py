"""Tests for unified sequence alignment (normals, height SIFT, ICP)."""

import numpy as np
import pytest

from tmd.sequence.alignment import (
    ICPHeightAlignmentConfig,
    NormalMapSequenceAlignmentConfig,
    _has_cv2,
    align_height_map_sequence_icp,
    align_height_map_sequence_sift,
    align_normal_map_sequence,
    icp_rigid_point_to_point,
    point_cloud_from_height_map,
    warp_scalar_sequence_with_affine_crop,
)


def _flat_normal_bgr(h: int, w: int) -> np.ndarray:
    """BGR flat normal (128, 128, 255) with light noise for SIFT."""
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:, :] = (128, 128, 255)
    rng = np.random.default_rng(0)
    bgr = bgr.astype(np.int16)
    bgr[..., 0] += rng.integers(-4, 5, size=(h, w), dtype=np.int16)
    bgr[..., 1] += rng.integers(-4, 5, size=(h, w), dtype=np.int16)
    bgr[..., 2] += rng.integers(-2, 3, size=(h, w), dtype=np.int16)
    return np.clip(bgr, 0, 255).astype(np.uint8)


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_normal_map_sequence_identical_frames():
    h, w = 96, 112
    a = _flat_normal_bgr(h, w)
    cfg = NormalMapSequenceAlignmentConfig(sift_features=400)
    aligned, info = align_normal_map_sequence([a, a], two_pass=False, crop=True, config=cfg)
    assert len(aligned) == 2
    assert info["reference_index"] == 0
    assert info["crop_region"] is not None
    cr = info["crop_region"]
    assert aligned[0].shape == (cr["height"], cr["width"], 3)
    assert aligned[0].shape == aligned[1].shape
    np.testing.assert_allclose(
        aligned[0].astype(np.float32), aligned[1].astype(np.float32), atol=3.0
    )


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_normal_map_sequence_empty():
    aligned, info = align_normal_map_sequence([])
    assert aligned == []
    assert info["per_frame"] == []


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_warp_scalar_sequence_with_affine_crop():
    h, w = 32, 40
    gray = np.arange(h * w, dtype=np.uint16).reshape(h, w)
    eye = np.eye(2, 3, dtype=np.float32)
    crop = (2, 3, 10, 12)
    out = warp_scalar_sequence_with_affine_crop(
        [gray, gray], [eye, eye], (w, h), crop, border_value=0
    )
    x, y, cw, ch = crop
    assert out[0].shape == (ch, cw)
    np.testing.assert_array_equal(out[0], gray[y : y + ch, x : x + cw])


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_normal_map_importable_from_alignment_module():
    from tmd.sequence.alignment import align_normal_map_sequence as from_alignment

    h, w = 64, 72
    a = _flat_normal_bgr(h, w)
    aligned, _ = from_alignment([a, a], two_pass=False, crop=False)
    assert len(aligned) == 2
    assert aligned[0].shape == (h, w, 3)


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_height_map_sequence_sift_identical():
    rng = np.random.default_rng(1)
    z = rng.random((48, 56)).astype(np.float32)
    cfg = NormalMapSequenceAlignmentConfig(sift_features=300)
    aligned, info = align_height_map_sequence_sift([z, z], two_pass=False, crop=False, config=cfg)
    assert info["method"] == "opencv_sift_height_2d"
    np.testing.assert_allclose(aligned[0], aligned[1], rtol=0.02, atol=0.05)


def test_point_cloud_from_height_map_shape():
    z = np.arange(12, dtype=np.float32).reshape(3, 4)
    pts = point_cloud_from_height_map(z, stride=1, max_points=1000)
    assert pts.shape[1] == 3
    assert pts.shape[0] == 12


def test_icp_rigid_pure_translation():
    rng = np.random.default_rng(42)
    src = rng.random((24, 3)).astype(np.float64)
    delta = np.array([0.5, -0.25, 0.1])
    tgt = src + delta
    out = icp_rigid_point_to_point(src, tgt, max_iterations=50, tolerance=1e-8)
    assert out["mean_error"] < 1e-6
    np.testing.assert_allclose(out["t"], delta, rtol=0, atol=1e-5)
    assert out["R"].shape == (3, 3)


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_height_map_sequence_icp_identical():
    rng = np.random.default_rng(2)
    z = rng.random((24, 28)).astype(np.float32)
    icfg = ICPHeightAlignmentConfig(stride=1, max_points=300, max_iterations=15)
    aligned, info = align_height_map_sequence_icp([z, z], reference_index=0, icp_config=icfg)
    assert info["method"] == "icp_height_cloud"
    np.testing.assert_allclose(aligned[0], aligned[1], rtol=0.05, atol=0.05)
