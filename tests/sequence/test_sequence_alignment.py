"""Tests for OpenCV height-map sequence alignment."""

import cv2
import numpy as np
import pytest

from tmd.core.sequence import TMDSequence
from tmd.surface.transformations import (
    _has_cv2,
    align_height_map_sequence_opencv,
    register_heightmaps_affine_orb_ransac,
)


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_sequence_phase_correlation_recovers_roll():
    rng = np.random.default_rng(42)
    ref = rng.random((48, 64)).astype(np.float32)
    ref[20:28, 30:38] += 2.0
    shifted = np.roll(np.roll(ref, 5, axis=0), -7, axis=1)
    frames = [ref, shifted, np.roll(np.roll(ref, 2, axis=0), 3, axis=1)]
    aligned, info = align_height_map_sequence_opencv(
        frames, reference_index=0, method="phase_correlation", crop=False
    )
    assert info["reference_index"] == 0
    assert len(aligned) == 3
    c0 = float(np.corrcoef(ref.ravel(), aligned[1].ravel())[0, 1])
    assert c0 > 0.82, f"expected high correlation after align, got {c0}"


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_sequence_crop_reduces_size():
    rng = np.random.default_rng(1)
    ref = rng.random((40, 40)).astype(np.float32)
    ref[15:25, 15:25] += 1.5
    shifted = np.roll(np.roll(ref, 12, axis=0), 12, axis=1)
    frames = [ref, shifted]
    aligned, info = align_height_map_sequence_opencv(
        frames, reference_index=0, method="phase_correlation", crop=True, margin=2
    )
    assert info["crop_slices"] is not None
    assert aligned[0].shape[0] < ref.shape[0]
    assert aligned[0].shape[1] < ref.shape[1]


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_tmd_sequence_align_stores_metadata():
    seq = TMDSequence(name="test")
    rng = np.random.default_rng(0)
    base = rng.random((32, 32)).astype(np.float32)
    base[10:14, 10:14] += 1.0
    seq.add_frame(base, timestamp="a")
    seq.add_frame(np.roll(np.roll(base, 4, axis=0), -3, axis=1), timestamp="b")
    info = seq.align_height_maps_opencv(reference_index=0, method="phase_correlation", crop=False)
    assert "alignment" in seq.metadata
    assert seq.metadata["alignment"] == info
    assert len(seq.frames) == 2


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_sequence_second_full_pass_flag():
    rng = np.random.default_rng(101)
    ref = rng.random((40, 48)).astype(np.float32)
    shifted = np.roll(np.roll(ref, 2, axis=0), -3, axis=1)
    frames = [ref, shifted]
    aligned, info = align_height_map_sequence_opencv(
        frames,
        reference_index=0,
        method="phase_correlation",
        crop=False,
        phase_refine=False,
        second_full_pass=True,
    )
    assert info.get("second_full_pass") is True
    assert info.get("second_full_pass_detail") is not None
    assert len(aligned) == 2


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_sequence_registration_channel_in_info():
    rng = np.random.default_rng(55)
    ref = rng.random((32, 40)).astype(np.float32)
    shifted = np.roll(np.roll(ref, 2, axis=0), -3, axis=1)
    frames = [ref, shifted]
    _, info = align_height_map_sequence_opencv(
        frames,
        reference_index=0,
        method="phase_correlation",
        crop=False,
        registration_channel="gradient",
    )
    assert info.get("registration_channel") == "gradient"


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_sequence_phase_refine_metadata():
    """Second phase pass records per-frame refine displacement."""
    rng = np.random.default_rng(99)
    ref = rng.random((48, 64)).astype(np.float32)
    ref[20:28, 30:38] += 2.0
    shifted = np.roll(np.roll(ref, 3, axis=0), -4, axis=1)
    frames = [ref, shifted]
    aligned, info = align_height_map_sequence_opencv(
        frames,
        reference_index=0,
        method="phase_correlation",
        crop=False,
        phase_refine=True,
    )
    assert info.get("phase_refine") is True
    assert "phase_refine" in info["per_frame"][1]
    assert aligned[1].shape == ref.shape


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_register_affine_orb_known_transform():
    """ORB + RANSAC recovers a known partial-affine warp between two height maps."""
    h, w = 128, 128
    rng = np.random.default_rng(7)
    ref = rng.random((h, w)).astype(np.float32) * 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ref += 2.0 * np.exp(-((yy - 64) ** 2 + (xx - 72) ** 2) / (2 * 10.0**2))

    M_fwd = np.float32([[0.998, -0.06, 5.0], [0.06, 0.998, -4.0]])
    tgt = cv2.warpAffine(
        ref,
        cv2.invertAffineTransform(M_fwd),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    reg, meta = register_heightmaps_affine_orb_ransac(
        ref, tgt, min_inliers=6, orb_nfeatures=1200, ransac_reproj_threshold=4.0
    )
    assert reg.shape == ref.shape
    assert meta.get("method") == "affine_orb_ransac"
    assert meta.get("affine_2x3") is not None
    M_est = np.array(meta["affine_2x3"], dtype=np.float32)
    assert np.allclose(M_est, M_fwd, atol=0.08), f"estimated M diverges from ground truth, meta={meta}"
    c = float(np.corrcoef(ref.ravel(), reg.ravel())[0, 1])
    assert c > 0.78, f"correlation {c} too low, meta={meta}"
