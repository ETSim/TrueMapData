"""TMDSequence wiring for TextureFriction-style SIFT alignment (OpenCV)."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.core.sequence import TMDSequence
from tmd.sequence.alignment import NormalMapSequenceAlignmentConfig, _has_cv2


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_height_maps_sift_identity_stack() -> None:
    rng = np.random.default_rng(42)
    h, w = 48, 56
    z = rng.standard_normal((h, w)).astype(np.float32) * 0.01
    seq = TMDSequence("t")
    seq.add_frame(z.copy(), "a", {})
    seq.add_frame(z.copy(), "b", {})
    cfg = NormalMapSequenceAlignmentConfig(sift_features=300)
    info = seq.align_height_maps_sift(reference_index=0, two_pass=False, crop=False, config=cfg)
    assert info.get("method", "").startswith("opencv_sift_height")
    assert len(seq.frames) == 2
    np.testing.assert_allclose(seq.frames[0], z, rtol=0, atol=1e-5)
    np.testing.assert_allclose(seq.frames[1], z, rtol=0, atol=1e-5)


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_height_maps_from_normals_identity_stack() -> None:
    rng = np.random.default_rng(7)
    h, w = 48, 56
    z = rng.standard_normal((h, w)).astype(np.float32) * 0.01
    seq = TMDSequence("t")
    seq.add_frame(z.copy(), "a", {})
    seq.add_frame(z.copy(), "b", {})
    cfg = NormalMapSequenceAlignmentConfig(sift_features=300)
    info = seq.align_height_maps_from_normals(
        reference_index=0, two_pass=False, crop=False, config=cfg
    )
    assert info.get("registration_source") == "normals"
    assert len(seq.frames) == 2
    np.testing.assert_allclose(seq.frames[0], z, rtol=0, atol=2e-4)
    np.testing.assert_allclose(seq.frames[1], z, rtol=0, atol=2e-4)


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_align_height_maps_sift_restores_order_with_reference_index() -> None:
    """reference_index=1 should treat frame 1 as SIFT anchor (algorithm frame 0) then unpermute."""
    h, w = 32, 40
    z0 = np.zeros((h, w), dtype=np.float32)
    z1 = np.ones((h, w), dtype=np.float32) * 0.5
    seq = TMDSequence("t")
    seq.add_frame(z0.copy(), "0", {})
    seq.add_frame(z1.copy(), "1", {})
    cfg = NormalMapSequenceAlignmentConfig(sift_features=200)
    info = seq.align_height_maps_sift(reference_index=1, two_pass=False, crop=False, config=cfg)
    assert info["reference_index"] == 1
    assert seq.frames[0].shape == (h, w)
    assert seq.frames[1].shape == (h, w)
    np.testing.assert_array_equal(seq.frames[0], z0)
    np.testing.assert_array_equal(seq.frames[1], z1)


@pytest.mark.skipif(not _has_cv2, reason="OpenCV not available")
def test_sequential_wear_metrics_align_before_sift() -> None:
    rng = np.random.default_rng(1)
    h, w = 40, 44
    z = rng.standard_normal((h, w)).astype(np.float32) * 0.02
    seq = TMDSequence("w")
    seq.add_frame(z.copy(), "a", {})
    seq.add_frame(z.copy(), "b", {})
    cfg = NormalMapSequenceAlignmentConfig(sift_features=250)
    out = seq.sequential_wear_metrics(
        dx_mm=0.01,
        dy_mm=0.01,
        reference_index=0,
        align_before="sift",
        align_sift_kwargs={"two_pass": False, "crop": False, "config": cfg},
    )
    assert "alignment" in out
    assert out["alignment"].get("method", "").startswith("opencv_sift_height")
