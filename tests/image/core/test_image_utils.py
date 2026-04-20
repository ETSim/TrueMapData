"""Tests for image utility helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from tmd.image.core.image_utils import save_image


def test_save_image_preserves_uint8_when_not_normalized(tmp_path: Path) -> None:
    img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    out = tmp_path / "uint8.png"

    written = save_image(img, str(out), normalize=False, format="png")

    assert written is not None
    restored = np.array(Image.open(out))
    assert np.array_equal(restored, img)


def test_save_image_preserves_uint16_when_not_normalized(tmp_path: Path) -> None:
    img16 = np.array([[0, 65535], [32768, 1024]], dtype=np.uint16)
    out = tmp_path / "uint16.png"

    written = save_image(img16, str(out), bit_depth=16, normalize=False, format="png")

    assert written is not None
    restored = np.array(Image.open(out))
    assert restored.dtype == np.uint16
    assert np.array_equal(restored, img16)
