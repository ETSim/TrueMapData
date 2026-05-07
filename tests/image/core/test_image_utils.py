"""Tests for image utility helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tmd.image.core.image_utils import (
    get_output_filepath,
    handle_nan_values,
    is_valid_height_map,
    normalize_array,
    prepare_height_map,
    resize_image,
    save_image,
)


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
    assert restored.dtype.kind in {"i", "u"}
    assert restored.max() > 255
    assert np.array_equal(restored.astype(np.uint16), img16)


def test_normalize_array_empty_returns_single_zero() -> None:
    out = normalize_array(np.array([]))
    assert out.shape == (1, 1)
    assert out.dtype == np.float32
    assert float(out[0, 0]) == 0.0


def test_normalize_array_constant_is_min_val() -> None:
    h = np.ones((2, 3), dtype=np.float64)
    out = normalize_array(h, min_val=-1.0, max_val=1.0)
    assert np.all(out == -1.0)


def test_normalize_array_scales_to_range() -> None:
    h = np.array([[0.0, 10.0], [5.0, 5.0]], dtype=np.float32)
    out = normalize_array(h, min_val=0.0, max_val=100.0)
    assert float(out.min()) == 0.0
    assert float(out.max()) == 100.0


def test_handle_nan_values_no_nan_unchanged() -> None:
    h = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    assert np.array_equal(handle_nan_values(h), h)


def test_handle_nan_values_zero_strategy() -> None:
    h = np.array([[1.0, np.nan], [np.nan, 2.0]], dtype=np.float64)
    out = handle_nan_values(h, strategy="zero")
    assert not np.any(np.isnan(out))
    assert out[0, 1] == 0.0


def test_handle_nan_values_mean_strategy() -> None:
    h = np.array([[2.0, np.nan], [4.0, np.nan]], dtype=np.float64)
    out = handle_nan_values(h, strategy="mean")
    assert not np.any(np.isnan(out))
    assert out[0, 1] == 3.0


def test_get_output_filepath_adds_extension() -> None:
    assert get_output_filepath("/tmp/foo", "png").endswith(".png")


def test_is_valid_height_map() -> None:
    assert is_valid_height_map(np.ones((2, 2), dtype=np.float32)) is True
    assert is_valid_height_map(np.full((2, 2), np.nan)) is False


def test_resize_image_identity_and_resize() -> None:
    img = np.ones((10, 10), dtype=np.float32)
    assert resize_image(img).shape == img.shape
    out = resize_image(img, width=5, height=5)
    assert out.shape[0] == 5 and out.shape[1] == 5


def test_prepare_height_map_from_array() -> None:
    h = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float64)
    out = prepare_height_map(h, normalize=False)
    assert out.dtype == np.float32


def test_prepare_height_map_from_tmd(tmp_tmd_path: Path) -> None:
    out = prepare_height_map(tmp_tmd_path, normalize=True)
    assert out.ndim == 2


def test_handle_nan_unknown_strategy_fallback() -> None:
    h = np.array([[np.nan]], dtype=np.float64)
    out = handle_nan_values(h, strategy="not_a_real_strategy")
    assert not np.any(np.isnan(out))


def test_handle_nan_values_nearest_uses_scipy_when_available() -> None:
    pytest.importorskip("scipy")
    h = np.array(
        [[1.0, 1.0, 1.0], [1.0, np.nan, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    out = handle_nan_values(h, strategy="nearest")
    assert not np.any(np.isnan(out))
    assert np.isfinite(out).all()
