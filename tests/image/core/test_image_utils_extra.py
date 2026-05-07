"""Extra coverage for tmd.image.core.image_utils."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tmd.image.core import image_utils as iu


def test_normalize_array_constant_returns_constant() -> None:
    arr = np.full((3, 3), 5.0)
    out = iu.normalize_array(arr, min_val=0.25, max_val=0.75)
    assert np.allclose(out, 0.25)


def test_normalize_array_empty_returns_zero() -> None:
    out = iu.normalize_array(np.array([]))
    assert out.shape == (1, 1)
    assert out.dtype == np.float32


def test_normalize_array_range_correct() -> None:
    arr = np.array([[0.0, 1.0], [2.0, 3.0]])
    out = iu.normalize_array(arr, 0.0, 1.0)
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_handle_nan_zero_strategy() -> None:
    arr = np.array([[1.0, np.nan], [3.0, 4.0]])
    out = iu.handle_nan_values(arr, strategy="zero")
    assert not np.any(np.isnan(out))
    assert out[0, 1] == 0.0


def test_handle_nan_mean_strategy() -> None:
    arr = np.array([[1.0, np.nan], [3.0, 5.0]])
    out = iu.handle_nan_values(arr, strategy="mean")
    assert not np.any(np.isnan(out))
    assert out[0, 1] == pytest.approx(np.nanmean(arr))


def test_handle_nan_nearest_strategy() -> None:
    arr = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]])
    out = iu.handle_nan_values(arr, strategy="nearest")
    assert not np.any(np.isnan(out))


def test_handle_nan_unknown_strategy_falls_back() -> None:
    arr = np.array([[np.nan, 2.0], [3.0, 4.0]])
    out = iu.handle_nan_values(arr, strategy="bogus")
    assert not np.any(np.isnan(out))
    assert out[0, 0] == 0.0


def test_handle_nan_no_nan_short_circuit() -> None:
    arr = np.array([[1.0, 2.0]])
    out = iu.handle_nan_values(arr)
    assert np.array_equal(out, arr)


def test_prepare_height_map_with_nan_and_normalize(small_heightmap: np.ndarray) -> None:
    arr = small_heightmap.astype(np.float32).copy()
    arr[0, 0] = np.nan
    out = iu.prepare_height_map(arr, normalize=True, nan_strategy="mean")
    assert out.shape == arr.shape
    assert out.min() == pytest.approx(0.0)
    assert out.max() == pytest.approx(1.0)


def test_prepare_height_map_no_normalize() -> None:
    arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    out = iu.prepare_height_map(arr, normalize=False)
    assert np.array_equal(out, arr)


def test_prepare_height_map_blur_branch() -> None:
    arr = np.zeros((4, 4), dtype=np.float32)
    arr[2, 2] = 1.0
    out = iu.prepare_height_map(arr, blur_radius=1.0, normalize=False)
    assert out.shape == arr.shape


def test_save_image_grayscale_png(tmp_path: Path) -> None:
    arr = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    out = tmp_path / "gray.png"
    saved = iu.save_image(arr, str(out), bit_depth=8, normalize=True, format="png")
    assert saved is not None
    assert Path(saved).exists()


def test_save_image_with_colormap(tmp_path: Path) -> None:
    arr = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    out = tmp_path / "viridis.png"
    saved = iu.save_image(arr, str(out), colormap="viridis", format="png")
    assert saved is not None
    assert Path(saved).exists()


def test_save_image_with_metadata(tmp_path: Path) -> None:
    arr = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    out = tmp_path / "meta.png"
    saved = iu.save_image(
        arr,
        str(out),
        format="png",
        metadata={"author": "tmd", "count": 42, "flag": True},
    )
    assert saved is not None


def test_save_image_rgb_uint8(tmp_path: Path) -> None:
    arr = np.full((4, 4, 3), 128, dtype=np.uint8)
    out = tmp_path / "rgb.png"
    saved = iu.save_image(arr, str(out), normalize=False)
    assert saved is not None


def test_save_image_16bit_path(tmp_path: Path) -> None:
    arr = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    out = tmp_path / "16bit.png"
    saved = iu.save_image(arr, str(out), bit_depth=16, normalize=True, format="png")
    assert saved is not None


def test_save_image_appends_extension_when_missing(tmp_path: Path) -> None:
    arr = np.zeros((2, 2), dtype=np.float32)
    out = tmp_path / "no_ext_image"
    saved = iu.save_image(arr, str(out), format="png")
    assert saved is not None
    assert saved.lower().endswith(".png")


def test_resize_image_returns_input_when_no_dims() -> None:
    arr = np.zeros((4, 4), dtype=np.float32)
    out = iu.resize_image(arr)
    assert out is arr


def test_resize_image_preserves_aspect_when_only_width() -> None:
    arr = np.zeros((8, 4), dtype=np.float32)
    out = iu.resize_image(arr, width=2)
    assert out.shape[1] == 2
    assert out.shape[0] >= 1


def test_resize_image_preserves_aspect_when_only_height() -> None:
    arr = np.zeros((8, 4), dtype=np.float32)
    out = iu.resize_image(arr, height=4)
    assert out.shape[0] == 4


def test_resize_image_rgb_input(tmp_path: Path) -> None:
    arr = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = iu.resize_image(arr, width=2, height=2)
    assert out.shape == (2, 2, 3)


def test_resize_image_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        iu.resize_image(np.zeros((1, 2, 7), dtype=np.uint8), width=2, height=2)


def test_is_valid_height_map_true_and_false_branches() -> None:
    assert iu.is_valid_height_map(np.zeros((4, 4))) is True
    assert iu.is_valid_height_map(None) is False  # type: ignore[arg-type]
    assert iu.is_valid_height_map(np.array([])) is False
    assert iu.is_valid_height_map(np.array([["a", "b"]], dtype=object)) is False
    assert iu.is_valid_height_map(np.full((2, 2), np.nan)) is False


def test_get_output_filepath_appends_format() -> None:
    assert iu.get_output_filepath("foo", format="png") == "foo.png"
    assert iu.get_output_filepath("foo.png", format="png") == "foo.png"
    assert iu.get_output_filepath("foo.tif", format="png") == "foo.tif.png"
    assert iu.get_output_filepath("foo") == "foo"
