"""Extra coverage for ``NormalMapGenerator`` branches."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.image.maps.normal import NormalMapGenerator


@pytest.fixture
def ramp_height_map() -> np.ndarray:
    return np.tile(np.linspace(0.0, 1.0, 8, dtype=np.float32), (8, 1))


def test_generate_with_flat_height_map_returns_valid_rgb() -> None:
    gen = NormalMapGenerator()
    out = gen.generate(np.zeros((8, 8), dtype=np.float32))
    assert out.shape == (8, 8, 3)
    assert np.all(np.isfinite(out))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_generate_with_ramp_normal_z_dominant(ramp_height_map: np.ndarray) -> None:
    gen = NormalMapGenerator()
    out = gen.generate(ramp_height_map)
    assert out.shape == (8, 8, 3)
    assert np.all(np.isfinite(out))


def test_generate_with_strength_kwarg(ramp_height_map: np.ndarray) -> None:
    gen = NormalMapGenerator()
    out_low = gen.generate(ramp_height_map, strength=0.1)
    out_high = gen.generate(ramp_height_map, strength=5.0)
    diff_low = np.linalg.norm(out_low[..., :2] - 0.5, axis=2).mean()
    diff_high = np.linalg.norm(out_high[..., :2] - 0.5, axis=2).mean()
    assert diff_high >= diff_low


def test_generate_with_normalize_kwarg(ramp_height_map: np.ndarray) -> None:
    gen = NormalMapGenerator(normalize=True)
    out = gen.generate(ramp_height_map * 100.0)
    assert out.shape == (8, 8, 3)
    assert np.all(np.isfinite(out))


def test_generate_with_metadata_x_y_length(ramp_height_map: np.ndarray) -> None:
    gen = NormalMapGenerator()
    out = gen.generate(ramp_height_map, metadata={"x_length": 10.0, "y_length": 10.0})
    assert np.all(np.isfinite(out))


def test_generate_with_metadata_mmpp_and_magnification(ramp_height_map: np.ndarray) -> None:
    gen = NormalMapGenerator()
    out = gen.generate(ramp_height_map, metadata={"mmpp": 0.05, "magnification": 2.0})
    assert np.all(np.isfinite(out))


def test_generate_handles_nan_in_height_map() -> None:
    arr = np.zeros((6, 6), dtype=np.float32)
    arr[0, 0] = np.nan
    gen = NormalMapGenerator()
    out = gen.generate(arr)
    assert np.all(np.isfinite(out))


def test_generate_with_too_small_height_map() -> None:
    gen = NormalMapGenerator()
    out = gen.generate(np.zeros((2, 2), dtype=np.float32))
    assert out.shape == (2, 2, 3)


def test_generate_with_debug_kwarg_logs(capsys: pytest.CaptureFixture[str]) -> None:
    gen = NormalMapGenerator()
    gen.generate(np.zeros((4, 4), dtype=np.float32), debug=True)
    captured = capsys.readouterr()
    assert "NormalMapGenerator" in captured.out


def test_validate_params_clamps_invalid_strength() -> None:
    gen = NormalMapGenerator()
    fixed = gen._validate_params({"strength": -1.0, "normalize": "yes"})
    assert fixed["strength"] == 1.0
    assert fixed["normalize"] is True


def test_prepare_height_map_handles_none() -> None:
    gen = NormalMapGenerator()
    out = gen._prepare_height_map(None, normalize=True)
    assert out.shape == (1, 1)


def test_prepare_height_map_flat_with_normalize_short_circuit() -> None:
    gen = NormalMapGenerator()
    out = gen._prepare_height_map(np.full((4, 4), 0.5, dtype=np.float32), normalize=True)
    assert out.shape == (4, 4)


def test_log_and_print_levels(capsys: pytest.CaptureFixture[str]) -> None:
    gen = NormalMapGenerator(debug=True)
    gen._log_and_print("info-msg", level="info")
    gen._log_and_print("warning-msg", level="warning")
    gen._log_and_print("error-msg", level="error")
    gen._log_and_print("debug-msg", level="debug")
    out = capsys.readouterr().out
    assert "info-msg" in out
