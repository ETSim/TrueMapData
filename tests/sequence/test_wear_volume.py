"""Tests for :mod:`tmd.sequence.wear_analysis` (volume series)."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.sequence.wear_analysis import (
    localization_index_top_fraction,
    positive_loss_volume,
    wear_incremental_series,
    wear_series_vs_reference,
)


def test_positive_loss_volume_constant_step() -> None:
    ref = np.ones((4, 4), dtype=np.float64)
    cur = np.zeros_like(ref)
    vol, _ = positive_loss_volume(ref, cur, dx=0.5, dy=0.5)
    assert vol == pytest.approx(16 * 0.25 * 1.0)


def test_identical_frames_zero_volume() -> None:
    z = np.random.default_rng(1).random((8, 8)).astype(np.float64)
    rows = wear_series_vs_reference([z, z], reference_index=0, dx=1.0, dy=1.0)
    assert rows[0]["volume_positive_loss"] == 0.0
    assert rows[1]["volume_positive_loss"] == 0.0


def test_shape_mismatch_raises() -> None:
    a = np.zeros((2, 2))
    b = np.zeros((3, 3))
    with pytest.raises(ValueError):
        wear_series_vs_reference([a, b], reference_index=0, dx=1.0, dy=1.0)


def test_localization_index_concentrated_mass() -> None:
    """~90% of positive loss on 10% of pixels → localization index high."""
    loss = np.full((10, 10), 0.01, dtype=np.float64)
    loss[:1, :] = 1.0  # 10 pixels at 1.0, 90 at 0.01
    idx = localization_index_top_fraction(loss, top_fraction=0.10)
    assert idx > 0.85


def test_incremental_cumulative_monotone() -> None:
    z0 = np.zeros((4, 4), dtype=np.float64)
    z1 = np.full((4, 4), -0.1, dtype=np.float64)
    z2 = np.full((4, 4), -0.2, dtype=np.float64)
    rows = wear_incremental_series([z0, z1, z2], dx=1.0, dy=1.0, top_fraction=0.10)
    assert rows[0]["cumulative_incremental_volume"] == 0.0
    assert rows[1]["cumulative_incremental_volume"] >= rows[0]["cumulative_incremental_volume"]
    assert rows[2]["cumulative_incremental_volume"] >= rows[1]["cumulative_incremental_volume"]
