"""Tests for :mod:`tmd.surface.metrics` (debris score)."""

from __future__ import annotations

import numpy as np

from tmd.surface.metrics import debris_pocket_score


def test_debris_flat_map_low_scores() -> None:
    z = np.zeros((64, 64), dtype=np.float64)
    score, meta = debris_pocket_score(z)
    assert score.shape == z.shape
    assert float(meta["mean_score"]) < 0.5


def test_debris_deterministic() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(48, 48)).astype(np.float64) * 0.01
    _, m1 = debris_pocket_score(z)
    _, m2 = debris_pocket_score(z)
    assert m1["mean_score"] == m2["mean_score"]
