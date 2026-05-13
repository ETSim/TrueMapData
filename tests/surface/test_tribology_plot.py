"""Tests for ``tmd.surface.metrics.save_tribology_dashboard_png`` PNG dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tmd.surface.metrics import save_tribology_dashboard_png


def test_save_tribology_dashboard_png_writes_file(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    z = (rng.standard_normal((36, 40)).astype(np.float32) * 0.02).cumsum(axis=0).cumsum(axis=1)
    z -= float(z.mean())
    out = tmp_path / "tribo.png"
    save_tribology_dashboard_png(
        z,
        {"mmpp": 0.01},
        title="synthetic",
        output_path=out,
        plane_removal="mean",
        z_reference="mean",
        curve_n=12,
        dpi=80.0,
        include_proxy_maps=True,
        include_anomaly_angle=False,
    )
    assert out.is_file()
    assert out.stat().st_size > 2_000


def test_save_tribology_dashboard_no_maps(tmp_path: Path) -> None:
    z = np.zeros((24, 28), dtype=np.float32)
    z[8:16, 10:18] = 0.5
    out = tmp_path / "tribo_compact.png"
    save_tribology_dashboard_png(
        z,
        None,
        title="flat+pad",
        output_path=out,
        plane_removal="none",
        curve_n=8,
        dpi=72.0,
        include_proxy_maps=False,
    )
    assert out.is_file()
