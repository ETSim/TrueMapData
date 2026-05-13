"""Smoke-load canonical example TMDs when present (GelSight + v1/v2 Dime)."""
from __future__ import annotations

from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
EXAMPLE_TMDS = [
    _REPO / "examples" / "gelsight" / "circle_0mm_100g_heightmap_linear_detrend.tmd",
    _REPO / "examples" / "v1" / "Dime.tmd",
    _REPO / "examples" / "v2" / "Dime.tmd",
]


@pytest.mark.parametrize(
    "tmd_path",
    EXAMPLE_TMDS,
    ids=[str(p.relative_to(_REPO)) for p in EXAMPLE_TMDS],
)
def test_example_tmd_loads(tmd_path: Path) -> None:
    if not tmd_path.is_file():
        pytest.skip(f"example not in workspace: {tmd_path}")

    from tmd import TMD

    tmd = TMD.load(str(tmd_path))
    assert tmd.height_map.ndim == 2
    assert tmd.height_map.size > 0
    h, w = tmd.height_map.shape
    assert h > 1 and w > 1


@pytest.mark.parametrize(
    "tmd_path",
    EXAMPLE_TMDS[:2],
    ids=[str(p.relative_to(_REPO)) for p in EXAMPLE_TMDS[:2]],
)
def test_example_tmd_tribology_core(tmd_path: Path) -> None:
    """GelSight + v1 Dime: tribology helpers run without Surfalize (numpy path)."""
    if not tmd_path.is_file():
        pytest.skip(f"example not in workspace: {tmd_path}")

    import numpy as np

    from tmd import TMD
    from tmd.surface.metrics import bearing_area_curve, preferred_slip_axis

    tmd = TMD.load(str(tmd_path))
    z = np.asarray(tmd.height_map, dtype=np.float32)
    meta = tmd.metadata or {}
    axis = preferred_slip_axis(z, meta, plane_removal="mean", include_anomaly_angle=False)
    assert "axis_rad" in axis
    curve = bearing_area_curve(z, n=8, metadata=meta, plane_removal="mean")
    assert curve["separations"].size == 8
    assert 0.0 <= float(curve["area_fraction"][0]) <= 1.0


@pytest.mark.parametrize(
    "tmd_path",
    EXAMPLE_TMDS,
    ids=[str(p.relative_to(_REPO)) for p in EXAMPLE_TMDS],
)
def test_example_tmd_tmd_wear_bearing_json(tmd_path: Path) -> None:
    """``tmd-wear bearing curve`` runs on each canonical example when present."""
    if not tmd_path.is_file():
        pytest.skip(f"example not in workspace: {tmd_path}")

    import json

    from typer.testing import CliRunner

    from tmd.cli.main import wear_app as app

    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["bearing", "curve", str(tmd_path), "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert "depths" in data and "rmr_percent" in data


def test_example_gelsight_tmd_wear_volume_pair() -> None:
    """Two-frame wear volume on GelSight example pair when both exist."""
    p0 = _REPO / "examples" / "gelsight" / "circle_0mm_100g_heightmap_linear_detrend.tmd"
    p1 = _REPO / "examples" / "gelsight" / "circle_worn_0mm_100g_heightmap_linear_detrend.tmd"
    if not p0.is_file() or not p1.is_file():
        pytest.skip("gelsight example pair not in workspace")

    import json

    from typer.testing import CliRunner

    from tmd.cli.main import wear_app as app

    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["volume-series", str(p0), str(p1), "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert len(data["rows"]) == 2
