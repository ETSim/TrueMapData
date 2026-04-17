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
