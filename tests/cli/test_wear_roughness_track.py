"""Optional Surfalize-backed ``tmd-wear roughness-track`` test."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

pytest.importorskip("surfalize")

from tmd.cli.main import wear_app as app
from tmd.utils.utils import TMDUtils


def test_wear_roughness_track_json(tmp_path: Path) -> None:
    p = tmp_path / "a.tmd"
    hm = np.sin(np.linspace(0, 3, 32)).astype(np.float32).reshape(1, -1)
    hm = np.broadcast_to(hm, (32, 32)).copy()
    TMDUtils.write_tmd_file(hm, p, comment="rt\n", version=2, x_length=1.0, y_length=1.0)

    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["roughness-track", str(p), "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert "rows" in data
    row0 = data["rows"][0]
    assert "Sp" in row0 or "__error__" in row0
