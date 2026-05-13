"""Smoke tests for ``tmd-wear`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from tmd.cli.main import wear_app as app
from tmd.utils.utils import TMDUtils


def _write_flat_tmd(path: Path, value: float = 0.0) -> None:
    hm = np.full((16, 16), value, dtype=np.float32)
    TMDUtils.write_tmd_file(hm, path, comment="wear-cli\n", version=2, x_length=1.0, y_length=1.0)


def test_wear_help() -> None:
    runner = CliRunner()
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "bearing" in (r.stdout or "").lower()


def test_wear_bearing_curve_json(tmp_path: Path) -> None:
    p = tmp_path / "a.tmd"
    _write_flat_tmd(p, 0.0)
    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["bearing", "curve", str(p), "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert "depths" in data and "rmr_percent" in data


def test_wear_hazard_map_png(tmp_path: Path) -> None:
    p = tmp_path / "a.tmd"
    _write_flat_tmd(p)
    out = tmp_path / "h.png"
    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["hazard-map", str(p), "--out", str(out)])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    assert out.exists()


def test_wear_debris_risk_json(tmp_path: Path) -> None:
    p = tmp_path / "a.tmd"
    _write_flat_tmd(p)
    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["debris-risk", str(p), "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert "summary" in data


def test_wear_volume_series_json(tmp_path: Path) -> None:
    p0 = tmp_path / "f0.tmd"
    p1 = tmp_path / "f1.tmd"
    _write_flat_tmd(p0, 0.1)
    _write_flat_tmd(p1, 0.0)
    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["volume-series", str(p0), str(p1), "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert "rows" in data and len(data["rows"]) == 2


def test_wear_slip_axis_json(tmp_path: Path) -> None:
    p = tmp_path / "a.tmd"
    rng = np.random.default_rng(42)
    hm = rng.random((24, 24)).astype(np.float32) * 0.01
    TMDUtils.write_tmd_file(hm, p, comment="x\n", version=2)
    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["slip-axis", str(p), "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert "psd_wedge_asymmetry" in data


def test_wear_scratch_evolve_two_frames(tmp_path: Path) -> None:
    p0 = tmp_path / "f0.tmd"
    p1 = tmp_path / "f1.tmd"
    _write_defect_like(p0, groove_row=False)
    _write_defect_like(p1, groove_row=True)
    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["scratch-evolve", str(p0), str(p1), "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert "pairs" in data
    assert "series" in data
    assert len(data["series"]) == 2


def test_wear_slip_axis_series_json(tmp_path: Path) -> None:
    p0 = tmp_path / "f0.tmd"
    p1 = tmp_path / "f1.tmd"
    _write_defect_like(p0, groove_row=False)
    _write_defect_like(p1, groove_row=True)
    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["slip-axis-series", str(p0), str(p1), "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert "frames" in data and len(data["frames"]) == 2
    assert data["frames"][0]["file"] == "f0.tmd"


def test_wear_volume_series_align_phase_fft_json(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    ref = rng.random((28, 28)).astype(np.float32) * 0.02
    ref[8:12, 8:12] += 0.2
    mov = np.roll(np.roll(ref, 2, axis=0), -1, axis=1)
    p0 = tmp_path / "a.tmd"
    p1 = tmp_path / "b.tmd"
    TMDUtils.write_tmd_file(ref, p0, comment="x\n", version=2, x_length=1.0, y_length=1.0)
    TMDUtils.write_tmd_file(mov, p1, comment="x\n", version=2, x_length=1.0, y_length=1.0)
    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["volume-series", str(p0), str(p1), "--align", "phase-fft", "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert "alignment" in data
    assert data["alignment"].get("method") == "phase_fft_numpy"


def _write_defect_like(path: Path, *, groove_row: bool) -> None:
    size = 64
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(x, y)
    height_map = 0.05 * np.sin(3.0 * np.pi * xx) + 0.05 * np.cos(4.0 * np.pi * yy)
    if groove_row:
        height_map[30:32, 8:56] -= 0.35
    else:
        height_map[30:32, 8:56] -= 0.05
    TMDUtils.write_tmd_file(height_map.astype(np.float32), path, comment="scratch\n", version=2)
