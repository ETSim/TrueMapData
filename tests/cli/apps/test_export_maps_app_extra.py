"""Smoke coverage for ``maps`` sub-commands in :mod:`tmd.cli.apps.export_maps_app`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from tmd.cli.main import app
from tmd.utils.utils import TMDUtils


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


@pytest.fixture
def tiny_tmd(tmp_path: Path) -> Path:
    import numpy as np

    hm = np.array([[0.0, 0.5], [0.2, 0.8]], dtype=np.float32)
    p = tmp_path / "in.tmd"
    TMDUtils.write_tmd_file(hm, str(p), comment="m\n", version=2)
    return p


def _patch_export(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    m = MagicMock(return_value=True)
    monkeypatch.setattr("tmd.cli.apps.export_maps_app.export_maps_command", m)
    return m


def test_maps_subcommands_delegate_export_maps(runner: CliRunner, tiny_tmd: Path, monkeypatch) -> None:
    mock = _patch_export(monkeypatch)

    for cmd, types in [
        ("ao", ["ao"]),
        ("bump", ["bump"]),
        ("roughness", ["roughness"]),
        ("metallic", ["metallic"]),
        ("displacement", ["displacement"]),
        ("height", ["height"]),
        ("hillshade", ["hillshade"]),
        ("parallax_ao", ["parallax_ao"]),
        ("depth", ["depth"]),
    ]:
        mock.reset_mock()
        r = runner.invoke(app, ["maps", cmd, str(tiny_tmd)])
        assert r.exit_code == 0, r.stdout + r.stderr
        assert mock.called
        assert mock.call_args[0][2] == types

    mock.reset_mock()
    r = runner.invoke(app, ["maps", "all", str(tiny_tmd), "--output", str(tiny_tmd.parent / "out_all")])
    assert r.exit_code == 0
    assert mock.called


def test_maps_angle_and_curvature_tuple_parsing(runner: CliRunner, tiny_tmd: Path, monkeypatch) -> None:
    mock = _patch_export(monkeypatch)
    r = runner.invoke(
        app,
        ["maps", "angle", str(tiny_tmd), "--highlight-range", "10:20"],
    )
    assert r.exit_code == 0
    assert mock.call_args[1].get("highlight_range") == (10.0, 20.0)

    mock.reset_mock()
    r = runner.invoke(
        app,
        ["maps", "curvature", str(tiny_tmd), "--percentile-clip", "1:99"],
    )
    assert r.exit_code == 0
    assert mock.call_args[1].get("percentile_clip") == (1.0, 99.0)


def test_maps_angle_bad_range_exits(runner: CliRunner, tiny_tmd: Path) -> None:
    r = runner.invoke(
        app,
        ["maps", "angle", str(tiny_tmd), "--highlight-range", "not_a_range"],
    )
    assert r.exit_code != 0


def test_maps_batch_and_synthetic(runner: CliRunner, tmp_path: Path, tiny_tmd: Path, monkeypatch) -> None:
    mock = _patch_export(monkeypatch)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    import shutil

    shutil.copy(tiny_tmd, data_dir / "a.tmd")

    r = runner.invoke(
        app,
        ["maps", "batch", str(data_dir), "--output", str(tmp_path / "tex"), "--pattern", "*.tmd"],
    )
    assert r.exit_code == 0
    assert mock.called

    monkeypatch.setattr(
        "tmd.cli.apps.export_maps_app.generate_synthetic_terrain",
        lambda *a, **k: True,
    )
    r2 = runner.invoke(
        app,
        ["maps", "synthetic", "flat", "--width", "8", "--height", "8", "--output", str(tmp_path / "syn")],
    )
    assert r2.exit_code == 0
