"""Thin smoke tests for ``info``, ``sequence``, and extra ``visualize`` CLI paths."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

sys.modules.setdefault("noise", SimpleNamespace(snoise2=lambda *a, **k: 0.0))

from tmd.cli.main import app
from tmd.utils.utils import TMDUtils


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


def test_info_command_help(runner: CliRunner) -> None:
    r = runner.invoke(app, ["info", "--help"])
    assert r.exit_code == 0
    assert "tmd" in r.stdout.lower() or "file" in r.stdout.lower()


def test_info_command_runs_on_small_tmd(runner: CliRunner, tmp_path: Path) -> None:
    tmd_path = tmp_path / "sample.tmd"
    TMDUtils.write_tmd_file(
        np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4),
        str(tmd_path),
        comment="cli smoke",
        x_length=1.0,
        y_length=1.0,
    )
    r = runner.invoke(app, ["info", str(tmd_path)])
    assert r.exit_code == 0


def test_check_command_runs(runner: CliRunner) -> None:
    r = runner.invoke(app, ["check"])
    assert r.exit_code == 0


def test_sequence_align_help(runner: CliRunner) -> None:
    r = runner.invoke(app, ["sequence", "align", "--help"])
    assert r.exit_code == 0
    assert "align" in r.stdout.lower() or "reference" in r.stdout.lower()


def test_sequence_export_help(runner: CliRunner) -> None:
    r = runner.invoke(app, ["sequence", "export", "--help"])
    assert r.exit_code == 0
    assert "aligned" in r.stdout.lower()


def test_sequence_export_no_maps_no_mesh(runner: CliRunner, tmp_path: Path) -> None:
    d = tmp_path / "aligned"
    d.mkdir()
    tmd = d / "frame01_aligned.tmd"
    TMDUtils.write_tmd_file(
        np.ones((8, 8), dtype=np.float32),
        str(tmd),
        comment="aligned",
        x_length=2.0,
        y_length=2.0,
    )
    r = runner.invoke(app, ["sequence", "export", str(d), "--no-maps", "--no-mesh"])
    assert r.exit_code == 0


@pytest.mark.parametrize(
    "sub,needle",
    [
        ("ps-pointcloud", "point"),
        ("ps-mesh", "mesh"),
        ("contour", "contour"),
    ],
)
def test_visualize_subcommand_help(runner: CliRunner, sub: str, needle: str) -> None:
    r = runner.invoke(app, ["visualize", sub, "--help"])
    assert r.exit_code == 0
    assert needle in r.stdout.lower()
