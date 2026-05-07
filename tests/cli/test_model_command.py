"""CLI tests for mesh / model export commands (``mesh`` sub-app)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tmd.cli.main import app
from tmd.utils.utils import TMDUtils


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


def test_mesh_formats_lists_formats(runner: CliRunner) -> None:
    r = runner.invoke(app, ["mesh", "formats"])
    assert r.exit_code == 0
    assert "stl" in r.stdout.lower()


def test_mesh_generate_stl(tmp_path: Path, runner: CliRunner, small_heightmap) -> None:
    tmd_path = tmp_path / "in.tmd"
    TMDUtils.write_tmd_file(small_heightmap, str(tmd_path), comment="cli\n", version=2)
    out = tmp_path / "out.stl"
    r = runner.invoke(
        app,
        [
            "mesh",
            "generate",
            str(tmd_path),
            "--format",
            "stl",
            "--output-file",
            str(out),
            "--max-triangles",
            "300",
            "--quality",
            "high",
        ],
    )
    assert r.exit_code == 0
    assert out.exists()


def test_mesh_generate_help(runner: CliRunner) -> None:
    r = runner.invoke(app, ["mesh", "generate", "--help"])
    assert r.exit_code == 0
