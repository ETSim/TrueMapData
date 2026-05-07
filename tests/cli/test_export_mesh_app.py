"""CLI tests targeting mesh export app wiring."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tmd.cli.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


def test_mesh_batch_help(runner: CliRunner) -> None:
    r = runner.invoke(app, ["mesh", "batch", "--help"])
    assert r.exit_code == 0


def test_mesh_list_legacy_help(runner: CliRunner) -> None:
    r = runner.invoke(app, ["mesh", "list-legacy"])
    assert r.exit_code == 0
