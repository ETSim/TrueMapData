"""CLI smoke tests for ``maps`` export app."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tmd.cli.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


def test_maps_height_help(runner: CliRunner) -> None:
    r = runner.invoke(app, ["maps", "height", "--help"])
    assert r.exit_code == 0


def test_maps_all_help(runner: CliRunner) -> None:
    r = runner.invoke(app, ["maps", "all", "--help"])
    assert r.exit_code == 0
