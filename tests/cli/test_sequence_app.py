"""CLI smoke tests for ``sequence`` app."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tmd.cli.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


def test_sequence_help(runner: CliRunner) -> None:
    r = runner.invoke(app, ["sequence", "--help"])
    assert r.exit_code == 0


def test_sequence_align_help(runner: CliRunner) -> None:
    r = runner.invoke(app, ["sequence", "align", "--help"])
    assert r.exit_code == 0
