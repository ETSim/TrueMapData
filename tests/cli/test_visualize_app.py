"""CLI smoke tests for ``visualize`` app."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tmd.cli.main import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


def test_visualize_backends(runner: CliRunner) -> None:
    r = runner.invoke(app, ["visualize", "backends"])
    assert r.exit_code == 0


def test_visualize_basic_help(runner: CliRunner) -> None:
    r = runner.invoke(app, ["visualize", "basic", "--help"])
    assert r.exit_code == 0
