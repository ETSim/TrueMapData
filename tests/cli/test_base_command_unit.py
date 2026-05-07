"""Unit tests for ``tmd.cli.commands.base``."""

from __future__ import annotations

import importlib

import pytest

from tmd.cli.commands import base as base_mod


def test_base_command_convert_value_types() -> None:
    cmd = base_mod.BaseCommand("test", "desc")
    assert cmd._convert_value_type("TRUE") is True
    assert cmd._convert_value_type("no") is False
    assert cmd._convert_value_type("7") == 7
    assert cmd._convert_value_type("2.5") == 2.5
    assert cmd._convert_value_type("hello") == "hello"


def test_base_command_display_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        base_mod,
        "load_config",
        lambda: {"_secret": 1, "recent_files": ["a.tmd"], "theme": "dark"},
    )
    printed: list[str] = []

    monkeypatch.setattr(base_mod, "print_rich_table", lambda *a, **k: printed.append("table"))
    cmd = base_mod.BaseCommand("c", "d")
    cmd.display_config()
    assert printed == ["table"]


def test_base_command_update_config(monkeypatch: pytest.MonkeyPatch) -> None:
    saved: dict = {}
    msgs: list[str] = []

    monkeypatch.setattr(base_mod, "load_config", lambda: {})
    monkeypatch.setattr(base_mod, "save_config", lambda cfg: saved.update(cfg))
    monkeypatch.setattr(base_mod, "print_success", lambda m: msgs.append(m))

    cmd = base_mod.BaseCommand("c", "d")
    cmd.update_config("k", "v")
    assert saved.get("k") == "v" and msgs


@pytest.mark.filterwarnings("ignore:.*pkg_resources.*:DeprecationWarning")
def test_check_dependencies_all_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    pr = importlib.import_module("pkg_resources")

    class _Dist:
        def __init__(self, key: str) -> None:
            self.key = key
            self.version = "0.0"

    fake_set = [
        _Dist("matplotlib"),
        _Dist("rich"),
        _Dist("typer"),
        _Dist("plotly"),
        _Dist("seaborn"),
    ]
    monkeypatch.setattr(pr, "working_set", fake_set)
    assert base_mod.check_dependencies_and_install() is True
