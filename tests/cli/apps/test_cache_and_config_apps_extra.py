"""Extra coverage for cache/config CLI apps."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from typer.testing import CliRunner

from tmd.cli.apps import cache_app as cache_mod
from tmd.cli.apps import config_app as config_mod


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


@dataclass
class _FakeConsole:
    printed: list[object] = field(default_factory=list)
    status_messages: list[str] = field(default_factory=list)

    def print(self, value: object) -> None:
        self.printed.append(value)

    def status(self, msg: str):
        self.status_messages.append(msg)

        class _Ctx:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _Ctx()


def test_cache_info_clear_and_clear_all_commands(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_console = _FakeConsole()
    monkeypatch.setattr(cache_mod, "console", fake_console)
    monkeypatch.setattr(
        cache_mod,
        "get_cache_stats",
        lambda: {
            "cache_dir": "tmp/cache",
            "entry_count": 5,
            "expired_count": 2,
            "total_size_mb": 1.25,
        },
    )
    monkeypatch.setattr(cache_mod, "clear_cache", lambda expired_only=True: 2 if expired_only else 5)

    success_messages: list[str] = []
    monkeypatch.setattr(cache_mod, "print_success", lambda msg: success_messages.append(msg))

    app = cache_mod.create_cache_app()

    info = runner.invoke(app, ["info"])
    assert info.exit_code == 0
    assert len(fake_console.printed) == 1

    clear_expired = runner.invoke(app, ["clear"])
    assert clear_expired.exit_code == 0
    assert any("expired entries" in msg for msg in success_messages)

    clear_all = runner.invoke(app, ["clear", "--no-expired-only"])
    assert clear_all.exit_code == 0
    assert any("entire cache" in msg for msg in success_messages)

    clear_all_cmd = runner.invoke(app, ["clear-all"])
    assert clear_all_cmd.exit_code == 0
    assert fake_console.status_messages


def test_cache_commands_unavailable_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []
    monkeypatch.setattr(cache_mod, "print_error", lambda msg: errors.append(msg))
    monkeypatch.setattr(cache_mod, "get_cache_stats", lambda: (_ for _ in ()).throw(NameError("missing")))
    monkeypatch.setattr(cache_mod, "clear_cache", lambda **_k: (_ for _ in ()).throw(ImportError("missing")))

    assert cache_mod.cache_info_command() == 1
    assert cache_mod.cache_clear_command(expired_only=True) == 1
    assert cache_mod.cache_clear_all_command() == 1
    assert len(errors) == 3


def test_config_show_set_and_reset(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_console = _FakeConsole()
    monkeypatch.setattr(config_mod, "console", fake_console)

    cfg = {"z_last": "x", "a_first": 1}
    saved: list[dict] = []
    success_messages: list[str] = []

    monkeypatch.setattr(config_mod, "load_config", lambda: dict(cfg))
    monkeypatch.setattr(config_mod, "save_config", lambda payload: saved.append(dict(payload)))
    monkeypatch.setattr(config_mod, "print_success", lambda msg: success_messages.append(msg))

    app = config_mod.create_config_app()

    shown = runner.invoke(app, ["show"])
    assert shown.exit_code == 0
    # Header + sorted key-value rows
    assert len(fake_console.printed) >= 3
    assert fake_console.printed[1] == "a_first: 1"
    assert fake_console.printed[2] == "z_last: x"

    assert runner.invoke(app, ["set", "flag", "true"]).exit_code == 0
    assert saved[-1]["flag"] is True
    assert runner.invoke(app, ["set", "flag", "false"]).exit_code == 0
    assert saved[-1]["flag"] is False
    assert runner.invoke(app, ["set", "count", "12"]).exit_code == 0
    assert saved[-1]["count"] == 12
    assert runner.invoke(app, ["set", "scale", "3.5"]).exit_code == 0
    assert saved[-1]["scale"] == 3.5
    assert runner.invoke(app, ["set", "name", "demo"]).exit_code == 0
    assert saved[-1]["name"] == "demo"

    reset = runner.invoke(app, ["reset"])
    assert reset.exit_code == 0
    assert saved[-1]["default_colormap"] == "viridis"
    assert any("updated" in msg or "reset" in msg for msg in success_messages)
