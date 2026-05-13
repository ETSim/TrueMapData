"""Branch coverage for ``tmd.cli.core.check_dependencies``."""

from __future__ import annotations

import builtins
import subprocess
from typing import Any

import pytest


def test_check_dependencies_success() -> None:
    from tmd.cli.core import check_dependencies

    assert check_dependencies(auto_install=False, exit_on_failure=False) is True


def test_check_dependencies_missing_raises_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tmd.cli import core as core_pkg

    real_import = builtins.__import__

    def guarded_import(name: str, globals_: Any = None, locals_: Any = None, fromlist=(), level: int = 0):
        root = name.split(".", 1)[0]
        if root == "matplotlib":
            raise ImportError("matplotlib unavailable for test")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(SystemExit):
        core_pkg.check_dependencies(auto_install=False, exit_on_failure=True)


def test_check_dependencies_missing_no_exit_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tmd.cli import core as core_pkg

    real_import = builtins.__import__

    def guarded_import(name: str, globals_: Any = None, locals_: Any = None, fromlist=(), level: int = 0):
        root = name.split(".", 1)[0]
        if root == "matplotlib":
            raise ImportError("matplotlib unavailable for test")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    assert core_pkg.check_dependencies(auto_install=False, exit_on_failure=False) is False


def test_check_dependencies_auto_install_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tmd.cli import core as core_pkg

    real_import = builtins.__import__

    def guarded_import(name: str, globals_: Any = None, locals_: Any = None, fromlist=(), level: int = 0):
        root = name.split(".", 1)[0]
        if root == "matplotlib":
            raise ImportError("simulate missing matplotlib for test")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: None)

    assert core_pkg.check_dependencies(auto_install=True, exit_on_failure=False) is True


def test_check_dependencies_auto_install_failure_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tmd.cli import core as core_pkg

    real_import = builtins.__import__

    def guarded_import(name: str, globals_: Any = None, locals_: Any = None, fromlist=(), level: int = 0):
        root = name.split(".", 1)[0]
        if root == "matplotlib":
            raise ImportError("matplotlib unavailable for test")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    def boom(*a, **k):
        raise subprocess.CalledProcessError(1, cmd=a[0] if a else "pip")

    monkeypatch.setattr(subprocess, "check_call", boom)
    assert core_pkg.check_dependencies(auto_install=True, exit_on_failure=False) is False


def test_check_dependencies_auto_install_failure_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tmd.cli import core as core_pkg

    real_import = builtins.__import__

    def guarded_import(name: str, globals_: Any = None, locals_: Any = None, fromlist=(), level: int = 0):
        root = name.split(".", 1)[0]
        if root == "matplotlib":
            raise ImportError("matplotlib unavailable for test")
        return real_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(
        subprocess,
        "check_call",
        lambda *a, **k: (_ for _ in ()).throw(subprocess.CalledProcessError(1, cmd="pip")),
    )
    with pytest.raises(SystemExit):
        core_pkg.check_dependencies(auto_install=True, exit_on_failure=True)
