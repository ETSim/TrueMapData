"""Extra coverage for tmd.cli.commands.visualize."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest

import tmd.cli.commands.visualize as visualize_mod


@pytest.fixture(autouse=True)
def force_matplotlib_agg() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)


def test_get_available_plotters_factory_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """When tmd.plotters exposes get_registered_plotters it is returned as-is."""
    fake = {"matplotlib": True, "plotly": False}

    import sys
    import types

    fake_pkg = types.ModuleType("tmd.plotters")
    fake_pkg.get_registered_plotters = lambda: fake
    monkeypatch.setitem(sys.modules, "tmd.plotters", fake_pkg)
    out = visualize_mod.get_available_plotters()
    assert out == fake


def test_get_available_plotters_falls_back_to_find_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the import fails, the helper builds the dict from importlib.util.find_spec."""
    import sys
    import types

    fake_pkg = types.ModuleType("tmd.plotters")

    def _raise():  # type: ignore[unused-ignore]
        raise ImportError("no plotters")

    fake_pkg.get_registered_plotters = _raise
    monkeypatch.setitem(sys.modules, "tmd.plotters", fake_pkg)

    fake_specs = {"matplotlib": True, "plotly": True, "seaborn": False, "polyscope": False}

    def _fake_find_spec(name: str):
        return object() if fake_specs[name] else None

    monkeypatch.setattr(visualize_mod.importlib.util, "find_spec", _fake_find_spec)
    out = visualize_mod.get_available_plotters()
    assert out == fake_specs


def test_select_plotter_returns_requested_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        visualize_mod, "get_available_plotters", lambda: {"matplotlib": True, "plotly": True}
    )
    assert visualize_mod.select_plotter("matplotlib") == "matplotlib"


def test_select_plotter_falls_back_to_alternative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        visualize_mod, "get_available_plotters", lambda: {"matplotlib": False, "plotly": True}
    )
    assert visualize_mod.select_plotter("matplotlib") == "plotly"


def test_select_plotter_no_backends_returns_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        visualize_mod, "get_available_plotters", lambda: {"matplotlib": False, "plotly": False}
    )
    assert visualize_mod.select_plotter("matplotlib") == "matplotlib"


def _fake_data(height_map: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(height_map=lambda: height_map)


def test_visualize_tmd_file_load_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(visualize_mod, "load_tmd_file", lambda *a, **k: None)
    res = visualize_mod.visualize_tmd_file(tmp_path / "missing.tmd")
    assert res is False


def test_visualize_tmd_file_uses_create_visualization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        visualize_mod, "load_tmd_file", lambda *a, **k: _fake_data(np.zeros((4, 4)))
    )
    captured: Dict[str, Any] = {}

    def _fake_create(**kwargs):
        captured.update(kwargs)
        return True

    import tmd.cli.utils.visualization as vis_utils

    monkeypatch.setattr(vis_utils, "create_visualization", _fake_create)
    out = tmp_path / "out.png"
    res = visualize_mod.visualize_tmd_file(tmp_path / "fixture.tmd", output=out)
    assert res is True
    assert captured["mode"] == "2d"
    assert captured["output"] == out


def test_visualize_tmd_file_calls_auto_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        visualize_mod, "load_tmd_file", lambda *a, **k: _fake_data(np.zeros((4, 4)))
    )
    auto_called = {"called": False}

    def _fake_auto_open(_path):
        auto_called["called"] = True

    monkeypatch.setattr(visualize_mod, "auto_open_file", _fake_auto_open)

    import tmd.cli.utils.visualization as vis_utils

    monkeypatch.setattr(vis_utils, "create_visualization", lambda **kwargs: True)

    out = tmp_path / "out.png"
    res = visualize_mod.visualize_tmd_file(
        tmp_path / "fixture.tmd", output=out, auto_open=True, plotter="matplotlib"
    )
    assert res is True
    assert auto_called["called"] is True


def test_visualize_tmd_file_polyscope_skips_auto_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        visualize_mod, "load_tmd_file", lambda *a, **k: _fake_data(np.zeros((4, 4)))
    )
    auto_called = {"called": False}

    monkeypatch.setattr(
        visualize_mod, "auto_open_file", lambda _p: auto_called.__setitem__("called", True)
    )

    import tmd.cli.utils.visualization as vis_utils

    monkeypatch.setattr(vis_utils, "create_visualization", lambda **kwargs: True)

    out = tmp_path / "out.png"
    res = visualize_mod.visualize_tmd_file(
        tmp_path / "fixture.tmd", output=out, auto_open=True, plotter="polyscope"
    )
    assert res is True
    assert auto_called["called"] is False


def test_check_available_visualization_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        visualize_mod,
        "get_available_plotters",
        lambda: {"matplotlib": True, "plotly": False, "seaborn": False, "polyscope": False},
    )
    out = visualize_mod.check_available_visualization_backends()
    assert out["matplotlib"] is True
    assert out["plotly"] is False


def test_check_available_visualization_backends_no_3d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the matplotlib branch when mpl_toolkits is missing."""
    monkeypatch.setattr(
        visualize_mod,
        "get_available_plotters",
        lambda: {"matplotlib": True, "plotly": False, "seaborn": False, "polyscope": False},
    )

    real_import = __import__

    def _failing_import(name: str, *args, **kwargs):
        if name == "mpl_toolkits.mplot3d":
            raise ImportError("no 3d")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _failing_import)
    out = visualize_mod.check_available_visualization_backends()
    assert out["matplotlib"] is True
