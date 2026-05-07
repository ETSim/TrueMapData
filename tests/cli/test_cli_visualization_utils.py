"""Tests for :mod:`tmd.cli.utils.visualization`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tmd.cli.utils import visualization as viz_mod


def test_check_available_visualization_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(viz_mod, "_get_ui_module", lambda: MagicMock(print_success=lambda *a, **k: None))
    out = viz_mod.check_available_visualization_backends()
    assert isinstance(out, dict)
    assert "matplotlib" in out


def test_get_height_map_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    ui = MagicMock()
    monkeypatch.setattr(viz_mod, "_get_ui_module", lambda: ui)
    assert viz_mod._get_height_map(None) is None

    m = MagicMock()
    m.height_map = np.ones((3, 3), dtype=np.float32)
    assert viz_mod._get_height_map(m).shape == (3, 3)

    m2 = MagicMock()

    def _hm():
        return np.full((2, 2), 2.0, dtype=np.float32)

    m2.height_map = _hm
    assert viz_mod._get_height_map(m2)[0, 0] == 2.0

    bare = MagicMock(spec=[])
    assert viz_mod._get_height_map(bare) is None


def test_prepare_data_and_title_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    io_mod = MagicMock()
    obj = MagicMock()
    obj.height_map = np.ones((4, 4), dtype=np.float32)
    io_mod.load_tmd_file.return_value = obj

    ui_mod = MagicMock()
    h, fp, title = viz_mod._prepare_data_and_title(
        Path(tmp_path / "x.tmd"),
        "2d",
        None,
        True,
        io_mod,
        ui_mod,
    )
    assert h.shape == (4, 4)
    assert title


def test_try_fallback_visualization_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ui = MagicMock()
    io = MagicMock()
    monkeypatch.setattr(viz_mod, "_get_ui_module", lambda: ui)
    monkeypatch.setattr(viz_mod, "_get_io_module", lambda: io)

    out = Path(tmp_path / "sub/out.png")
    assert viz_mod._try_fallback_visualization(
        np.ones((6, 6), dtype=np.float32),
        "2d",
        "matplotlib",
        out,
        None,
        "t",
        "viridis",
        1.0,
        False,
        False,
        auto_open=False,
    )

    assert viz_mod._try_fallback_visualization(
        np.ones((6, 6), dtype=np.float32),
        "profile",
        "matplotlib",
        None,
        2,
        None,
        "viridis",
        1.0,
        True,
        False,
    )

    assert viz_mod._try_fallback_visualization(
        np.ones((6, 6), dtype=np.float32),
        "3d",
        "matplotlib",
        tmp_path / "z.png",
        None,
        "t",
        "viridis",
        1.0,
        True,
        False,
        auto_open=False,
    )


def test_create_visualization_with_mock_plotter(tmp_path: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    plotter = MagicMock()
    plotter.NAME = "matplotlib"
    plotter.plot.return_value = MagicMock()
    plotter.save.return_value = "saved.png"

    monkeypatch.setattr(
        "tmd.cli.utils.visualization.TMDPlotterFactory.create_plotter",
        lambda name: plotter,
    )

    ui_mod = MagicMock()
    ui_mod.console.status.return_value.__enter__ = lambda s: None
    ui_mod.console.status.return_value.__exit__ = lambda s, *a: None
    monkeypatch.setattr(viz_mod, "_get_ui_module", lambda: ui_mod)

    arr = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)

    class _T:
        height_map = arr

    ok = viz_mod.create_visualization(
        _T(),
        "2d",
        "matplotlib",
        output=tmp_path / "out.png",
        title="x",
    )
    assert ok is True
    plotter.plot.assert_called_once()
