"""Extra coverage for tmd.cli.utils.visualization."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tmd.cli.utils import visualization as cli_viz


class _FakePlotter:
    NAME = "fake"

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.saved: list[str] = []

    def plot(self, height_map: np.ndarray, **kwargs):
        self.calls.append({"height_map_shape": height_map.shape, **kwargs})
        return SimpleNamespace(_payload="figure")

    def save(self, fig, path: str):
        self.saved.append(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"fake")
        return path


@pytest.fixture
def fake_plotter(monkeypatch: pytest.MonkeyPatch) -> _FakePlotter:
    fp = _FakePlotter()
    monkeypatch.setattr(cli_viz.TMDPlotterFactory, "create_plotter", lambda *_a, **_k: fp)
    return fp


@pytest.fixture
def fake_io_ui(monkeypatch: pytest.MonkeyPatch):
    """Replace lazy io/ui modules with simple recording stand-ins."""

    class _Ui:
        def __init__(self) -> None:
            self.warnings: list[str] = []
            self.errors: list[str] = []
            self.successes: list[str] = []

        def print_warning(self, msg: str) -> None:
            self.warnings.append(msg)

        def print_error(self, msg: str) -> None:
            self.errors.append(msg)

        def print_success(self, msg: str) -> None:
            self.successes.append(msg)

        class _Console:
            def status(self, *_a, **_k):
                class _Ctx:
                    def __enter__(self_inner):
                        return self_inner

                    def __exit__(self_inner, *_a):
                        return False

                return _Ctx()

        console = _Console()

    class _Io:
        def __init__(self) -> None:
            self.opened: list[Path] = []

        def get_output_filename(self, file_path: Path, plotter: str, **_kwargs) -> Path:
            return file_path.with_suffix(".png")

        def load_tmd_file(self, path, **_kwargs):  # pragma: no cover - safety net
            return SimpleNamespace(height_map=np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4))

        def auto_open_file(self, path: Path) -> None:
            self.opened.append(path)

    ui = _Ui()
    io_mod = _Io()
    monkeypatch.setattr(cli_viz, "_get_ui_module", lambda: ui)
    monkeypatch.setattr(cli_viz, "_get_io_module", lambda: io_mod)
    return ui, io_mod


def test_check_available_visualization_backends_returns_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli_viz.importlib.util,
        "find_spec",
        lambda name: object() if name == "matplotlib" else None,
    )

    class _Ui:
        def print_success(self, *_a) -> None: ...
        def print_warning(self, *_a) -> None: ...

    monkeypatch.setattr(cli_viz, "_get_ui_module", lambda: _Ui())
    backends = cli_viz.check_available_visualization_backends()
    assert backends == {"matplotlib": True, "plotly": False, "seaborn": False, "polyscope": False}


def test_get_height_map_property_and_method() -> None:
    arr = np.zeros((2, 2), dtype=np.float32)
    obj_property = SimpleNamespace(height_map=arr)
    assert cli_viz._get_height_map(obj_property) is arr

    class _WithMethod:
        def height_map(self) -> np.ndarray:
            return arr

    assert cli_viz._get_height_map(_WithMethod()) is arr


def test_get_height_map_handles_none(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []

    class _Ui:
        def print_error(self, msg: str) -> None:
            errors.append(msg)

    monkeypatch.setattr(cli_viz, "_get_ui_module", lambda: _Ui())
    assert cli_viz._get_height_map(None) is None
    assert errors


def test_get_height_map_missing_attr(monkeypatch: pytest.MonkeyPatch) -> None:
    errors: list[str] = []

    class _Ui:
        def print_error(self, msg: str) -> None:
            errors.append(msg)

    monkeypatch.setattr(cli_viz, "_get_ui_module", lambda: _Ui())
    assert cli_viz._get_height_map(SimpleNamespace()) is None
    assert errors


def test_create_visualization_2d_with_fake_plotter(
    tmp_path: Path,
    small_heightmap: np.ndarray,
    fake_plotter: _FakePlotter,
    fake_io_ui,
) -> None:
    ui, _io = fake_io_ui
    tmd_obj = SimpleNamespace(height_map=small_heightmap)
    output = tmp_path / "out.png"
    ok = cli_viz.create_visualization(
        tmd_file_or_data=tmd_obj,
        mode="2d",
        plotter="matplotlib",
        output=output,
    )
    assert ok is True
    assert fake_plotter.calls
    assert fake_plotter.saved
    assert "Visualization saved" in " ".join(ui.successes)


def test_create_visualization_3d_passes_z_scale(
    tmp_path: Path,
    small_heightmap: np.ndarray,
    fake_plotter: _FakePlotter,
    fake_io_ui,
) -> None:
    tmd_obj = SimpleNamespace(height_map=small_heightmap)
    output = tmp_path / "scene.png"
    ok = cli_viz.create_visualization(
        tmd_file_or_data=tmd_obj,
        mode="3d",
        plotter="plotly",
        output=output,
        z_scale=2.0,
    )
    assert ok is True
    assert fake_plotter.calls[0]["z_scale"] == 2.0


def test_create_visualization_profile_invalid_row_returns_false(
    tmp_path: Path,
    small_heightmap: np.ndarray,
    fake_plotter: _FakePlotter,
    fake_io_ui,
) -> None:
    tmd_obj = SimpleNamespace(height_map=small_heightmap)
    output = tmp_path / "p.png"
    ok = cli_viz.create_visualization(
        tmd_file_or_data=tmd_obj,
        mode="profile",
        plotter="matplotlib",
        output=output,
        profile_row=100,
    )
    assert ok is False


def test_create_visualization_seaborn_profile_handled_specially(
    tmp_path: Path,
    small_heightmap: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    fake_io_ui,
) -> None:
    """``_handle_seaborn_profile`` should short-circuit ``create_visualization``."""

    sentinel = {"called": False}

    def _fake_handler(*_a, **_k):
        sentinel["called"] = True
        return True

    monkeypatch.setattr(cli_viz, "_handle_seaborn_profile", _fake_handler)
    monkeypatch.setattr(
        cli_viz.TMDPlotterFactory,
        "create_plotter",
        lambda *_a, **_k: pytest.fail("plotter should not be created"),
    )

    tmd_obj = SimpleNamespace(height_map=small_heightmap)
    ok = cli_viz.create_visualization(
        tmd_file_or_data=tmd_obj,
        mode="profile",
        plotter="seaborn",
        output=tmp_path / "p.png",
        profile_row=0,
    )
    assert ok is True
    assert sentinel["called"]


def test_create_visualization_factory_failure_uses_outer_fallback(
    tmp_path: Path,
    small_heightmap: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    fake_io_ui,
) -> None:
    """When primary plotter fails, the outer ``_try_fallback_visualization`` rescues the call."""

    def _raise(*_a, **_k):
        raise ImportError("primary not installed")

    monkeypatch.setattr(cli_viz.TMDPlotterFactory, "create_plotter", _raise)

    fallback_calls: list[dict] = []

    def _fake_fallback(*_args, **kwargs):
        fallback_calls.append(kwargs)
        return True

    monkeypatch.setattr(cli_viz, "_try_fallback_visualization", _fake_fallback)

    tmd_obj = SimpleNamespace(height_map=small_heightmap)
    ok = cli_viz.create_visualization(
        tmd_file_or_data=tmd_obj,
        mode="2d",
        plotter="matplotlib",
        output=tmp_path / "out.png",
    )
    assert ok is True
    assert fallback_calls


def test_create_visualization_polyscope_no_fallback_returns_false(
    tmp_path: Path,
    small_heightmap: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    fake_io_ui,
) -> None:
    def _raise(*_a, **_k):
        raise ImportError("polyscope missing")

    monkeypatch.setattr(cli_viz.TMDPlotterFactory, "create_plotter", _raise)
    tmd_obj = SimpleNamespace(height_map=small_heightmap)
    ok = cli_viz.create_visualization(
        tmd_file_or_data=tmd_obj,
        mode="3d",
        plotter="polyscope",
        output=tmp_path / "out.png",
        use_fallback=False,
    )
    assert ok is False


def test_prepare_data_and_title_from_tmd_object(small_heightmap: np.ndarray) -> None:
    tmd_obj = SimpleNamespace(height_map=small_heightmap)
    hm, fp, title = cli_viz._prepare_data_and_title(
        tmd_obj, "2d", None, True, io_module=None, ui_module=None
    )
    assert hm is small_heightmap
    assert fp is None
    assert title is not None
    assert "2D" in title


def test_prepare_data_and_title_explicit_title(small_heightmap: np.ndarray) -> None:
    tmd_obj = SimpleNamespace(height_map=small_heightmap)
    _, _, title = cli_viz._prepare_data_and_title(
        tmd_obj, "3d", "My Custom Title", True, io_module=None, ui_module=None
    )
    assert title == "My Custom Title"


def test_prepare_data_and_title_from_path(
    tmp_path: Path, small_heightmap: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Io:
        def load_tmd_file(self, path, **_k):
            return SimpleNamespace(height_map=small_heightmap)

    class _Ui:
        def print_error(self, *_a) -> None: ...

    fp = tmp_path / "fixture.tmd"
    fp.write_bytes(b"")
    hm, file_path, title = cli_viz._prepare_data_and_title(
        fp, "profile", None, True, io_module=_Io(), ui_module=_Ui()
    )
    assert hm is small_heightmap
    assert file_path == fp
    assert "fixture" in title


def test_prepare_data_and_title_load_failure_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Io:
        def load_tmd_file(self, path, **_k):
            return None

    fp = tmp_path / "missing.tmd"
    fp.write_bytes(b"")
    hm, _fp, _title = cli_viz._prepare_data_and_title(
        fp, "2d", None, True, io_module=_Io(), ui_module=SimpleNamespace(print_error=lambda *_a: None)
    )
    assert hm is None


def test_handle_seaborn_profile_invalid_row(small_heightmap: np.ndarray, fake_io_ui) -> None:
    ui, io_mod = fake_io_ui
    out = cli_viz._handle_seaborn_profile(
        small_heightmap, profile_row=100, title=None, output=None, kwargs={}, ui_module=ui, io_module=io_mod, auto_open=False
    )
    assert out is False


def test_handle_seaborn_profile_falls_back_when_seaborn_missing(
    small_heightmap: np.ndarray,
    fake_io_ui,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ui, io_mod = fake_io_ui

    import sys

    monkeypatch.setitem(sys.modules, "tmd.plotters.seaborn", None)
    out = cli_viz._handle_seaborn_profile(
        small_heightmap, profile_row=0, title="t", output=None, kwargs={}, ui_module=ui, io_module=io_mod, auto_open=False
    )
    assert out is None


def test_try_fallback_visualization_2d(tmp_path: Path, small_heightmap: np.ndarray, fake_io_ui) -> None:
    out = tmp_path / "fb.png"
    ok = cli_viz._try_fallback_visualization(
        small_heightmap,
        "2d",
        "matplotlib",
        out,
        None,
        None,
        "viridis",
        1.0,
        False,
        False,
    )
    assert ok is True
    assert out.exists()


def test_try_fallback_visualization_3d(tmp_path: Path, small_heightmap: np.ndarray, fake_io_ui) -> None:
    out = tmp_path / "fb3d.png"
    ok = cli_viz._try_fallback_visualization(
        small_heightmap,
        "3d",
        "matplotlib",
        out,
        None,
        "Title",
        "viridis",
        1.5,
        True,
        False,
    )
    assert ok is True


def test_try_fallback_visualization_profile(tmp_path: Path, small_heightmap: np.ndarray, fake_io_ui) -> None:
    out = tmp_path / "fbprof.png"
    ok = cli_viz._try_fallback_visualization(
        small_heightmap,
        "profile",
        "matplotlib",
        out,
        None,
        None,
        "viridis",
        1.0,
        False,
        False,
    )
    assert ok is True


def test_get_utils_and_mesh_converter_module() -> None:
    """Cover the lazy import helpers."""
    cli_viz._utils_module = None
    cli_viz._mesh_converter_module = None
    utils_mod = cli_viz._get_utils_module()
    assert utils_mod is not None
    again = cli_viz._get_utils_module()
    assert again is utils_mod

    try:
        mc = cli_viz._get_mesh_converter_module()
        assert mc is cli_viz._get_mesh_converter_module()
    except ImportError:
        pass


def test_create_visualization_save_failure_reports_error(
    tmp_path: Path,
    small_heightmap: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
    fake_io_ui,
) -> None:
    class _BadSavePlotter:
        NAME = "fake"

        def plot(self, height_map, **_kwargs):
            return SimpleNamespace(_payload="figure")

        def save(self, _fig, _path):
            return None

    monkeypatch.setattr(
        cli_viz.TMDPlotterFactory, "create_plotter", lambda *_a, **_k: _BadSavePlotter()
    )

    tmd_obj = SimpleNamespace(height_map=small_heightmap)
    ok = cli_viz.create_visualization(
        tmd_file_or_data=tmd_obj,
        mode="2d",
        plotter="matplotlib",
        output=tmp_path / "out.png",
    )
    assert ok is False


def test_create_visualization_calls_auto_open(
    tmp_path: Path,
    small_heightmap: np.ndarray,
    fake_plotter: _FakePlotter,
    fake_io_ui,
) -> None:
    ui, io_mod = fake_io_ui
    output = tmp_path / "ao.png"
    ok = cli_viz.create_visualization(
        tmd_file_or_data=SimpleNamespace(height_map=small_heightmap),
        mode="2d",
        plotter="matplotlib",
        output=output,
        auto_open=True,
    )
    assert ok is True
    assert output in io_mod.opened


def test_create_visualization_polyscope_does_not_auto_open(
    tmp_path: Path,
    small_heightmap: np.ndarray,
    fake_plotter: _FakePlotter,
    fake_io_ui,
) -> None:
    ui, io_mod = fake_io_ui
    output = tmp_path / "ps.png"
    ok = cli_viz.create_visualization(
        tmd_file_or_data=SimpleNamespace(height_map=small_heightmap),
        mode="2d",
        plotter="polyscope",
        output=output,
        auto_open=True,
    )
    assert ok is True
    assert output not in io_mod.opened


def test_prepare_data_and_title_comparison_loads_second_file(
    tmp_path: Path, small_heightmap: np.ndarray
) -> None:
    fp = tmp_path / "first.tmd"
    fp.write_bytes(b"")
    second_fp = tmp_path / "second.tmd"
    second_fp.write_bytes(b"")

    loaded: list[Path] = []

    class _Io:
        def load_tmd_file(self, path, **_k):
            loaded.append(Path(path))
            return SimpleNamespace(height_map=small_heightmap)

    class _Ui:
        def print_error(self, *_a) -> None: ...

    hm, file_path, title = cli_viz._prepare_data_and_title(
        fp,
        "comparison",
        None,
        True,
        io_module=_Io(),
        ui_module=_Ui(),
        second_file=second_fp,
    )
    assert hm is small_heightmap
    assert file_path == fp
    assert second_fp in loaded


def test_handle_seaborn_profile_happy_path_saves_figure(
    tmp_path: Path,
    small_heightmap: np.ndarray,
    fake_io_ui,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cover the success branch of _handle_seaborn_profile."""
    ui, io_mod = fake_io_ui

    class _FakeFig:
        def savefig(self, *args, **kwargs):
            Path(args[0]).parent.mkdir(parents=True, exist_ok=True)
            Path(args[0]).write_bytes(b"img")

    class _FakeProfilePlotter:
        def plot_profile(self, *args, **kwargs):
            return _FakeFig()

    fake_pkg = SimpleNamespace(SeabornProfilePlotter=_FakeProfilePlotter)
    import sys

    monkeypatch.setitem(sys.modules, "tmd.plotters.seaborn", fake_pkg)

    out = tmp_path / "sea.png"
    ok = cli_viz._handle_seaborn_profile(
        small_heightmap,
        profile_row=0,
        title=None,
        output=out,
        kwargs={},
        ui_module=ui,
        io_module=io_mod,
        auto_open=True,
    )
    assert ok is True
    assert out.exists()
    assert out in io_mod.opened


def test_try_fallback_visualization_handles_failure(
    monkeypatch: pytest.MonkeyPatch, fake_io_ui
) -> None:
    """Force matplotlib import to fail via plt.subplots raising."""
    import matplotlib.pyplot as plt

    def _boom(*_a, **_k):
        raise RuntimeError("simulated plot failure")

    monkeypatch.setattr(plt, "subplots", _boom)

    ok = cli_viz._try_fallback_visualization(
        np.zeros((2, 2), dtype=np.float32),
        "2d",
        "matplotlib",
        None,
        None,
        None,
        "viridis",
        1.0,
        False,
        False,
    )
    assert ok is False
