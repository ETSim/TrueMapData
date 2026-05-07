"""Extra coverage for the visualize CLI app."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tmd.cli.apps import visualize_app as viz_mod


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


@pytest.fixture
def fake_create_viz(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(viz_mod, "create_visualization", _fake)
    return calls


@pytest.fixture
def fake_polyscope_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force ``_check_polyscope_available`` to return True without touching imports."""
    monkeypatch.setattr(viz_mod, "_check_polyscope_available", lambda: True)


def _app():
    return viz_mod.create_visualize_app()


def test_visualize_basic_invokes_create_visualization(
    runner: CliRunner, tmp_tmd_path: Path, fake_create_viz: list[dict]
) -> None:
    r = runner.invoke(_app(), ["basic", str(tmp_tmd_path), "--plotter", "matplotlib"])
    assert r.exit_code == 0
    assert fake_create_viz[-1]["mode"] == "2d"
    assert fake_create_viz[-1]["plotter"] == "matplotlib"


def test_visualize_3d_passes_z_scale(
    runner: CliRunner, tmp_tmd_path: Path, fake_create_viz: list[dict]
) -> None:
    r = runner.invoke(_app(), ["3d", str(tmp_tmd_path), "--plotter", "plotly", "--z-scale", "2.5"])
    assert r.exit_code == 0
    assert fake_create_viz[-1]["mode"] == "3d"
    assert fake_create_viz[-1]["z_scale"] == 2.5


def test_visualize_profile_with_seaborn_extras(
    runner: CliRunner, tmp_tmd_path: Path, fake_create_viz: list[dict]
) -> None:
    r = runner.invoke(
        _app(),
        ["profile", str(tmp_tmd_path), "--plotter", "seaborn", "--row", "2", "--marker-size", "8"],
    )
    assert r.exit_code == 0
    assert fake_create_viz[-1]["mode"] == "profile"
    assert fake_create_viz[-1]["profile_row"] == 2
    assert fake_create_viz[-1]["marker_size"] == 8


def test_visualize_contour(runner: CliRunner, tmp_tmd_path: Path, fake_create_viz: list[dict]) -> None:
    r = runner.invoke(_app(), ["contour", str(tmp_tmd_path), "--plotter", "matplotlib", "--levels", "5"])
    assert r.exit_code == 0
    assert fake_create_viz[-1]["mode"] == "contour"
    assert fake_create_viz[-1]["levels"] == 5


def test_visualize_enhanced_distribution_mode(
    runner: CliRunner, tmp_tmd_path: Path, fake_create_viz: list[dict]
) -> None:
    r = runner.invoke(
        _app(), ["fancy", str(tmp_tmd_path), "--plotter", "seaborn", "--mode", "distribution"]
    )
    assert r.exit_code == 0
    last = fake_create_viz[-1]
    assert last["mode"] == "enhanced"
    assert last["viz_type"] == "distribution"
    assert last["kde"] is True


def test_visualize_enhanced_joint_mode(
    runner: CliRunner, tmp_tmd_path: Path, fake_create_viz: list[dict]
) -> None:
    r = runner.invoke(_app(), ["fancy", str(tmp_tmd_path), "--plotter", "plotly", "--mode", "joint"])
    assert r.exit_code == 0
    last = fake_create_viz[-1]
    assert last["viz_type"] == "joint"


def test_visualize_enhanced_default_with_warning(
    runner: CliRunner, tmp_tmd_path: Path, fake_create_viz: list[dict]
) -> None:
    r = runner.invoke(_app(), ["fancy", str(tmp_tmd_path), "--plotter", "matplotlib"])
    assert r.exit_code == 0


def test_visualize_comparison_single_file(
    runner: CliRunner, tmp_tmd_path: Path, fake_create_viz: list[dict]
) -> None:
    r = runner.invoke(_app(), ["compare", str(tmp_tmd_path), "--plotter", "matplotlib"])
    assert r.exit_code == 0
    assert fake_create_viz[-1]["mode"] == "multi_profile"


def test_visualize_comparison_two_files(
    runner: CliRunner, tmp_path: Path, tmp_tmd_path: Path, fake_create_viz: list[dict]
) -> None:
    second = tmp_path / "second.tmd"
    second.write_bytes(tmp_tmd_path.read_bytes())
    r = runner.invoke(
        _app(),
        [
            "compare",
            str(tmp_tmd_path),
            "--second-file",
            str(second),
            "--plotter",
            "matplotlib",
        ],
    )
    assert r.exit_code == 0
    assert fake_create_viz[-1]["mode"] == "comparison"


def test_visualize_polyscope_3d_when_available(
    runner: CliRunner,
    tmp_tmd_path: Path,
    fake_create_viz: list[dict],
    fake_polyscope_available,
) -> None:
    r = runner.invoke(_app(), ["ps-3d", str(tmp_tmd_path), "--no-interactive"])
    assert r.exit_code == 0
    last = fake_create_viz[-1]
    assert last["plotter"] == "polyscope"
    assert last["mode"] == "3d"
    assert last["show"] is False


def test_visualize_polyscope_pointcloud_when_available(
    runner: CliRunner,
    tmp_tmd_path: Path,
    fake_create_viz: list[dict],
    fake_polyscope_available,
) -> None:
    r = runner.invoke(
        _app(),
        ["ps-pointcloud", str(tmp_tmd_path), "--point-size", "3.5", "--no-interactive"],
    )
    assert r.exit_code == 0
    last = fake_create_viz[-1]
    assert last["mode"] == "point_cloud"
    assert last["point_size"] == 3.5


def test_visualize_polyscope_mesh_when_available(
    runner: CliRunner,
    tmp_tmd_path: Path,
    fake_create_viz: list[dict],
    fake_polyscope_available,
) -> None:
    r = runner.invoke(_app(), ["ps-mesh", str(tmp_tmd_path), "--no-interactive"])
    assert r.exit_code == 0
    last = fake_create_viz[-1]
    assert last["mode"] == "mesh"


def test_visualize_polyscope_commands_when_unavailable(
    runner: CliRunner, tmp_tmd_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(viz_mod, "_check_polyscope_available", lambda: False)

    def _should_not_run(**_k):
        raise AssertionError("create_visualization called when polyscope unavailable")

    monkeypatch.setattr(viz_mod, "create_visualization", _should_not_run)

    for cmd in ("ps-3d", "ps-pointcloud", "ps-mesh"):
        r = runner.invoke(_app(), [cmd, str(tmp_tmd_path)])
        assert r.exception is None or isinstance(r.exception, SystemExit)


def test_list_backends_lists_known_names(runner: CliRunner) -> None:
    r = runner.invoke(_app(), ["backends"])
    assert r.exit_code == 0


def test_resolve_plotter_passthrough() -> None:
    assert viz_mod._resolve_plotter(viz_mod.PlotterChoice.MATPLOTLIB) == "matplotlib"
    assert viz_mod._resolve_plotter(viz_mod.PlotterChoice.PLOTLY) == "plotly"


def test_resolve_plotter_auto_with_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import tmd.plotters as plotters_pkg

    monkeypatch.setattr(plotters_pkg, "get_available_plotters", lambda: ["seaborn", "matplotlib"])
    assert viz_mod._resolve_plotter(viz_mod.PlotterChoice.AUTO) == "seaborn"


def test_resolve_plotter_auto_no_available(monkeypatch: pytest.MonkeyPatch) -> None:
    import tmd.plotters as plotters_pkg

    monkeypatch.setattr(plotters_pkg, "get_available_plotters", lambda: [])
    assert viz_mod._resolve_plotter(viz_mod.PlotterChoice.AUTO) == "matplotlib"


def test_check_polyscope_available_true(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util as _util

    monkeypatch.setattr(_util, "find_spec", lambda name: object())
    assert viz_mod._check_polyscope_available() is True


def test_check_polyscope_available_false(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util as _util

    monkeypatch.setattr(_util, "find_spec", lambda name: None)
    assert viz_mod._check_polyscope_available() is False
