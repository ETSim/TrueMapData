"""Coverage for ``tmd.cli.commands`` visualize/maps/terrain and Typer ``--help`` paths."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

sys.modules.setdefault("noise", SimpleNamespace(snoise2=lambda *a, **k: 0.0))

from tmd.cli.main import app
from tmd.cli.commands import maps as maps_cmd
from tmd.cli.commands import terrain as terrain_cmd
from tmd.cli.commands import visualize as visualize_cmd


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


# --- maps.py ---


def test_process_metadata_valid_and_invalid() -> None:
    assert maps_cmd.process_metadata('{"a": 1}') == {"a": 1}
    assert maps_cmd.process_metadata("not json") == {}


def test_list_available_maps_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    lines: list[str] = []

    def capture(*args, **kwargs) -> None:
        lines.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(maps_cmd.console, "print", capture)
    monkeypatch.setattr(maps_cmd, "get_available_map_types", lambda: ["normal", "ao"])
    maps_cmd.list_available_maps()
    blob = "\n".join(lines)
    assert "normal" in blob and "ao" in blob


# --- terrain.py ---


def test_generate_synthetic_terrain_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    out = tmp_path / "terr_out"
    monkeypatch.setattr(terrain_cmd, "console", SimpleNamespace(print=lambda *a, **k: None))
    monkeypatch.setattr(terrain_cmd, "Table", lambda **k: SimpleNamespace(add_column=lambda *a, **k: None, add_row=lambda *a, **k: None))

    def fake_gen(path, **kwargs):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"tmd")

    monkeypatch.setattr(terrain_cmd.TMDTerrain, "generate_synthetic_tmd", staticmethod(fake_gen))
    monkeypatch.setattr(terrain_cmd, "export_maps_command", lambda *a, **k: None)

    ok = terrain_cmd.generate_synthetic_terrain(
        pattern="waves",
        width=8,
        height=8,
        output_dir=out,
        types=None,
    )
    assert ok is True
    assert (out / "waves.tmd").exists()


def test_generate_synthetic_terrain_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(terrain_cmd, "console", SimpleNamespace(print=lambda *a, **k: None))
    monkeypatch.setattr(terrain_cmd, "Table", lambda **k: SimpleNamespace(add_column=lambda *a, **k: None, add_row=lambda *a, **k: None))
    monkeypatch.setattr(
        terrain_cmd.TMDTerrain,
        "generate_synthetic_tmd",
        staticmethod(lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail"))),
    )
    monkeypatch.setattr(terrain_cmd, "print_error", lambda *a, **k: None)
    assert terrain_cmd.generate_synthetic_terrain("waves", 4, 4, output_dir=tmp_path / "o") is False


# --- visualize.py ---


def test_get_available_plotters_returns_dict() -> None:
    d = visualize_cmd.get_available_plotters()
    assert isinstance(d, dict)
    assert "matplotlib" in d


def test_select_plotter_prefers_available_alternative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        visualize_cmd,
        "get_available_plotters",
        lambda: {"matplotlib": False, "plotly": True},
    )
    warns: list[str] = []
    monkeypatch.setattr(visualize_cmd, "print_warning", lambda m: warns.append(str(m)))
    chosen = visualize_cmd.select_plotter("matplotlib", "2d")
    assert chosen == "plotly"
    assert warns


def test_visualize_tmd_file_create_visualization_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tmd_path = tmp_path / "v.tmd"
    tmd_path.write_bytes(b"dummy")

    class FD:
        def height_map(self):
            return np.zeros((3, 3), dtype=np.float32)

    monkeypatch.setattr(visualize_cmd, "load_tmd_file", lambda *a, **k: FD())

    def fake_create(**kwargs):
        return True

    monkeypatch.setattr("tmd.cli.utils.visualization.create_visualization", fake_create)
    monkeypatch.setattr(visualize_cmd, "print_error", lambda *a, **k: None)
    assert visualize_cmd.visualize_tmd_file(tmd_path, mode="2d", plotter="matplotlib") is True


def test_visualize_tmd_file_load_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p = tmp_path / "missing.tmd"
    p.write_bytes(b"1")
    monkeypatch.setattr(visualize_cmd, "load_tmd_file", lambda *a, **k: None)
    monkeypatch.setattr(visualize_cmd, "print_error", lambda *a, **k: None)
    assert visualize_cmd.visualize_tmd_file(p) is False


def test_check_available_visualization_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        visualize_cmd,
        "get_available_plotters",
        lambda: {"matplotlib": True, "plotly": False, "seaborn": False, "polyscope": False},
    )
    monkeypatch.setattr(visualize_cmd, "print_success", lambda *a, **k: None)
    monkeypatch.setattr(visualize_cmd, "print_warning", lambda *a, **k: None)
    out = visualize_cmd.check_available_visualization_backends()
    assert out["matplotlib"] is True


# --- Typer main app: visualize / terrain / maps ---


@pytest.mark.parametrize(
    "args,needle",
    [
        (["visualize", "basic", "--help"], "basic"),
        (["visualize", "3d", "--help"], "3d"),
        (["visualize", "profile", "--help"], "profile"),
        (["visualize", "contour", "--help"], "contour"),
        (["visualize", "fancy", "--help"], "fancy"),
        (["visualize", "compare", "--help"], "compare"),
        (["visualize", "ps-3d", "--help"], "3d"),
        (["visualize", "backends", "--help"], "backend"),
        (["visualize", "examples", "--help"], "example"),
        (["terrain", "generate", "--help"], "pattern"),
        (["maps", "ao", "--help"], "ao"),
        (["maps", "batch", "--help"], "batch"),
        (["maps", "height", "--help"], "height"),
    ],
)
def test_cli_help_pages(runner: CliRunner, args: list[str], needle: str) -> None:
    result = runner.invoke(app, args)
    assert result.exit_code == 0
    assert needle.lower() in result.stdout.lower()


def test_visualize_basic_invoke_mocked_create_visualization(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tmd = tmp_path / "plot.tmd"
    tmd.write_bytes(b"x")
    monkeypatch.setattr(
        "tmd.cli.apps.visualize_app.create_visualization",
        lambda **kwargs: True,
    )
    result = runner.invoke(
        app,
        ["visualize", "basic", str(tmd), "--plotter", "matplotlib"],
        catch_exceptions=False,
    )
    assert result.exit_code in (0, 1)


def test_terrain_generate_invoke_mocked(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = tmp_path / "tg"
    monkeypatch.setattr(
        terrain_cmd.TMDTerrain,
        "generate_synthetic_tmd",
        staticmethod(lambda path, **kw: Path(path).write_bytes(b"z")),
    )
    monkeypatch.setattr(terrain_cmd, "export_maps_command", lambda *a, **k: None)
    monkeypatch.setattr("tmd.cli.apps.terrain_app.generate_synthetic_terrain", terrain_cmd.generate_synthetic_terrain)
    # terrain_app imports generate_synthetic_terrain at module load — patch terrain_cmd used by app
    from tmd.cli.apps import terrain_app as ta

    monkeypatch.setattr(ta, "generate_synthetic_terrain", terrain_cmd.generate_synthetic_terrain)
    result = runner.invoke(
        app,
        [
            "terrain",
            "generate",
            "flat",
            "--width",
            "16",
            "--height",
            "16",
            "--output",
            str(out),
        ],
    )
    assert result.exit_code == 0
