"""Unit tests for mesh export CLI wiring."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from tmd.cli.main import app


def test_mesh_obj_forwards_bind_material_maps_flag(monkeypatch) -> None:
    captured: dict = {}

    def fake_export_model(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr("tmd.cli.apps.export_mesh_app.export_model", fake_export_model)

    runner = CliRunner(env={"TERM": "dumb"})
    result = runner.invoke(
        app,
        [
            "mesh",
            "obj",
            "input.tmd",
            "--bind-material-maps",
            "--output-file",
            "out.obj",
        ],
    )

    assert result.exit_code == 0, result.stdout or str(result.exception)
    assert captured["input_file"] == Path("input.tmd")
    assert captured["output_file"] == Path("out.obj")
    assert captured["format"] == "obj"
    assert captured["bind_material_maps"] is True


def test_mesh_obj_default_does_not_bind_material_maps(monkeypatch) -> None:
    captured: dict = {}

    def fake_export_model(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr("tmd.cli.apps.export_mesh_app.export_model", fake_export_model)

    runner = CliRunner(env={"TERM": "dumb"})
    result = runner.invoke(
        app,
        ["mesh", "obj", "input.tmd", "--output-file", "out.obj"],
    )

    assert result.exit_code == 0, result.stdout or str(result.exception)
    assert captured["bind_material_maps"] is False


def test_mesh_generate_forwards_bind_material_maps(monkeypatch) -> None:
    captured: dict = {}

    def fake_export_model(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr("tmd.cli.apps.export_mesh_app.export_model", fake_export_model)

    runner = CliRunner(env={"TERM": "dumb"})
    result = runner.invoke(
        app,
        [
            "mesh",
            "generate",
            "terrain.tmd",
            "--format",
            "obj",
            "--bind-material-maps",
            "--output-file",
            "gen.obj",
        ],
    )

    assert result.exit_code == 0, result.stdout or str(result.exception)
    assert captured["input_file"] == Path("terrain.tmd")
    assert captured["output_file"] == Path("gen.obj")
    assert captured["format"] == "obj"
    assert captured["bind_material_maps"] is True


def test_mesh_gltf_forwards_bind_material_maps(monkeypatch) -> None:
    captured: dict = {}

    def fake_export_model(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr("tmd.cli.apps.export_mesh_app.export_model", fake_export_model)

    runner = CliRunner(env={"TERM": "dumb"})
    result = runner.invoke(
        app,
        ["mesh", "gltf", "in.tmd", "--bind-material-maps", "--output-file", "out.glb"],
    )

    assert result.exit_code == 0, result.stdout or str(result.exception)
    assert captured["format"] == "gltf"
    assert captured["bind_material_maps"] is True


def test_mesh_apply_max_texture_edge_zero_passes_none(monkeypatch, tmp_path) -> None:
    captured: dict = {}

    def fake_apply_maps_to_mesh(**kwargs):
        captured.update(kwargs)
        return {
            "obj": tmp_path / "a.obj",
            "mtl": tmp_path / "a.mtl",
            "textures_dir": tmp_path / "tex",
            "tile_size_px": 1,
            "target_size_px": 2,
        }

    monkeypatch.setattr(
        "tmd.cli.apps.export_mesh_app.apply_maps_to_mesh",
        fake_apply_maps_to_mesh,
    )

    out_dir = tmp_path / "bundle"
    out_dir.mkdir()

    runner = CliRunner(env={"TERM": "dumb"})
    result = runner.invoke(
        app,
        [
            "mesh",
            "apply",
            "in.tmd",
            "-o",
            str(out_dir),
            "--max-texture-edge",
            "0",
        ],
    )

    assert result.exit_code == 0, result.stdout or str(result.exception)
    assert captured["max_texture_edge"] is None


def test_mesh_apply_default_max_texture_edge_is_8192(monkeypatch, tmp_path) -> None:
    captured: dict = {}

    def fake_apply_maps_to_mesh(**kwargs):
        captured.update(kwargs)
        return {
            "obj": tmp_path / "a.obj",
            "mtl": tmp_path / "a.mtl",
            "textures_dir": tmp_path / "tex",
            "tile_size_px": 1,
            "target_size_px": 2,
        }

    monkeypatch.setattr(
        "tmd.cli.apps.export_mesh_app.apply_maps_to_mesh",
        fake_apply_maps_to_mesh,
    )

    out_dir = tmp_path / "bundle"
    out_dir.mkdir()

    runner = CliRunner(env={"TERM": "dumb"})
    result = runner.invoke(
        app,
        ["mesh", "apply", "in.tmd", "-o", str(out_dir)],
    )

    assert result.exit_code == 0, result.stdout or str(result.exception)
    assert captured["max_texture_edge"] == 8192
