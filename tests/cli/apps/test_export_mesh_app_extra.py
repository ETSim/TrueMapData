"""Extra coverage for the mesh export Typer app."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from tmd.cli.apps import export_mesh_app as mesh_app_mod


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


@pytest.fixture
def fake_export_model(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace ``export_model`` with a sentinel that records kwargs."""
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        out = kwargs.get("output_file")
        if out is not None:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_bytes(b"fake mesh")
        return True

    monkeypatch.setattr(mesh_app_mod, "export_model", _fake)
    return calls


@pytest.fixture
def fake_batch(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        return True

    monkeypatch.setattr(mesh_app_mod, "batch_export_models", _fake)
    return calls


@pytest.fixture
def fake_apply(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    calls: list[dict] = []

    def _fake(**kwargs):
        calls.append(kwargs)
        out_root = Path(kwargs["output_root"])
        out_root.mkdir(parents=True, exist_ok=True)
        return {
            "obj": str(out_root / "applied.obj"),
            "mtl": str(out_root / "applied.mtl"),
            "textures_dir": str(out_root / "textures"),
            "tile_size_px": 256,
            "target_size_px": 1024,
        }

    monkeypatch.setattr(mesh_app_mod, "apply_maps_to_mesh", _fake)
    return calls


def _app():
    return mesh_app_mod.create_export_mesh_app()


def test_list_legacy_lists_known_formats(runner: CliRunner) -> None:
    r = runner.invoke(_app(), ["list-legacy"])
    assert r.exit_code == 0


def test_export_stl_legacy_smoke(runner: CliRunner, tmp_tmd_path: Path) -> None:
    r = runner.invoke(_app(), ["stl-legacy", str(tmp_tmd_path)])
    assert r.exit_code == 0


def test_generate_command_default_output(runner: CliRunner, tmp_tmd_path: Path, fake_export_model: list[dict]) -> None:
    r = runner.invoke(
        _app(),
        ["generate", str(tmp_tmd_path), "--format", "stl"],
    )
    assert r.exit_code == 0
    assert fake_export_model
    assert fake_export_model[0]["format"] == "stl"
    assert fake_export_model[0]["output_file"].suffix == ".stl"


def test_export_stl_command(
    runner: CliRunner, tmp_path: Path, tmp_tmd_path: Path, fake_export_model: list[dict]
) -> None:
    out = tmp_path / "out.stl"
    r = runner.invoke(
        _app(),
        ["stl", str(tmp_tmd_path), "--output-file", str(out), "--max-triangles", "200", "--error-threshold", "0.1"],
    )
    assert r.exit_code == 0
    assert fake_export_model[-1]["format"] == "stl"
    assert Path(fake_export_model[-1]["output_file"]) == out


def test_export_obj_command(
    runner: CliRunner, tmp_path: Path, tmp_tmd_path: Path, fake_export_model: list[dict]
) -> None:
    out = tmp_path / "out.obj"
    r = runner.invoke(
        _app(),
        ["obj", str(tmp_tmd_path), "--output-file", str(out), "--bind-material-maps"],
    )
    assert r.exit_code == 0
    assert fake_export_model[-1]["format"] == "obj"
    assert fake_export_model[-1]["bind_material_maps"] is True


def test_export_ply_command(runner: CliRunner, tmp_tmd_path: Path, fake_export_model: list[dict]) -> None:
    r = runner.invoke(_app(), ["ply", str(tmp_tmd_path)])
    assert r.exit_code == 0
    assert fake_export_model[-1]["format"] == "ply"
    assert fake_export_model[-1]["output_file"].suffix == ".ply"


def test_export_gltf_command_glb_default(runner: CliRunner, tmp_tmd_path: Path, fake_export_model: list[dict]) -> None:
    r = runner.invoke(_app(), ["gltf", str(tmp_tmd_path)])
    assert r.exit_code == 0
    assert fake_export_model[-1]["format"] == "gltf"
    assert fake_export_model[-1]["output_file"].suffix == ".glb"


def test_export_gltf_command_text(
    runner: CliRunner, tmp_path: Path, tmp_tmd_path: Path, fake_export_model: list[dict]
) -> None:
    out = tmp_path / "scene.gltf"
    r = runner.invoke(_app(), ["gltf", str(tmp_tmd_path), "--no-binary", "--output-file", str(out)])
    assert r.exit_code == 0
    assert fake_export_model[-1]["binary"] is False


def test_export_usd_command(runner: CliRunner, tmp_tmd_path: Path, fake_export_model: list[dict]) -> None:
    r = runner.invoke(_app(), ["usd", str(tmp_tmd_path), "--max-triangles", "150"])
    assert r.exit_code == 0
    assert fake_export_model[-1]["format"] == "usd"


def test_batch_command_success(
    runner: CliRunner, tmp_path: Path, tmp_tmd_path: Path, fake_batch: list[dict]
) -> None:
    in_dir = tmp_path / "inputs"
    in_dir.mkdir()
    (in_dir / "fixture.tmd").write_bytes(tmp_tmd_path.read_bytes())
    out_dir = tmp_path / "out"
    r = runner.invoke(
        _app(),
        [
            "batch",
            str(in_dir),
            "--output-dir",
            str(out_dir),
            "--format",
            "stl",
            "--max-triangles",
            "100",
            "--error-threshold",
            "0.05",
            "--no-binary",
        ],
    )
    assert r.exit_code == 0
    assert fake_batch
    kwargs = fake_batch[0]
    assert kwargs["max_triangles"] == 100
    assert kwargs["binary"] is False
    assert kwargs["coordinate_system"] == "right-handed"


def test_apply_command_success(
    runner: CliRunner, tmp_path: Path, tmp_tmd_path: Path, fake_apply: list[dict]
) -> None:
    out_dir = tmp_path / "bundle"
    r = runner.invoke(
        _app(),
        [
            "apply",
            str(tmp_tmd_path),
            "--output-dir",
            str(out_dir),
            "--template-kind",
            "plane",
            "--max-texture-edge",
            "0",
        ],
    )
    assert r.exit_code == 0
    assert fake_apply
    kwargs = fake_apply[0]
    assert kwargs["template_kind"] == "plane"
    assert kwargs["max_texture_edge"] is None


def test_formats_lists(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(mesh_app_mod, "list_model_formats", lambda: called.append(True))
    r = runner.invoke(_app(), ["formats"])
    assert r.exit_code == 0
    assert called
