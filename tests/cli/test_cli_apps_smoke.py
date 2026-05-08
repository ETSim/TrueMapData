"""CLI smoke and mocked happy-path tests for compress, maps, and info."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    sys.modules.setdefault("noise", SimpleNamespace(snoise2=lambda *args, **kwargs: 0.0))
    return CliRunner(env={"TERM": "dumb"})


def _get_app():
    from tmd.cli.main import app

    return app


def test_compress_downsample_help(runner: CliRunner) -> None:
    result = runner.invoke(_get_app(), ["compress", "downsample", "--help"])
    assert result.exit_code == 0
    assert "Downsample" in result.stdout


def test_compress_quantize_help(runner: CliRunner) -> None:
    result = runner.invoke(_get_app(), ["compress", "quantize", "--help"])
    assert result.exit_code == 0
    assert "Quantize" in result.stdout


def test_compress_combined_help(runner: CliRunner) -> None:
    result = runner.invoke(_get_app(), ["compress", "combined", "--help"])
    assert result.exit_code == 0
    assert "downsampling" in result.stdout.lower() and "quantization" in result.stdout.lower()


def test_compress_batch_help(runner: CliRunner) -> None:
    result = runner.invoke(_get_app(), ["compress", "batch", "--help"])
    assert result.exit_code == 0
    assert "batch" in result.stdout.lower() or "compress" in result.stdout.lower()


def test_terrain_generate_help(runner: CliRunner) -> None:
    result = runner.invoke(_get_app(), ["terrain", "generate", "--help"])
    assert result.exit_code == 0
    assert "terrain" in result.stdout.lower() or "pattern" in result.stdout.lower()


def test_maps_normal_help(runner: CliRunner) -> None:
    result = runner.invoke(_get_app(), ["maps", "normal", "--help"])
    assert result.exit_code == 0
    assert "normal map" in result.stdout.lower() or "normal" in result.stdout


def test_maps_list_runs(runner: CliRunner) -> None:
    result = runner.invoke(_get_app(), ["maps", "list"])
    assert result.exit_code == 0
    assert "normal" in result.stdout.lower()


def test_version_command_runs(runner: CliRunner) -> None:
    result = runner.invoke(_get_app(), ["version"])
    assert result.exit_code == 0
    assert "TMD" in result.stdout


def test_compress_downsample_forwards_to_compress_tmd_file(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inp = tmp_path / "in.tmd"
    inp.write_bytes(b"dummy")

    captured: dict = {}

    def fake_compress(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "input_file": str(kwargs["tmd_file"]),
            "output_file": str(tmp_path / "out.tmd"),
            "mode": "downsample",
            "original_dimensions": "4x4",
            "compressed_dimensions": "2x2",
            "original_size": 1024,
            "compressed_size": 400,
            "size_reduction": 0.6,
            "original_range": (0.0, 1.0),
            "compressed_range": (0.0, 1.0),
            "scale": kwargs.get("scale"),
            "method": kwargs.get("method"),
        }

    monkeypatch.setattr("tmd.cli.apps.compress_app.compress_tmd_file", fake_compress)

    result = runner.invoke(
        _get_app(),
        [
            "compress",
            "downsample",
            str(inp),
            "--scale",
            "0.25",
            "--method",
            "nearest",
        ],
    )
    assert result.exit_code == 0, result.stdout or str(result.exception)
    assert captured["tmd_file"] == inp
    assert captured["mode"] == "downsample"
    assert captured["scale"] == pytest.approx(0.25)
    assert captured["method"] == "nearest"


def test_compress_quantize_forwards_to_compress_tmd_file(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inp = tmp_path / "q.tmd"
    inp.write_bytes(b"dummy")

    captured: dict = {}

    def fake_compress(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "input_file": str(kwargs["tmd_file"]),
            "output_file": str(tmp_path / "out_q.tmd"),
            "mode": "quantize",
            "original_dimensions": "8x8",
            "compressed_dimensions": "8x8",
            "original_size": 2048,
            "compressed_size": 900,
            "size_reduction": 0.55,
            "original_range": (0.0, 1.0),
            "compressed_range": (0.0, 1.0),
            "levels": kwargs.get("levels"),
        }

    monkeypatch.setattr("tmd.cli.apps.compress_app.compress_tmd_file", fake_compress)

    result = runner.invoke(
        _get_app(),
        ["compress", "quantize", str(inp), "--levels", "128"],
    )
    assert result.exit_code == 0, result.stdout or str(result.exception)
    assert captured["tmd_file"] == inp
    assert captured["mode"] == "quantize"
    assert captured["levels"] == 128


def test_compress_combined_forwards_to_compress_tmd_file(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inp = tmp_path / "c.tmd"
    inp.write_bytes(b"dummy")

    captured: dict = {}

    def fake_compress(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "input_file": str(kwargs["tmd_file"]),
            "output_file": str(tmp_path / "out_c.tmd"),
            "mode": "both",
            "original_dimensions": "10x10",
            "compressed_dimensions": "5x5",
            "original_size": 4096,
            "compressed_size": 800,
            "size_reduction": 0.8,
            "original_range": (0.0, 2.0),
            "compressed_range": (0.0, 2.0),
            "scale": kwargs.get("scale"),
            "method": kwargs.get("method"),
            "levels": kwargs.get("levels"),
        }

    monkeypatch.setattr("tmd.cli.apps.compress_app.compress_tmd_file", fake_compress)

    result = runner.invoke(
        _get_app(),
        [
            "compress",
            "combined",
            str(inp),
            "--scale",
            "0.4",
            "--levels",
            "64",
            "--method",
            "nearest",
        ],
    )
    assert result.exit_code == 0, result.stdout or str(result.exception)
    assert captured["mode"] == "both"
    assert captured["scale"] == pytest.approx(0.4)
    assert captured["levels"] == 64
    assert captured["method"] == "nearest"


def test_maps_normal_forwards_to_export_maps_command(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inp = tmp_path / "surface.tmd"
    inp.write_bytes(b"x")

    class FakeTMD:
        @classmethod
        def load(cls, path: str):
            return SimpleNamespace(metadata={"file": "md"})

    monkeypatch.setattr("tmd.cli.apps.export_maps_app.TMD", FakeTMD)

    captured: dict = {}

    def fake_export(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("tmd.cli.apps.export_maps_app.export_maps_command", fake_export)

    result = runner.invoke(
        _get_app(),
        [
            "maps",
            "normal",
            str(inp),
            "--strength",
            "2.5",
            "--metadata",
            '{"extra": 1}',
        ],
    )
    assert result.exit_code == 0, result.stdout or str(result.exception)
    assert captured["args"][0] == inp
    assert captured["args"][2] == ["normal"]
    assert captured["kwargs"]["strength"] == pytest.approx(2.5)
    merged = captured["kwargs"]["metadata"]
    assert merged["file"] == "md"
    assert merged["extra"] == 1


def test_info_cli_delegates_to_display_file_info(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tmd_path = tmp_path / "sample.tmd"
    tmd_path.write_bytes(b"y")

    called: dict = {}

    def fake_display(path, show_sample=False):
        called["path"] = path
        called["show_sample"] = show_sample
        return True

    monkeypatch.setattr("tmd.cli.apps.info_app.display_file_info", fake_display)

    result = runner.invoke(_get_app(), ["info", str(tmd_path), "--show-sample"])
    assert result.exit_code == 0, result.stdout or str(result.exception)
    assert called["path"] == tmd_path
    assert called["show_sample"] is True
