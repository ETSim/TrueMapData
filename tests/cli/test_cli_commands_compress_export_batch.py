"""Direct tests for ``compress``, ``export``, and ``batch`` CLI command modules."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from tmd.cli.main import app
from tmd.cli.commands import batch as batch_mod
from tmd.cli.commands import compress as compress_mod
from tmd.cli.commands import export as export_mod


def test_batch_processor_find_files_non_recursive(tmp_path: Path) -> None:
    (tmp_path / "a.tmd").write_bytes(b"1")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.tmd").write_bytes(b"2")
    p = batch_mod.BatchProcessor(tmp_path, recursive=False)
    names = {f.name for f in p.find_files()}
    assert names == {"a.tmd"}


def test_batch_processor_find_files_recursive(tmp_path: Path) -> None:
    (tmp_path / "root.tmd").write_bytes(b"1")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "deep.tmd").write_bytes(b"2")
    p = batch_mod.BatchProcessor(tmp_path, recursive=True, pattern="*.tmd")
    assert len(p.find_files()) >= 2


def test_batch_processor_no_matching_files(tmp_path: Path) -> None:
    p = batch_mod.BatchProcessor(tmp_path)
    r = p.process_files(lambda path: True, description="noop")
    assert r["total"] == 0 and r["success"] == 0


def test_display_file_info_command_ok(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hm = np.linspace(0, 1, 6, dtype=np.float32).reshape(2, 3)

    class Loaded:
        def height_map(self):
            return hm

        def metadata(self):
            return {"x_length": 10.0, "y_length": 10.0}

    class _FakePanel:
        @staticmethod
        def fit(*a, **k):
            return "[panel]"

    monkeypatch.setattr(compress_mod, "load_tmd_file", lambda *a, **k: Loaded())
    monkeypatch.setattr(compress_mod, "console", SimpleNamespace(print=lambda *a, **k: None))
    monkeypatch.setattr("rich.panel.Panel", _FakePanel)
    monkeypatch.setattr("tmd.cli.core.display_metadata", lambda *a, **k: None)

    tmd = tmp_path / "x.tmd"
    tmd.write_bytes(b"pad")
    assert compress_mod.display_file_info_command(tmd, show_sample=False) is True


def test_compress_tmd_command_rejects_invalid_scale(tmp_path: Path) -> None:
    f = tmp_path / "in.tmd"
    f.write_bytes(b"x")
    assert compress_mod.compress_tmd_command(f, mode="downsample", scale=1.0) is False


def test_compress_batch_command_delegates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "one.tmd").write_bytes(b"1")
    calls: list[Path] = []

    def capture(tmd_file, **kwargs):
        calls.append(Path(tmd_file))
        return True

    monkeypatch.setattr(compress_mod, "compress_tmd_command", capture)
    assert compress_mod.compress_batch_command(tmp_path, mode="downsample", scale=0.5) is True
    assert calls and calls[0].name == "one.tmd"


def test_export_command_unknown_format(tmp_path: Path) -> None:
    f = tmp_path / "f.tmd"
    f.write_bytes(b"x")
    assert export_mod.export_command(f, None, "___unknown_export_fmt___") is False


def test_display_config_info_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(export_mod, "console", SimpleNamespace(print=lambda *a, **k: None))
    export_mod.display_config_info(tmp_path / "in.tmd", tmp_path / "out", ["normal"], {"fast": True})


def test_export_maps_command_single_type_mocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    hm = np.zeros((4, 4), dtype=np.float32)
    fake_tmd = SimpleNamespace(height_map=hm, metadata={"x_length": 1.0, "y_length": 1.0})

    class _FakeProgInner:
        def add_task(self, *a, **k):
            return 0

        def update(self, *a, **k):
            pass

        def advance(self, *a, **k):
            pass

    class _FakeProgress:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return _FakeProgInner()

        def __exit__(self, *a, **k):
            return False

    monkeypatch.setattr(export_mod.TMD, "load", classmethod(lambda cls, p: fake_tmd))
    monkeypatch.setattr(
        export_mod.MapExporter,
        "export_map",
        staticmethod(lambda *a, **k: str(tmp_path / "out.png")),
    )
    monkeypatch.setattr(export_mod, "Progress", lambda *a, **k: _FakeProgress())

    inp = tmp_path / "in.tmd"
    inp.write_bytes(b"1")
    outd = tmp_path / "textures"
    assert export_mod.export_maps_command(inp, output_dir=outd, types=["height"], compress=0, format="png") is True


def test_maps_batch_typer_help() -> None:
    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["maps", "batch", "--help"])
    assert r.exit_code == 0
    assert "batch" in r.stdout.lower() or "export" in r.stdout.lower()
