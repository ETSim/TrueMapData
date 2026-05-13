"""Unit tests for sequence exporter factory, model ExporterRegistry, and model export CLI slice."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest
from rich.console import Console
from typer.testing import CliRunner

from tmd.cli.main import app
from tmd.model.base import ModelExporter, ExportConfig
from tmd.model.registry import ExporterRegistry, get_exporter as module_get_exporter, register_format
from tmd.sequence.exporters.gif import GifExporter
from tmd.sequence.exporters.video import VideoExporter
from tmd.sequence.base import BaseExporter
from tmd.sequence.factory import SequenceExporterFactory


def test_sequence_exporter_factory_builtin_formats() -> None:
    assert isinstance(SequenceExporterFactory.get_exporter("gif"), GifExporter)
    exp = SequenceExporterFactory.get_exporter("MP4")
    assert isinstance(exp, VideoExporter)
    assert "gif" in SequenceExporterFactory.supported_formats()
    assert SequenceExporterFactory.get_file_extension("video") == "mp4"
    assert SequenceExporterFactory.get_file_extension("animated_gif") == "gif"


def test_sequence_exporter_factory_register_and_resolve() -> None:
    class StubExporter(BaseExporter):
        @classmethod
        def supports_format(cls, fmt: str) -> bool:
            return fmt.lower() == "stubseq"

        def export(self, **kwargs) -> Optional[str]:
            return kwargs.get("output_file", "stub.out")

    SequenceExporterFactory.register_exporter("stubseq", StubExporter)
    got = SequenceExporterFactory.get_exporter("stubseq")
    assert isinstance(got, StubExporter)


def test_sequence_exporter_factory_unknown_returns_none() -> None:
    assert SequenceExporterFactory.get_exporter("___no_such_sequence_format___") is None


def test_exporter_registry_rejects_non_subclass(caplog: pytest.LogCaptureFixture) -> None:
    class NotExporter:
        pass

    ExporterRegistry.register(NotExporter)  # type: ignore[arg-type]
    assert any("Not a subclass" in r.message for r in caplog.records)


def test_exporter_registry_get_unknown_format() -> None:
    assert ExporterRegistry.get_exporter("___unknown_model_format___") is None


def test_exporter_registry_discover_registers_stl_exporter() -> None:
    """Regression: ``discover_exporters`` must import ``tmd.model.formats.*`` (not ``tmd.exporters...``)."""
    ExporterRegistry.reset()
    ExporterRegistry.ensure_initialized()
    stl_cls = ExporterRegistry.get_exporter("stl")
    assert stl_cls is not None
    assert getattr(stl_cls, "format_name", "").lower() == "stl"


def test_exporter_registry_manual_register_lookup() -> None:
    """``ExporterRegistry`` accepts explicit ``register`` (package auto-discover may be empty)."""

    class TinyExporter(ModelExporter):
        format_name = "tinyplanonly001"
        file_extensions = ["tp1"]
        binary_supported = False

        @classmethod
        def export(cls, height_map: np.ndarray, filename: str, config: ExportConfig) -> Optional[str]:
            return filename

    ExporterRegistry.register(TinyExporter)
    assert ExporterRegistry.get_exporter("tinyplanonly001") is TinyExporter
    assert ExporterRegistry.get_exporter("tp1") is TinyExporter


def test_module_get_exporter_stl_registered() -> None:
    assert module_get_exporter("stl") is not None


def test_module_register_format_and_get_exporter() -> None:
    class ModExporter(ModelExporter):
        format_name = "zzzplanfmt"
        file_extensions = ["zzz"]
        binary_supported = False

        @classmethod
        def export(cls, height_map: np.ndarray, filename: str, config: ExportConfig) -> Optional[str]:
            return filename

    register_format("zzzplanfmt", ModExporter)
    assert module_get_exporter("zzzplanfmt") is ModExporter
    with pytest.raises(ValueError, match="Unknown format"):
        module_get_exporter("___not_registered_format___")


def test_mesh_formats_cli_invokes_list_model_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    def fake_list() -> None:
        called.append(True)

    monkeypatch.setattr("tmd.cli.apps.export_mesh_app.list_model_formats", fake_list)
    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["mesh", "formats"])
    assert r.exit_code == 0
    assert called == [True]


def test_export_model_stl_path_mocked_factory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from tmd.core import tmd as tmd_mod
    from tmd.cli.commands import model as model_cmd

    hm = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    meta = {"x_length": 1.0, "y_length": 1.0, "x_offset": 0.0, "y_offset": 0.0}
    fake_tmd = SimpleNamespace(height_map=hm, metadata=meta)

    def fake_load(cls, filepath, compute_initial_stats: bool = False):
        return fake_tmd

    monkeypatch.setattr(tmd_mod.TMD, "load", classmethod(fake_load))

    class FakeFactory:
        def export(self, **kwargs):
            assert kwargs["format_name"] == "stl"
            return True

    monkeypatch.setattr("tmd.model.factory.ModelExporterFactory", FakeFactory)
    buf = io.StringIO()
    monkeypatch.setattr(model_cmd, "console", Console(file=buf, force_terminal=True, width=120))

    inp = tmp_path / "in.tmd"
    inp.write_text("dummy", encoding="utf-8")
    outp = tmp_path / "out.stl"
    ok = model_cmd.export_model(inp, outp, "stl", method="adaptive", scale=1.0)
    assert ok is True
