"""Tests for ``SequenceExporterFactory.export_sequence`` and helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tmd.sequence.factory import SequenceExporterFactory


def test_export_sequence_empty_frames() -> None:
    assert SequenceExporterFactory.export_sequence([], str(Path("/tmp/x.gif")), "gif") is None


def test_export_sequence_unknown_format() -> None:
    frames = [np.ones((2, 2), dtype=np.float32)]
    assert SequenceExporterFactory.export_sequence(frames, str(Path("/tmp/x.zzz")), "___no_fmt___") is None


def test_export_sequence_success_mocked_exporter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Exp:
        def export(self, *args, **kwargs):
            out = args[1] if len(args) > 1 else kwargs.get("output_file")
            return str(out)

    def _get(cls, fmt: str):
        return _Exp() if fmt == "gif" else None

    monkeypatch.setattr(SequenceExporterFactory, "get_exporter", classmethod(_get))
    frames = [np.zeros((2, 2), dtype=np.float32)]
    outp = tmp_path / "seq_no_ext"
    got = SequenceExporterFactory.export_sequence(frames, str(outp), "gif")
    assert got is not None
    assert str(got).endswith(".gif")


def test_export_gif_delegates_to_export_sequence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple] = []

    def _seq(cls, frames, output_path, format_type, **kwargs):
        calls.append((format_type, kwargs.get("fps")))
        return str(tmp_path / "done.gif")

    monkeypatch.setattr(SequenceExporterFactory, "export_sequence", classmethod(_seq))
    r = SequenceExporterFactory.export_gif([np.ones((2, 2))], str(tmp_path / "x"), fps=5.0)
    assert r is not None and calls == [("gif", 5.0)]


def test_export_video_delegates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _seq(cls, frames, output_path, format_type, **kwargs):
        return output_path if format_type == "video" else None

    monkeypatch.setattr(SequenceExporterFactory, "export_sequence", classmethod(_seq))
    p = str(tmp_path / "v.mp4")
    assert SequenceExporterFactory.export_video([np.ones((2, 2))], p, fps=15.0) == p


def test_ensure_extension_adds_suffix(tmp_path: Path) -> None:
    p = str(tmp_path / "foo")
    out = SequenceExporterFactory._ensure_extension(p, "gif")
    assert out.endswith(".gif")
