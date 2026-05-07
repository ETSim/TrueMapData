"""Extra coverage for SequenceExporterFactory paths.

Adds coverage for ``export_powerpoint``, ``export_frames_as_images`` (success and
empty cases), and the canonical-format extension shortcut that's currently 0%.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tmd.sequence.factory import SequenceExporterFactory


def test_get_exporter_known_formats() -> None:
    for fmt in ("gif", "video", "mp4", "avi", "powerpoint", "pptx"):
        assert SequenceExporterFactory.get_exporter(fmt) is not None


def test_get_exporter_unknown_format_returns_none() -> None:
    assert SequenceExporterFactory.get_exporter("does_not_exist") is None


def test_supported_formats_includes_known_aliases() -> None:
    formats = SequenceExporterFactory.supported_formats()
    for expected in ("gif", "animated_gif", "ppt", "mp4"):
        assert expected in formats


def test_get_file_extension_canonical_and_passthrough() -> None:
    assert SequenceExporterFactory.get_file_extension("powerpoint") == "pptx"
    assert SequenceExporterFactory.get_file_extension("video") == "mp4"
    assert SequenceExporterFactory.get_file_extension("unknown_format") == "unknown_format"


def test_ensure_extension_keeps_correct_suffix(tmp_path: Path) -> None:
    p = tmp_path / "already.gif"
    out = SequenceExporterFactory._ensure_extension(str(p), "gif")
    assert out == str(p)


def test_export_powerpoint_delegates(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: list[tuple] = []

    def _seq(cls, frames, output_path, format_type, **kwargs):
        captured.append((format_type, output_path))
        return output_path

    monkeypatch.setattr(SequenceExporterFactory, "export_sequence", classmethod(_seq))
    out = str(tmp_path / "deck.pptx")
    got = SequenceExporterFactory.export_powerpoint([np.zeros((2, 2))], out, slides_per_row=3)
    assert got == out
    assert captured == [("powerpoint", out)]


def test_export_sequence_handles_exporter_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _BoomExporter:
        def export(self, **_kwargs):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(
        SequenceExporterFactory, "get_exporter", classmethod(lambda cls, fmt: _BoomExporter())
    )
    out = str(tmp_path / "x.gif")
    assert (
        SequenceExporterFactory.export_sequence(
            [np.zeros((2, 2), dtype=np.float32)], out, "gif"
        )
        is None
    )


def test_export_frames_as_images_writes_files(tmp_path: Path) -> None:
    frames = [np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4) for _ in range(2)]
    out_dir = tmp_path / "frames"
    paths = SequenceExporterFactory.export_frames_as_images(
        frames, str(out_dir), format_type="png", base_filename="seq", colormap="viridis", dpi=72
    )
    assert len(paths) == 2
    for p in paths:
        assert Path(p).exists()
        assert p.endswith(".png")


def test_export_frames_as_images_with_timestamps(tmp_path: Path) -> None:
    frames = [np.zeros((4, 4), dtype=np.float32), np.ones((4, 4), dtype=np.float32)]
    out_dir = tmp_path / "stamped"
    paths = SequenceExporterFactory.export_frames_as_images(
        frames,
        str(out_dir),
        format_type="png",
        timestamps=[0.0, 0.5],
        dpi=72,
    )
    assert len(paths) == 2


def test_export_frames_as_images_creates_output_directory(tmp_path: Path) -> None:
    frames = [np.zeros((2, 2), dtype=np.float32)]
    out_dir = tmp_path / "deep" / "nested" / "frames"
    assert not out_dir.exists()
    paths = SequenceExporterFactory.export_frames_as_images(frames, str(out_dir), dpi=72)
    assert paths
    assert out_dir.exists()


def test_export_frames_as_images_empty_returns_empty(tmp_path: Path) -> None:
    out_dir = tmp_path / "empty"
    paths = SequenceExporterFactory.export_frames_as_images([], str(out_dir))
    assert paths == []
    assert out_dir.exists()


def test_export_frames_as_images_handles_internal_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If matplotlib raises during render the factory should swallow and return []."""
    import matplotlib.figure as _fig

    def _broken_init(self, *_a, **_k):
        raise RuntimeError("figure init blew up")

    monkeypatch.setattr(_fig.Figure, "__init__", _broken_init)
    paths = SequenceExporterFactory.export_frames_as_images(
        [np.zeros((2, 2), dtype=np.float32)], str(tmp_path / "frames")
    )
    assert paths == []


def test_register_exporter_adds_format() -> None:
    class _DummyExporter:
        def export(self, **kwargs):  # pragma: no cover - registration only
            return None

    SequenceExporterFactory.register_exporter("dummy_test_fmt", _DummyExporter)
    try:
        assert "dummy_test_fmt" in SequenceExporterFactory.supported_formats()
        assert isinstance(
            SequenceExporterFactory.get_exporter("dummy_test_fmt"), _DummyExporter
        )
    finally:
        SequenceExporterFactory._exporters.pop("dummy_test_fmt", None)
        SequenceExporterFactory._format_mapping.pop("dummy_test_fmt", None)
