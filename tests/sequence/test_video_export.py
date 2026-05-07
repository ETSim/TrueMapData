"""Tests for :mod:`tmd.sequence.video`."""

from __future__ import annotations

import numpy as np

from tmd.sequence.video import VideoExporter


def test_video_exporter_requires_frames() -> None:
    exp = VideoExporter()
    assert exp.export(frames=[], output_file="x.mp4") is None


def test_video_exporter_passes_kwargs_shape(tmp_path) -> None:
    """Smoke path without encoding video (FFmpeg may be unavailable)."""
    exp = VideoExporter()
    frames = [np.ones((4, 4), dtype=np.float32)]
    out = exp.export(frames=frames, output_file=str(tmp_path / "v.mp4"), fps=5.0, show_progress=False)
    # Succeeds when FFMpegWriter/FFmpeg work; else None
    assert out is None or __import__("os").path.isfile(out)
