"""Tests for :mod:`tmd.core.sequence` (``TMDSequence``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tmd.core.sequence import TMDSequence


def test_empty_sequence() -> None:
    s = TMDSequence("empty")
    assert len(s.frames) == 0
    assert s.get_frame_count() == 0
    assert s.get_frame(0) is None


def test_add_frames_and_accessors(small_heightmap: np.ndarray) -> None:
    s = TMDSequence("s")
    i0 = s.add_frame(small_heightmap, "t0", {"k": 1})
    i1 = s.add_frame(small_heightmap * 2, "t1", None)
    assert i0 == 0 and i1 == 1
    assert s.get_frame_count() == 2
    assert s.get_all_frames()[0].shape == small_heightmap.shape
    assert s.get_timestamp(0) == "t0"
    assert s.get_frame_metadata(0) == {"k": 1}
    assert s.get_tmd_object(0) is None
    s.set_transformation(0, {"offset": 0.1})
    assert s.get_transformation(0) == {"offset": 0.1}


def test_add_frame_rejects_empty() -> None:
    s = TMDSequence()
    assert s.add_frame(np.array([])) == -1
    assert s.add_frame(np.empty((0, 0))) == -1


def test_add_tmd_file(tmp_tmd_path: Path) -> None:
    s = TMDSequence("from disk")
    idx = s.add_tmd_file(tmp_tmd_path)
    assert idx == 0
    assert s.get_frame(0) is not None
    assert s.get_tmd_object(0) is not None


def test_add_tmd_file_missing() -> None:
    s = TMDSequence()
    assert s.add_tmd_file("/no/such/file.tmd") == -1


def test_add_frames_from_folder(tmp_path: Path, tmp_tmd_path: Path) -> None:
    s = TMDSequence("batch")
    n = s.add_frames_from_folder(tmp_path, extension="tmd", sort_method="name", recursive=False)
    assert n == 1


def test_calculate_statistics_and_to_dict(small_heightmap: np.ndarray) -> None:
    s = TMDSequence("st")
    s.add_frame(small_heightmap, "a", {})
    stats = s.calculate_statistics()
    assert "mean" in stats and len(stats["mean"]) == 1
    d = s.to_dict()
    assert d["name"] == "st" and len(d["frames"]) == 1


def test_export_empty_returns_none() -> None:
    s = TMDSequence()
    assert s.export("x.gif", "gif") is None


def test_export_to_gif_integration_small(tmp_path: Path, small_heightmap: np.ndarray) -> None:
    s = TMDSequence("gifseq")
    s.add_frame(small_heightmap, "a")
    s.add_frame(small_heightmap + 0.1, "b")
    out = tmp_path / "out.gif"
    path = s.export_to_gif(str(out), fps=5.0)
    assert path is not None
    assert Path(path).exists()


def test_npz_roundtrip(tmp_path: Path, small_heightmap: np.ndarray) -> None:
    s = TMDSequence("npz")
    s.add_frame(small_heightmap, "t")
    path = str(tmp_path / "seq.npz")
    assert s.save_to_npz(path) is True
    loaded = TMDSequence.load_from_npz(path)
    assert loaded is not None
    assert loaded.get_frame_count() == 1


def test_get_supported_export_formats() -> None:
    fmts = TMDSequence().get_supported_export_formats()
    assert isinstance(fmts, list) and "gif" in fmts


def test_align_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        TMDSequence().align_height_maps_opencv()


def test_invalid_indices_warnings(small_heightmap: np.ndarray) -> None:
    s = TMDSequence()
    s.add_frame(small_heightmap, "t")
    assert s.get_frame(99) is None
    assert s.get_timestamp(99) is None
    assert s.get_frame_metadata(99) is None
