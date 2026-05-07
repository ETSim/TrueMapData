"""Edge-case tests for ``tmd.compression.npy``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tmd.compression.npy import NPYExporter, export_to_npy, load_from_npy


def test_export_to_npy_rejects_non_array(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="NumPy array"):
        export_to_npy("not an array", str(tmp_path / "x.npy"))


def test_export_to_npy_rejects_non_2d(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2D"):
        export_to_npy(np.ones((2, 2, 2), dtype=np.float32), str(tmp_path / "x.npy"))


def test_load_from_npy_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_from_npy(str(tmp_path / "missing.npy"))


def test_load_from_npy_invalid_bytes(tmp_path: Path) -> None:
    p = tmp_path / "bad.npy"
    p.write_bytes(b"not valid npy")
    with pytest.raises(Exception):
        load_from_npy(str(p))


def test_npy_exporter_uses_first_frame(tmp_path: Path) -> None:
    exp = NPYExporter()
    data = {
        "frames": [np.ones((2, 2), dtype=np.float32) * 2.0, np.zeros((2, 2), dtype=np.float32)],
    }
    out = exp.export(data, str(tmp_path / "first.npy"))
    loaded = np.load(out)
    np.testing.assert_array_equal(loaded, np.full((2, 2), 2.0, dtype=np.float32))


def test_npy_exporter_picks_first_ndarray_value(tmp_path: Path) -> None:
    exp = NPYExporter()
    data = {"meta": 1, "elevation": np.arange(6, dtype=np.float64).reshape(2, 3)}
    out = exp.export(data, str(tmp_path / "elev.npy"))
    loaded = np.load(out)
    np.testing.assert_array_equal(loaded, data["elevation"])


def test_npy_exporter_raises_when_no_array(tmp_path: Path) -> None:
    exp = NPYExporter()
    with pytest.raises(TypeError, match="No suitable array"):
        exp.export({"only_meta": "x"}, str(tmp_path / "none.npy"))


def test_npy_importer_returns_height_map_key(tmp_path: Path) -> None:
    from tmd.compression.npy import NPYImporter

    p = tmp_path / "hm.npy"
    export_to_npy(np.eye(3, dtype=np.float32), str(p))
    loaded = NPYImporter().load(str(p))
    assert set(loaded.keys()) == {"height_map"}
    assert loaded["height_map"].shape == (3, 3)
