"""ZIP compression exporter/importer branches and error paths."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from tmd.compression.factory import TMDDataIOFactory
from tmd.compression.zip import (
    ZIPExporter,
    ZIPImporter,
    _calculate_optimal_chunks,
    _export_zip,
    _load_zip,
)


def test_zip_roundtrip_compression_level_zero_stored(tmp_path: Path) -> None:
    path = tmp_path / "stored.zip"
    data = {"tag": "a", "arr": np.eye(3, dtype=np.float32)}
    ZIPExporter(compression_level=0, metadata_format="json").export(data, str(path))
    loaded = ZIPImporter().load(str(path))
    assert loaded["tag"] == "a"
    np.testing.assert_array_equal(loaded["arr"], data["arr"])


def test_zip_export_metadata_txt_format_writes_txt_and_arrays(tmp_path: Path) -> None:
    """TXT metadata is written; importer only supports JSON metadata today."""
    path = tmp_path / "meta_txt.zip"
    data = {"note": "hello", "hm": np.ones((4, 4), dtype=np.float64)}
    ZIPExporter(compression_level=3, metadata_format="txt").export(data, str(path))
    with zipfile.ZipFile(path, "r") as zf:
        assert "metadata.txt" in zf.namelist()
        assert "hm.npy" in zf.namelist()
        txt = zf.read("metadata.txt").decode("utf-8")
        assert "hello" in txt
        with zf.open("hm.npy") as f:
            arr = np.lib.format.read_array(f)
    np.testing.assert_array_equal(arr, data["hm"])


def test_zip_chunked_array_roundtrip(tmp_path: Path) -> None:
    """Force chunking with a low threshold so a modest array splits."""
    path = tmp_path / "chunked.zip"
    # ~4 MB float32 grid — well above 0.01 MB threshold
    hm = np.arange(800 * 800, dtype=np.float32).reshape(800, 800)
    data = {"height_map": hm, "id": 42}
    ZIPExporter(
        compression_level=5,
        optimize=True,
        chunk_threshold_mb=0.01,
        metadata_format="json",
    ).export(data, str(path))
    loaded = ZIPImporter().load(str(path))
    assert loaded["id"] == 42
    np.testing.assert_array_equal(loaded["height_map"], hm)


def test_zip_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="ZIP file not found"):
        ZIPImporter().load(str(tmp_path / "nope.zip"))


def test_zip_load_corrupt_metadata_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.json", "{ not json")
    with pytest.raises(Exception):
        _load_zip(str(path))


def test_zip_exporter_clamps_compression_level(tmp_path: Path) -> None:
    path = tmp_path / "clamped.zip"
    data = {"x": np.zeros((2, 2), dtype=np.float32)}
    ZIPExporter(compression_level=99).export(data, str(path))
    assert path.exists()


def test_factory_zip_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "factory.zip"
    payload = {"k": 1, "a": np.ones((3, 3), dtype=np.float64)}
    TMDDataIOFactory.get_exporter("zip", compression_level=1).export(payload, str(p))
    out = TMDDataIOFactory.get_importer("zip").load(str(p))
    assert out["k"] == 1
    np.testing.assert_array_equal(out["a"], payload["a"])


def test_calculate_optimal_chunks_ndim_not_two() -> None:
    chunks = _calculate_optimal_chunks((5, 4, 3))
    assert chunks
    assert len(chunks[0][0]) == 3


def test_export_zip_propagates_on_invalid_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom_mkdir(*a, **k):
        raise OSError("no dir")

    monkeypatch.setattr("os.makedirs", boom_mkdir)
    with pytest.raises(OSError, match="no dir"):
        _export_zip({"a": np.zeros((2, 2))}, "/nonexistent/__bad__/out.zip")
