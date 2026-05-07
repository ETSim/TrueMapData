"""Round-trip tests for TMD compression exporters/importers."""

from __future__ import annotations

import numpy as np
import pytest

from tmd.compression.npz import NPZExporter, NPZImporter
from tmd.compression.npy import NPYExporter, NPYImporter, export_to_npy, load_from_npy
from tmd.compression.pickle import PickleExporter, PickleImporter
from tmd.compression.zip import ZIPExporter, ZIPImporter


def _sample_payload() -> dict:
    return {
        "height_map": np.arange(12, dtype=np.float32).reshape(3, 4),
        "label": "test",
    }


@pytest.mark.parametrize("compress", [True, False])
def test_npz_roundtrip(tmp_path, compress: bool) -> None:
    path = tmp_path / "data.npz"
    data = _sample_payload()
    NPZExporter(compress=compress).export(data, str(path))
    loaded = NPZImporter().load(str(path))
    assert loaded["label"] == data["label"]
    np.testing.assert_array_equal(loaded["height_map"], data["height_map"])


def test_pickle_roundtrip(tmp_path) -> None:
    path = tmp_path / "data.pkl"
    data = _sample_payload()
    PickleExporter().export(data, str(path))
    loaded = PickleImporter().load(str(path))
    assert loaded["label"] == data["label"]
    np.testing.assert_array_equal(loaded["height_map"], data["height_map"])


def test_npy_export_to_npy_array_roundtrip(tmp_path) -> None:
    path = tmp_path / "h.npy"
    arr = np.linspace(0, 1, 12, dtype=np.float64).reshape(4, 3)
    export_to_npy(arr, str(path))
    loaded = load_from_npy(str(path))
    np.testing.assert_array_equal(loaded, arr)


def test_npy_export_to_npy_dict_roundtrip(tmp_path) -> None:
    path = tmp_path / "h2.npy"
    arr = np.ones((5, 6), dtype=np.float32)
    export_to_npy({"height_map": arr}, str(path))
    loaded = load_from_npy(str(path))
    np.testing.assert_array_equal(loaded, arr)


def test_npy_exporter_importer_roundtrip(tmp_path) -> None:
    path = tmp_path / "bundle.npy"
    data = _sample_payload()
    NPYExporter().export(data, str(path))
    loaded = NPYImporter().load(str(path))
    assert "height_map" in loaded
    np.testing.assert_array_equal(loaded["height_map"], data["height_map"])


def test_export_to_npy_rejects_non_array() -> None:
    with pytest.raises(TypeError, match="NumPy array"):
        export_to_npy("not an array", "out.npy")


def test_export_to_npy_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError, match="2D"):
        export_to_npy(np.zeros((2, 3, 4)), "out.npy")


def test_load_from_npy_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_from_npy(str(tmp_path / "missing.npy"))


def test_npz_load_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        NPZImporter().load(str(tmp_path / "missing.npz"))


def test_zip_roundtrip_small_payload(tmp_path) -> None:
    path = tmp_path / "data.zip"
    data = {
        "version": 1,
        "height_map": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    }
    ZIPExporter(
        compression_level=6,
        optimize=True,
        chunk_threshold_mb=1000.0,
        metadata_format="json",
    ).export(data, str(path))
    loaded = ZIPImporter().load(str(path))
    assert loaded["version"] == 1
    np.testing.assert_array_equal(loaded["height_map"], data["height_map"])
