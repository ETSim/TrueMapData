"""Roundtrip and error-path tests for NPY, NPZ, and Pickle compression I/O."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from tmd.compression.factory import TMDDataIOFactory
from tmd.compression.npy import NPYExporter, NPYImporter, export_to_npy, load_from_npy
from tmd.compression.npz import NPZExporter, NPZImporter
from tmd.compression.pickle import PickleExporter, PickleImporter


def test_npy_factory_roundtrip(tmp_path: Path) -> None:
    hm = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    p = tmp_path / "a.npy"
    TMDDataIOFactory.get_exporter("npy").export({"height_map": hm}, str(p))
    out = TMDDataIOFactory.get_importer("npy").load(str(p))
    np.testing.assert_array_equal(out["height_map"], hm)


def test_npy_exporter_dict_first_frame(tmp_path: Path) -> None:
    frames = [np.ones((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]
    p = tmp_path / "seq.npy"
    NPYExporter().export({"frames": frames}, str(p))
    loaded = load_from_npy(str(p))
    np.testing.assert_array_equal(loaded, frames[0])


def test_npy_export_invalid_raises(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        export_to_npy({"no_array": 1}, str(tmp_path / "bad.npy"))
    with pytest.raises(ValueError):
        export_to_npy(np.zeros((2, 2, 2), dtype=np.float32), str(tmp_path / "bad2.npy"))


def test_npy_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_from_npy(str(tmp_path / "missing.npy"))


def test_npz_factory_roundtrip_compressed_and_uncompressed(tmp_path: Path) -> None:
    data = {"height_map": np.eye(3, dtype=np.float64), "meta": np.array([1, 2, 3])}
    pc = tmp_path / "c.npz"
    TMDDataIOFactory.get_exporter("npz", compress=True).export(data, str(pc))
    lc = TMDDataIOFactory.get_importer("npz").load(str(pc))
    np.testing.assert_array_equal(lc["height_map"], data["height_map"])
    np.testing.assert_array_equal(lc["meta"], data["meta"])

    pu = tmp_path / "u.npz"
    TMDDataIOFactory.get_exporter("npz", compress=False).export(data, str(pu))
    lu = NPZImporter().load(str(pu))
    assert set(lu.keys()) == set(data.keys())


def test_npz_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="NPZ file not found"):
        NPZImporter().load(str(tmp_path / "nope.npz"))


def test_pickle_factory_roundtrip(tmp_path: Path) -> None:
    payload = {"a": 1, "b": np.arange(6).reshape(2, 3)}
    p = tmp_path / "x.pkl"
    TMDDataIOFactory.get_exporter("pickle").export(payload, str(p))
    out = TMDDataIOFactory.get_importer("pickle").load(str(p))
    assert out["a"] == 1
    np.testing.assert_array_equal(out["b"], payload["b"])


def test_pickle_load_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Pickle file not found"):
        PickleImporter().load(str(tmp_path / "missing.pkl"))
