"""Tests for :mod:`tmd.sequence.exporters.compression` strategies and helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tmd.compression.base import TMDDataExporter, TMDDataImporter
from tmd.sequence.exporters.compression import (
    CompressionStrategy,
    CompressionStrategyFactory,
    NPYCompressionStrategy,
    NPZCompressionStrategy,
    PickleCompressionStrategy,
    compress_sequence,
    decompress_sequence,
    get_appropriate_strategy,
)


def test_supported_formats_and_get_strategy_case_insensitive() -> None:
    fmt = set(CompressionStrategyFactory.supported_formats())
    assert fmt == {"npz", "pickle", "npy"}
    assert isinstance(CompressionStrategyFactory.get_strategy("NPZ"), NPZCompressionStrategy)
    assert isinstance(CompressionStrategyFactory.get_strategy("Pickle"), PickleCompressionStrategy)
    assert isinstance(CompressionStrategyFactory.get_strategy("nPy"), NPYCompressionStrategy)


def test_get_strategy_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported compression format"):
        CompressionStrategyFactory.get_strategy("zip")


def test_npz_strategy_roundtrip(tmp_path) -> None:
    path = tmp_path / "seq.npz"
    data = {"frames": [np.ones((2, 2), dtype=np.float32)], "n": 1}
    strat = NPZCompressionStrategy(compress=True)
    strat.compress(data, str(path))
    loaded = strat.decompress(str(path))
    assert loaded["n"] == 1
    assert len(loaded["frames"]) == 1
    np.testing.assert_array_equal(loaded["frames"][0], data["frames"][0])


def test_pickle_strategy_roundtrip(tmp_path) -> None:
    path = tmp_path / "seq.pkl"
    data = {"frames": [np.zeros((3, 3), dtype=np.float64)], "tag": "x"}
    strat = PickleCompressionStrategy()
    strat.compress(data, str(path))
    loaded = strat.decompress(str(path))
    assert loaded["tag"] == "x"
    np.testing.assert_array_equal(loaded["frames"][0], data["frames"][0])


def test_npy_strategy_roundtrip(tmp_path) -> None:
    path = tmp_path / "seq.npy"
    hm = np.arange(12, dtype=np.float32).reshape(3, 4)
    data = {"height_map": hm}
    strat = NPYCompressionStrategy()
    strat.compress(data, str(path))
    loaded = strat.decompress(str(path))
    assert "height_map" in loaded
    np.testing.assert_array_equal(loaded["height_map"], hm)


def test_compress_and_decompress_sequence_npz(tmp_path) -> None:
    path = tmp_path / "bundle.npz"
    data = {"k": np.array([1, 2, 3])}
    out = compress_sequence(data, str(path), format_type="npz", compress=True)
    assert Path(out).resolve() == path.resolve()
    loaded = decompress_sequence(str(path), format_type="npz")
    np.testing.assert_array_equal(loaded["k"], data["k"])


def test_decompress_sequence_infers_extension(tmp_path) -> None:
    path = tmp_path / "auto.npz"
    compress_sequence({"a": 1}, str(path), format_type="npz")
    loaded = decompress_sequence(str(path), format_type=None)
    assert loaded["a"] == 1


def test_decompress_sequence_cannot_infer_extension(tmp_path) -> None:
    path = tmp_path / "unknown.xyz"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="infer"):
        decompress_sequence(str(path), format_type=None)


def test_get_appropriate_strategy_npz() -> None:
    s = get_appropriate_strategy("data.npz")
    assert isinstance(s, NPZCompressionStrategy)


def test_get_appropriate_strategy_npy_and_pickle() -> None:
    assert isinstance(get_appropriate_strategy("seq.npy"), NPYCompressionStrategy)
    assert isinstance(get_appropriate_strategy("archive.pickle"), PickleCompressionStrategy)


def test_compress_sequence_propagates_exporter_errors(tmp_path: Path) -> None:
    path = tmp_path / "empty.npy"
    with pytest.raises(TypeError):
        compress_sequence({}, str(path), format_type="npy")


def test_decompress_sequence_propagates_corrupt_npz(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.npz"
    path.write_bytes(b"not a real npz")
    with pytest.raises(Exception):
        decompress_sequence(str(path), format_type="npz")


class _DummyStrategy(CompressionStrategy):
    def get_exporter(self, **kwargs) -> TMDDataExporter:
        raise NotImplementedError

    def get_importer(self) -> TMDDataImporter:
        raise NotImplementedError


def test_register_strategy() -> None:
    key = "zzz_unit_test_compression"
    CompressionStrategyFactory.register_strategy(key, _DummyStrategy)
    try:
        assert isinstance(CompressionStrategyFactory.get_strategy(key), _DummyStrategy)
    finally:
        CompressionStrategyFactory._strategies.pop(key, None)
