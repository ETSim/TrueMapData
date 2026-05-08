"""Tests for TMDDataIOFactory."""

from __future__ import annotations

import pytest

from tmd.compression.base import TMDDataExporter, TMDDataImporter
from tmd.compression.factory import TMDDataIOFactory


def test_supported_export_formats() -> None:
    fmt = set(TMDDataIOFactory.supported_export_formats())
    assert fmt == {"npz", "pickle", "npy", "zip", "mat"}


def test_supported_import_formats() -> None:
    fmt = set(TMDDataIOFactory.supported_import_formats())
    assert fmt == {"npz", "pickle", "npy", "zip", "mat"}


@pytest.mark.parametrize("key", ["npz", "NPZ", "Npy", "PICKLE", "ZiP"])
def test_get_exporter_case_insensitive(key: str) -> None:
    exp = TMDDataIOFactory.get_exporter(key)
    assert isinstance(exp, TMDDataExporter)


@pytest.mark.parametrize("key", ["npz", "ZIP", "Pickle", "NPY"])
def test_get_importer_case_insensitive(key: str) -> None:
    imp = TMDDataIOFactory.get_importer(key)
    assert isinstance(imp, TMDDataImporter)


def test_get_exporter_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported export format"):
        TMDDataIOFactory.get_exporter("not_a_format")


def test_get_importer_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported import format"):
        TMDDataIOFactory.get_importer("not_a_format")


def test_npz_exporter_accepts_compress_kwarg() -> None:
    exp = TMDDataIOFactory.get_exporter("npz", compress=False)
    assert exp.compress is False


class _DummyExporter(TMDDataExporter):
    def export(self, data, output_path: str) -> str:
        return output_path


class _DummyImporter(TMDDataImporter):
    def load(self, file_path: str):
        return {}


def test_register_exporter_and_importer_roundtrip() -> None:
    TMDDataIOFactory.register_exporter("dummy_fmt_unit_test", _DummyExporter)
    TMDDataIOFactory.register_importer("dummy_fmt_unit_test", _DummyImporter)
    try:
        assert isinstance(
            TMDDataIOFactory.get_exporter("dummy_fmt_unit_test"),
            _DummyExporter,
        )
        assert isinstance(
            TMDDataIOFactory.get_importer("dummy_fmt_unit_test"),
            _DummyImporter,
        )
    finally:
        TMDDataIOFactory._exporters.pop("dummy_fmt_unit_test", None)
        TMDDataIOFactory._importers.pop("dummy_fmt_unit_test", None)
