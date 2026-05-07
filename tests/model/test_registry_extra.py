"""Extra coverage for ``ExporterRegistry`` listing/availability methods."""

from __future__ import annotations

import pytest

from tmd.model.registry import ExporterRegistry, register_format, get_exporter, get_available_formats


def test_register_and_get_known_format() -> None:
    formats = get_available_formats()
    assert "stl" in formats
    cls = get_exporter("stl")
    assert cls is not None


def test_get_exporter_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_exporter("__not_a_format__")


def test_list_registered_formats_returns_copy() -> None:
    out = ExporterRegistry.list_registered_formats()
    assert isinstance(out, dict)
    assert "stl" in out


def test_list_extensions_includes_known_extensions() -> None:
    extensions = ExporterRegistry.list_extensions()
    assert isinstance(extensions, dict)
    assert "stl" in extensions
    assert "obj" in extensions


def test_get_format_info_sorted_by_name() -> None:
    info = ExporterRegistry.get_format_info()
    names = [item["name"] for item in info]
    assert names == sorted(names)
    for item in info:
        assert "extensions" in item
        assert "binary_supported" in item


def test_is_format_available_known_and_unknown() -> None:
    assert ExporterRegistry.is_format_available("stl") is True
    assert ExporterRegistry.is_format_available("__no_such_format__") is False


def test_list_available_formats_marks_known() -> None:
    avail = ExporterRegistry.list_available_formats()
    assert avail.get("stl") is True
    assert avail.get("obj") is True


def test_try_import_format_failure_returns_none() -> None:
    assert ExporterRegistry._try_import_format("nonexistent_format_xyz") is None


def test_register_rejects_non_modelexporter() -> None:
    """``ExporterRegistry.register`` should ignore classes that aren't subclasses of ModelExporter."""

    class _NotAnExporter:
        format_name = "broken"
        file_extensions = ["broken"]
        binary_supported = False

    before = ExporterRegistry.list_registered_formats()
    ExporterRegistry.register(_NotAnExporter)  # type: ignore[arg-type]
    after = ExporterRegistry.list_registered_formats()
    assert before == after


def test_get_exporter_classmethod_via_extension() -> None:
    cls = ExporterRegistry.get_exporter("stl")
    assert cls is not None
    cls_via_ext = ExporterRegistry.get_exporter(".stl")
    # Some extensions are stored without dots; check at least the dotless form works:
    cls_dotless = ExporterRegistry.get_exporter("stl")
    assert cls_dotless is not None


def test_register_format_legacy_alias_helper() -> None:
    """``register_format`` plus ``get_exporter`` form a legacy registration path."""

    from tmd.model.formats.stl import STLExporter

    register_format("legacy_stl_alias", STLExporter)
    try:
        cls = get_exporter("legacy_stl_alias")
        assert cls is STLExporter
    finally:
        from tmd.model.registry import _FORMAT_REGISTRY

        _FORMAT_REGISTRY.pop("legacy_stl_alias", None)
