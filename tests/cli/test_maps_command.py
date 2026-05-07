"""Tests for helpers in ``tmd.cli.commands.maps``."""

from __future__ import annotations

from tmd.cli.commands.maps import process_metadata


def test_process_metadata_valid_json_object() -> None:
    md = '{"scale": 1.5, "flag": true}'
    assert process_metadata(md) == {"scale": 1.5, "flag": True}


def test_process_metadata_valid_json_array_returns_list() -> None:
    assert process_metadata("[1, 2]") == [1, 2]


def test_process_metadata_invalid_json_returns_empty_dict() -> None:
    assert process_metadata("not json") == {}


def test_process_metadata_empty_string_returns_empty_dict() -> None:
    assert process_metadata("") == {}


def test_process_metadata_non_string_returns_empty_dict() -> None:
    assert process_metadata(42) == {}  # type: ignore[arg-type]


def test_process_metadata_json_null_returns_none() -> None:
    assert process_metadata("null") is None
