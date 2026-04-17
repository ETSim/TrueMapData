#!/usr/bin/env python3
"""Tests for TMDFileUtilities and related file helpers."""

from pathlib import Path


from tmd.utils.files import TMDFileUtilities, _check_visualization_capabilities


class TestTMDFileUtilities:
    def test_ensure_directory_creates_path(self, tmp_path: Path) -> None:
        d = tmp_path / "nested" / "dir"
        out = TMDFileUtilities.ensure_directory(d)
        assert out == d
        assert d.is_dir()

    def test_ensure_directory_exists_alias(self, tmp_path: Path) -> None:
        d = tmp_path / "a"
        assert TMDFileUtilities.ensure_directory_exists(d) == d

    def test_import_optional_dependency_numpy(self) -> None:
        m = TMDFileUtilities.import_optional_dependency("numpy")
        assert m is not None
        assert hasattr(m, "ndarray")

    def test_import_optional_dependency_missing(self) -> None:
        assert TMDFileUtilities.import_optional_dependency("not_a_real_module_xyz_12345") is None

    def test_check_tmd_dependencies(self) -> None:
        assert TMDFileUtilities.check_tmd_dependencies() is True

    def test_json_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        data = {"a": 1, "b": [2, 3]}
        TMDFileUtilities.save_json(data, p)
        assert TMDFileUtilities.load_json(p) == data

    def test_get_file_info(self, tmp_path: Path) -> None:
        p = tmp_path / "f.txt"
        p.write_text("hello", encoding="utf-8")
        info = TMDFileUtilities.get_file_info(p)
        assert info["exists"] is True
        assert info["is_file"] is True
        assert info["size"] == 5

    def test_delete_file(self, tmp_path: Path) -> None:
        p = tmp_path / "todel.txt"
        p.write_text("x", encoding="utf-8")
        assert TMDFileUtilities.delete_file(p) is True
        assert not p.exists()
        assert TMDFileUtilities.delete_file(p) is False


def test_check_visualization_capabilities_tuple() -> None:
    result = _check_visualization_capabilities()
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(x, bool) for x in result)
