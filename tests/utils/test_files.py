#!/usr/bin/env python3
"""Tests for TMDFileUtilities and related file helpers."""

from pathlib import Path

import pytest

import tmd.utils.files as files_mod
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

    def test_get_file_size(self, tmp_path: Path) -> None:
        p = tmp_path / "size.bin"
        p.write_bytes(b"abcd")
        assert TMDFileUtilities.get_file_size(p) == 4

    def test_delete_files_by_pattern(self, tmp_path: Path) -> None:
        (tmp_path / "a.tmp").write_text("1", encoding="utf-8")
        (tmp_path / "b.tmp").write_text("2", encoding="utf-8")
        (tmp_path / "c.txt").write_text("3", encoding="utf-8")

        deleted = TMDFileUtilities.delete_files_by_pattern(tmp_path, "*.tmp")

        assert deleted == 2
        assert not (tmp_path / "a.tmp").exists()
        assert not (tmp_path / "b.tmp").exists()
        assert (tmp_path / "c.txt").exists()

    def test_find_files_by_pattern_recursive(self, tmp_path: Path) -> None:
        (tmp_path / "root.tmd").write_text("1", encoding="utf-8")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "child.tmd").write_text("2", encoding="utf-8")

        files = TMDFileUtilities.find_files_by_pattern(tmp_path, "*.tmd", recursive=True)

        assert sorted(path.name for path in files) == ["child.tmd", "root.tmd"]

    def test_open_file_html_uses_webbrowser(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        p = tmp_path / "report.html"
        p.write_text("<html></html>", encoding="utf-8")
        opened: list[str] = []
        monkeypatch.setattr(files_mod.webbrowser, "open", opened.append)

        TMDFileUtilities.open_file(p)

        assert opened == [f"file://{p.absolute()}"]

    def test_open_file_non_html_uses_platform_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        p = tmp_path / "file.txt"
        p.write_text("x", encoding="utf-8")
        commands: list[str] = []

        monkeypatch.setattr(files_mod.sys, "platform", "darwin")
        monkeypatch.setattr(files_mod.os, "system", lambda command: commands.append(command) or 0)

        TMDFileUtilities.open_file(p)

        assert commands == [f"open '{p}'"]

    def test_import_optional_dependency_handles_syntax_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(name: str):
            raise SyntaxError("broken optional dependency")

        monkeypatch.setattr(files_mod.importlib, "import_module", boom)

        assert TMDFileUtilities.import_optional_dependency("broken.module") is None

    def test_check_tmd_dependencies_missing_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_find_spec(name: str):
            if name == "numpy":
                return None
            return object()

        monkeypatch.setattr(files_mod.importlib.util, "find_spec", fake_find_spec)

        assert TMDFileUtilities.check_tmd_dependencies() is False

    def test_check_tmd_dependencies_exit_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fake_find_spec(name: str):
            if name == "numpy":
                return None
            return object()

        monkeypatch.setattr(files_mod.importlib.util, "find_spec", fake_find_spec)

        with pytest.raises(SystemExit):
            TMDFileUtilities.check_tmd_dependencies(exit_on_failure=True)


def test_check_visualization_capabilities_tuple() -> None:
    result = _check_visualization_capabilities()
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert all(isinstance(x, bool) for x in result)
