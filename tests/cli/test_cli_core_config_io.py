"""Tests for ``tmd.cli.core.config`` and pure/path helpers in ``tmd.cli.core.io``."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.modules.setdefault("noise", SimpleNamespace(snoise2=lambda *args, **kwargs: 0.0))
config_mod = importlib.import_module("tmd.cli.core.config")
io_mod = importlib.import_module("tmd.cli.core.io")
find_files_by_pattern = io_mod.find_files_by_pattern
get_file_extension = io_mod.get_file_extension
FileError = importlib.import_module("tmd.cli.exceptions").FileError


@pytest.fixture
def isolated_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "tmd_config_test.json"
    monkeypatch.setattr(config_mod, "get_config_path", lambda: path)
    return path


def test_load_config_creates_file_with_defaults(isolated_config_path: Path) -> None:
    assert not isolated_config_path.exists()
    cfg = config_mod.load_config()
    assert cfg["output_dir"] == "tmd_output"
    assert cfg["default_plotter"] == "matplotlib"
    assert isolated_config_path.exists()


def test_get_set_config_value(isolated_config_path: Path) -> None:
    config_mod.set_config_value("custom_key", 99)
    assert config_mod.get_config_value("custom_key") == 99
    assert config_mod.get_config_value("missing", "fallback") == "fallback"


def test_update_recent_files_order_and_cap(isolated_config_path: Path) -> None:
    for i in range(12):
        config_mod.update_recent_files(f"/f{i}.tmd")
    recent = config_mod.load_config()["recent_files"]
    assert len(recent) == 10
    assert recent[0] == "/f11.tmd"
    assert recent[-1] == "/f2.tmd"


def test_reset_config(isolated_config_path: Path) -> None:
    config_mod.set_config_value("custom_key", 1)
    config_mod.reset_config()
    cfg = config_mod.load_config()
    assert "custom_key" not in cfg
    assert cfg["output_dir"] == "tmd_output"


def test_get_file_extension_plotly() -> None:
    assert get_file_extension("plotly") == ".html"
    assert get_file_extension("PLOTLY") == ".html"


def test_get_file_extension_matplotlib_uses_saved_image_format(isolated_config_path: Path) -> None:
    cfg = config_mod.load_config()
    cfg["image_format"] = "webp"
    config_mod.save_config(cfg)
    assert get_file_extension("matplotlib") == ".webp"


def test_find_files_by_pattern_non_recursive(tmp_path: Path) -> None:
    (tmp_path / "a.tmd").write_text("1")
    (tmp_path / "b.txt").write_text("2")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "c.tmd").write_text("3")
    files = find_files_by_pattern(tmp_path, "*.tmd", recursive=False)
    assert sorted(p.name for p in files) == ["a.tmd"]


def test_find_files_by_pattern_recursive(tmp_path: Path) -> None:
    (tmp_path / "a.tmd").write_text("1")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "b.tmd").write_text("2")
    files = find_files_by_pattern(tmp_path, "*.tmd", recursive=True)
    assert sorted(p.name for p in files) == ["a.tmd", "b.tmd"]


def test_create_output_dir_when_directory_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = tmp_path / "already_there"
    existing.mkdir()
    messages: list[str] = []

    def capture_print(*args, **kwargs) -> None:
        messages.extend(str(a) for a in args)

    monkeypatch.setattr(io_mod.console, "print", capture_print)

    out = io_mod.create_output_dir(base_dir=str(existing))
    assert out == existing
    assert any("Using existing directory" in m for m in messages)


def test_create_output_dir_uses_config_and_subdir(
    isolated_config_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = config_mod.load_config()
    cfg["output_dir"] = str(tmp_path / "exports")
    config_mod.save_config(cfg)

    captured: list[str] = []
    monkeypatch.setattr(io_mod.console, "print", lambda message: captured.append(str(message)))

    out = io_mod.create_output_dir(subdir="maps")

    assert out == tmp_path / "exports" / "maps"
    assert out.is_dir()
    assert any("Created directory" in message for message in captured)


def test_get_output_filename_uses_generated_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(io_mod, "create_output_dir", lambda base_dir=None, subdir=None: tmp_path / "out")
    printed: list[str] = []
    monkeypatch.setattr(io_mod.console, "print", lambda message: printed.append(str(message)))

    output = io_mod.get_output_filename(Path("sample.tmd"), "plotly", "height")

    assert output == tmp_path / "out" / "sample_height_plotly.html"
    assert any("Output will be saved to" in message for message in printed)


def test_get_output_filename_returns_explicit_output(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.png"
    assert io_mod.get_output_filename(Path("sample.tmd"), "matplotlib", "height", output=explicit) == explicit


def test_load_tmd_file_uses_cached_data(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tmd

    file_path = tmp_path / "cached.tmd"
    file_path.write_bytes(b"x")
    success_messages: list[str] = []

    class FakeTMD:
        def __init__(self, height_map, metadata):
            self.height_map = height_map
            self.metadata = metadata

    monkeypatch.setattr(tmd, "TMD", FakeTMD)
    monkeypatch.setattr(
        io_mod,
        "_get_caching_module",
        lambda: SimpleNamespace(get_cached_tmd_data=lambda path: ([[1.0]], {"cached": True})),
    )
    monkeypatch.setattr(io_mod, "print_success", lambda message: success_messages.append(message))

    tmd_obj = io_mod.load_tmd_file(file_path, with_console_status=True, use_cache=True)

    assert isinstance(tmd_obj, FakeTMD)
    assert tmd_obj.metadata == {"cached": True}
    assert success_messages == [f"Loaded {file_path.name} from cache"]


def test_load_tmd_file_caches_fresh_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import tmd

    file_path = tmp_path / "fresh.tmd"
    file_path.write_bytes(b"x")
    cached_calls: list[tuple[Path, object, object]] = []

    class FakeTMD:
        def __init__(self, height_map, metadata):
            self.height_map = height_map
            self.metadata = metadata

        @classmethod
        def load(cls, path: Path):
            assert path == file_path
            return cls([[2.0]], {"loaded": True})

    monkeypatch.setattr(tmd, "TMD", FakeTMD)
    monkeypatch.setattr(
        io_mod,
        "_get_caching_module",
        lambda: SimpleNamespace(
            get_cached_tmd_data=lambda path: None,
            cache_tmd_data=lambda path, height_map, metadata: cached_calls.append(
                (path, height_map, metadata)
            ),
        ),
    )

    tmd_obj = io_mod.load_tmd_file(file_path, use_cache=True)

    assert isinstance(tmd_obj, FakeTMD)
    assert cached_calls == [(file_path, [[2.0]], {"loaded": True})]


def test_load_tmd_file_raises_file_error_without_console_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tmd

    file_path = tmp_path / "broken.tmd"
    file_path.write_bytes(b"x")

    class FakeTMD:
        @classmethod
        def load(cls, path: Path):
            raise RuntimeError("boom")

    monkeypatch.setattr(tmd, "TMD", FakeTMD)

    with pytest.raises(FileError, match="Failed to load broken.tmd: boom"):
        io_mod.load_tmd_file(file_path, with_console_status=False, use_cache=False)
