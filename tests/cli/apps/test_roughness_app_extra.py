"""Extra coverage for tmd.cli.apps.roughness_app via a fake surfalize module."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from typer.testing import CliRunner

from tmd.cli.apps import roughness_app as ra


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


class _FakeSurface:
    """Minimal Surface stand-in that mimics Surfalize's API surface."""

    ISO_PARAMETERS: tuple[str, ...] = ("Sa", "Sq", "Sz", "Ssk", "Sku")

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def load(cls, path: str):
        if "fail" in path:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "fail")
        return cls()

    def level(self) -> "_FakeSurface":
        return self

    def roughness_parameters(self, names: Optional[List[str]] = None) -> Dict[str, Any]:
        full = {"Sa": 0.5, "Sq": 0.6, "Sz": 1.2, "Ssk": -0.1, "Sku": 3.0}
        if names is None:
            return full
        return {n: full.get(n, 0.0) for n in names}


@pytest.fixture
def fake_surfalize_module(monkeypatch: pytest.MonkeyPatch):
    fake_pkg = ModuleType("surfalize")
    fake_pkg.Surface = _FakeSurface  # type: ignore[attr-defined]

    fake_exc = ModuleType("surfalize.exceptions")

    class CorruptedFileError(Exception):
        pass

    fake_exc.CorruptedFileError = CorruptedFileError  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "surfalize", fake_pkg)
    monkeypatch.setitem(sys.modules, "surfalize.exceptions", fake_exc)
    return fake_pkg


def test_iso_parameter_names_uses_surface_attribute() -> None:
    names = ra._iso_parameter_names(_FakeSurface)
    assert names == list(_FakeSurface.ISO_PARAMETERS)


def test_iso_parameter_names_falls_back_when_missing() -> None:
    class _NoIso:
        pass

    names = ra._iso_parameter_names(_NoIso)
    assert "Sa" in names
    assert "Sq" in names


def test_surfalize_imports_returns_surface(fake_surfalize_module) -> None:
    Surface = ra._surfalize_imports()
    assert Surface is _FakeSurface


def test_surfalize_imports_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    import typer

    monkeypatch.setitem(sys.modules, "surfalize", None)
    with pytest.raises(typer.Exit):
        ra._surfalize_imports()


def test_surface_from_truemap_tmd(monkeypatch: pytest.MonkeyPatch, tmp_tmd_path: Path, fake_surfalize_module) -> None:
    surface = ra._surface_from_truemap_tmd(tmp_tmd_path, _FakeSurface)
    assert isinstance(surface, _FakeSurface)
    assert len(surface.args) >= 3


def test_load_surface_for_roughness_native_path(fake_surfalize_module, tmp_path: Path) -> None:
    fake = tmp_path / "fixture.tmd"
    fake.write_bytes(b"")
    surf = ra._load_surface_for_roughness(fake, _FakeSurface)
    assert isinstance(surf, _FakeSurface)


def test_load_surface_for_roughness_falls_back(
    monkeypatch: pytest.MonkeyPatch, fake_surfalize_module, tmp_tmd_path: Path
) -> None:
    """If Surfalize's native loader rejects the file, we fall back to TMD."""
    failure_path = tmp_tmd_path.parent / "fail-fixture.tmd"
    failure_path.write_bytes(tmp_tmd_path.read_bytes())
    surf = ra._load_surface_for_roughness(failure_path, _FakeSurface)
    assert isinstance(surf, _FakeSurface)


def test_parse_params_default_returns_full_set() -> None:
    out = ra._parse_params(None, all_params=False, quick=False, Surface=_FakeSurface)
    assert out == list(_FakeSurface.ISO_PARAMETERS)


def test_parse_params_all_returns_none() -> None:
    assert ra._parse_params(None, all_params=True, quick=False, Surface=_FakeSurface) is None


def test_parse_params_explicit_list() -> None:
    out = ra._parse_params("Sa, Sq , Sz", all_params=False, quick=False, Surface=_FakeSurface)
    assert out == ["Sa", "Sq", "Sz"]


def test_parse_params_quick_set() -> None:
    out = ra._parse_params(None, all_params=False, quick=True, Surface=_FakeSurface)
    assert out == list(ra._QUICK_PARAMS)


def test_roughness_dict_handles_numpy_scalars() -> None:
    class _NPSurface:
        def roughness_parameters(self, names=None):
            return {"Sa": np.float32(0.123), "Sq": "string", "Sz": None}

    out = ra._roughness_dict(_NPSurface(), names=None)
    assert isinstance(out["Sa"], float)
    assert out["Sq"] == "string"
    assert out["Sz"] is None


def test_sanitize_json_values_drops_nan_inf() -> None:
    raw = {"a": float("nan"), "b": float("inf"), "c": [float("-inf"), 1.0], "d": 2}
    cleaned = ra._sanitize_json_values(raw)
    assert cleaned["a"] is None
    assert cleaned["b"] is None
    assert cleaned["c"][0] is None
    assert cleaned["c"][1] == 1.0
    assert cleaned["d"] == 2


def test_roughness_rows_for_paths_handles_errors(
    monkeypatch: pytest.MonkeyPatch, fake_surfalize_module, tmp_tmd_path: Path
) -> None:
    bad = tmp_tmd_path.parent / "broken.tmd"
    bad.write_bytes(b"not a real tmd")

    def _boom(path, surface):
        raise RuntimeError("can't load")

    monkeypatch.setattr(ra, "_load_surface_for_roughness", _boom)
    rows = ra._roughness_rows_for_paths(
        [bad],
        level=False,
        quick=True,
        params=None,
        all_params=False,
        include_frame_index=True,
        include_full_path=True,
    )
    assert rows[0]["__error__"] == "can't load"
    assert rows[0]["frame"] == 0


def test_file_command_invalid_extension(runner: CliRunner, tmp_path: Path, fake_surfalize_module) -> None:
    bogus = tmp_path / "fixture.txt"
    bogus.write_text("noop")
    app = ra.create_roughness_app()
    r = runner.invoke(app, ["file", str(bogus)])
    assert r.exit_code != 0


def test_file_command_json_output(
    runner: CliRunner,
    tmp_tmd_path: Path,
    fake_surfalize_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ra, "_load_surface_for_roughness", lambda p, S: _FakeSurface())
    app = ra.create_roughness_app()
    r = runner.invoke(app, ["file", str(tmp_tmd_path), "--json", "--quick"])
    assert r.exit_code == 0
    assert "Sa" in r.stdout


def test_batch_command_writes_csv(
    runner: CliRunner,
    tmp_path: Path,
    tmp_tmd_path: Path,
    fake_surfalize_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    in_dir = tmp_path / "tmd_files"
    in_dir.mkdir()
    sample = in_dir / "a.tmd"
    sample.write_bytes(tmp_tmd_path.read_bytes())

    monkeypatch.setattr(ra, "_load_surface_for_roughness", lambda p, S: _FakeSurface())
    out = tmp_path / "results.csv"
    app = ra.create_roughness_app()
    r = runner.invoke(app, ["batch", str(in_dir), "--output", str(out), "--quick"])
    assert r.exit_code == 0
    assert out.exists()


def test_batch_command_no_files_exits_zero(
    runner: CliRunner, tmp_path: Path, fake_surfalize_module
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    app = ra.create_roughness_app()
    r = runner.invoke(app, ["batch", str(empty), "--quick"])
    assert r.exit_code == 0


def test_sequence_command_explicit_paths_json(
    runner: CliRunner,
    tmp_path: Path,
    tmp_tmd_path: Path,
    fake_surfalize_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s1 = tmp_path / "f1.tmd"
    s2 = tmp_path / "f2.tmd"
    s1.write_bytes(tmp_tmd_path.read_bytes())
    s2.write_bytes(tmp_tmd_path.read_bytes())

    monkeypatch.setattr(ra, "_load_surface_for_roughness", lambda p, S: _FakeSurface())
    app = ra.create_roughness_app()
    r = runner.invoke(app, ["sequence", str(s1), str(s2), "--json", "--quick"])
    assert r.exit_code == 0
    assert "frame" in r.stdout


def test_sequence_command_from_dir_invalid_sort(
    runner: CliRunner, tmp_path: Path, fake_surfalize_module
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    app = ra.create_roughness_app()
    r = runner.invoke(app, ["sequence", "--from-dir", str(src), "--sort-by", "bogus"])
    assert r.exit_code != 0


def test_sequence_command_no_paths_exits(runner: CliRunner, fake_surfalize_module) -> None:
    app = ra.create_roughness_app()
    r = runner.invoke(app, ["sequence"])
    assert r.exit_code != 0


def test_sequence_command_dir_and_paths_conflict(
    runner: CliRunner, tmp_path: Path, fake_surfalize_module, tmp_tmd_path: Path
) -> None:
    src = tmp_path / "src"
    src.mkdir()
    app = ra.create_roughness_app()
    r = runner.invoke(app, ["sequence", str(tmp_tmd_path), "--from-dir", str(src)])
    assert r.exit_code != 0
