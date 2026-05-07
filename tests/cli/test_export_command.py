"""Tests for tmd.cli.commands.export helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tmd.cli.commands import export as export_module


class _DummyProgress:
    """No-op stand-in for rich.progress.Progress in export_maps_command."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def add_task(self, *args, **kwargs):
        return 0

    def update(self, *args, **kwargs):
        pass

    def advance(self, *args, **kwargs):
        pass


def test_default_material_slot_types() -> None:
    slots = export_module.default_material_slot_types()
    assert slots == {
        "map_kd": "height",
        "map_bump": "normal",
        "map_disp": "displacement",
        "map_pr": "roughness",
    }


def test_export_command_map_format_calls_map_exporter(monkeypatch) -> None:
    called: dict = {}

    def fake_export(tmd_file, output, format, **kwargs):
        called["args"] = (tmd_file, output, format, kwargs)

    monkeypatch.setattr(
        export_module.MapExporter, "export", staticmethod(fake_export), raising=False
    )
    monkeypatch.setattr(export_module, "get_available_map_types", lambda: ["pngmap"])
    monkeypatch.setattr(export_module, "get_available_formats", lambda: ["stl"])

    tmd = Path("in.tmd")
    out = Path("out.png")
    assert export_module.export_command(tmd, out, "pngmap", foo=1) is True
    assert called["args"][0] == tmd
    assert called["args"][1] == out
    assert called["args"][2] == "pngmap"
    assert called["args"][3] == {"foo": 1}


def test_export_command_model_format_calls_model_exporter(monkeypatch) -> None:
    called: dict = {}

    def fake_export(cls, tmd_file, output, format, **kwargs):
        called["hit"] = (str(tmd_file), str(output) if output else None, format)

    monkeypatch.setattr(
        export_module.MapExporter,
        "export",
        staticmethod(lambda *a, **k: None),
        raising=False,
    )
    monkeypatch.setattr(export_module, "get_available_map_types", lambda: [])
    monkeypatch.setattr(export_module, "get_available_formats", lambda: ["stl"])
    monkeypatch.setattr(
        export_module.ModelExporter, "export", classmethod(fake_export), raising=False
    )

    assert export_module.export_command(Path("a.tmd"), Path("m.stl"), "stl") is True
    assert called["hit"][2] == "stl"


def test_export_command_unknown_format_returns_false(monkeypatch) -> None:
    monkeypatch.setattr(export_module, "get_available_map_types", lambda: [])
    monkeypatch.setattr(export_module, "get_available_formats", lambda: ["stl"])
    assert export_module.export_command(Path("a.tmd"), None, "not_real") is False


def test_export_command_exception_returns_false(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(export_module, "get_available_map_types", lambda: ["m"])
    monkeypatch.setattr(export_module, "get_available_formats", lambda: [])
    monkeypatch.setattr(export_module.MapExporter, "export", staticmethod(boom), raising=False)

    assert export_module.export_command(Path("a.tmd"), Path("o"), "m") is False


def test_export_maps_command_success_single_map_type(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    inp = tmp_path / "in.tmd"
    inp.write_bytes(b"x")
    out_dir = tmp_path / "textures"

    hm = np.zeros((3, 4), dtype=np.float32)

    def fake_load(cls, filepath, compute_initial_stats=False):
        return SimpleNamespace(metadata={"k": "v"}, height_map=hm)

    monkeypatch.setattr(export_module.TMD, "load", classmethod(fake_load))
    monkeypatch.setattr(export_module, "Progress", _DummyProgress)

    export_calls: list = []

    def fake_export_map(*args, **kwargs):
        export_calls.append({"args": args, "kwargs": kwargs})
        return True

    monkeypatch.setattr(export_module.MapExporter, "export_map", staticmethod(fake_export_map), raising=False)

    ok = export_module.export_maps_command(
        inp,
        out_dir,
        ["height"],
        compress=50,
        format="png",
        normalize=False,
        metadata={"extra": "m"},
    )
    assert ok is True
    expected_png = out_dir / "in_height.png"
    assert expected_png.parent.is_dir()
    assert export_calls
    assert export_calls[0]["kwargs"]["compress"] == 50
    assert export_calls[0]["kwargs"]["format"] == "png"
    assert export_calls[0]["kwargs"]["normalize"] is False
    merged = export_calls[0]["kwargs"]["metadata"]
    assert merged["k"] == "v"
    assert merged["extra"] == "m"


def test_export_maps_command_uses_default_textures_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inp = tmp_path / "nested" / "in.tmd"
    inp.parent.mkdir(parents=True)
    inp.write_bytes(b"x")

    hm = np.zeros((2, 3), dtype=np.float32)

    def fake_load(cls, filepath, compute_initial_stats=False):
        return SimpleNamespace(metadata={}, height_map=hm)

    monkeypatch.setattr(export_module.TMD, "load", classmethod(fake_load))
    monkeypatch.setattr(export_module, "Progress", _DummyProgress)
    monkeypatch.setattr(export_module.console, "print", lambda *a, **k: None)

    export_calls: list = []

    def fake_export_map(*args, **kwargs):
        export_calls.append({"args": args, "kwargs": kwargs})
        return True

    monkeypatch.setattr(export_module.MapExporter, "export_map", staticmethod(fake_export_map), raising=False)

    assert export_module.export_maps_command(inp, None, ["height"]) is True
    expected = inp.parent / "textures" / "in_height.png"
    assert expected.parent.is_dir()
    out_arg = Path(export_calls[0]["args"][1])
    assert out_arg == expected


def test_export_maps_command_fast_parallax_ao_clamps_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inp = tmp_path / "in.tmd"
    inp.write_bytes(b"x")
    out_dir = tmp_path / "out"
    hm = np.ones((2, 2), dtype=np.float32)

    def fake_load(cls, filepath, compute_initial_stats=False):
        return SimpleNamespace(metadata={}, height_map=hm)

    monkeypatch.setattr(export_module.TMD, "load", classmethod(fake_load))
    monkeypatch.setattr(export_module, "Progress", _DummyProgress)
    monkeypatch.setattr(export_module.console, "print", lambda *a, **k: None)

    export_calls: list = []

    def fake_export_map(*args, **kwargs):
        export_calls.append(kwargs)
        return True

    monkeypatch.setattr(export_module.MapExporter, "export_map", staticmethod(fake_export_map), raising=False)

    assert (
        export_module.export_maps_command(
            inp,
            out_dir,
            ["parallax_ao"],
            fast=True,
            samples=99,
            max_distance=0.1,
        )
        is True
    )
    kw = export_calls[0]
    assert kw["samples"] <= 8
    assert kw["multi_scale"] is False
    assert kw["max_distance"] <= 0.03


def test_export_maps_command_fast_excludes_parallax_when_types_auto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inp = tmp_path / "in.tmd"
    inp.write_bytes(b"x")
    hm = np.zeros((2, 2), dtype=np.float32)

    def fake_load(cls, filepath, compute_initial_stats=False):
        return SimpleNamespace(metadata={}, height_map=hm)

    monkeypatch.setattr(export_module.TMD, "load", classmethod(fake_load))
    monkeypatch.setattr(export_module, "Progress", _DummyProgress)
    monkeypatch.setattr(export_module.console, "print", lambda *a, **k: None)
    monkeypatch.setattr(
        export_module,
        "get_available_map_types",
        lambda: ["height", "parallax_ao"],
    )

    map_types_called: list = []

    def fake_export_map(hm_arg, out_path, map_type, **kwargs):
        map_types_called.append(map_type)
        return True

    monkeypatch.setattr(
        export_module.MapExporter, "export_map", staticmethod(fake_export_map), raising=False
    )

    assert export_module.export_maps_command(inp, tmp_path / "textures", None, fast=True) is True
    assert map_types_called == ["height"]


def test_display_config_info_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []
    row_cells: list = []

    class RecordingTable(export_module.Table):
        def add_row(self, *args, **kwargs):
            row_cells.append(tuple(args))
            return super().add_row(*args, **kwargs)

    monkeypatch.setattr(export_module, "Table", RecordingTable)

    def capture_print(*args, **kwargs):
        for a in args:
            captured.append(str(a))

    monkeypatch.setattr(export_module.console, "print", capture_print)

    export_module.display_config_info(
        Path("a.tmd"), Path("out"), ["height"], {"x": 1}
    )

    flat = " ".join(" ".join(str(c) for c in row) for row in row_cells)
    assert "Input File" in flat and "a.tmd" in flat
    assert "Output Directory" in flat and "out" in flat
    assert "Map Types" in flat and "height" in flat
    assert "Parameters" in flat and "x" in flat
    assert "Export Configuration" in "\n".join(captured)


def test_export_maps_command_includes_strength_in_params(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inp = tmp_path / "in.tmd"
    inp.write_bytes(b"x")
    out_dir = tmp_path / "out"
    hm = np.zeros((2, 2), dtype=np.float32)

    captured: dict = {}

    def fake_load(cls, filepath, compute_initial_stats=False):
        return SimpleNamespace(metadata={}, height_map=hm)

    def grab_display_config(input_file, output_dir, types, params):
        captured["params"] = params

    monkeypatch.setattr(export_module.TMD, "load", classmethod(fake_load))
    monkeypatch.setattr(export_module, "Progress", _DummyProgress)
    monkeypatch.setattr(export_module.console, "print", lambda *a, **k: None)
    monkeypatch.setattr(export_module, "display_config_info", grab_display_config)
    monkeypatch.setattr(
        export_module.MapExporter,
        "export_map",
        staticmethod(lambda *a, **k: True),
        raising=False,
    )

    assert export_module.export_maps_command(inp, out_dir, ["height"], strength=2.0) is True
    assert captured["params"]["strength"] == 2.0
    assert captured["params"]["fast"] is False


def test_export_maps_command_failure_returns_false(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    inp = tmp_path / "bad.tmd"
    inp.write_bytes(b"y")

    def boom(cls, filepath, compute_initial_stats=False):
        raise RuntimeError("load failed")

    monkeypatch.setattr(export_module.TMD, "load", classmethod(boom))
    monkeypatch.setattr(export_module, "Progress", _DummyProgress)
    monkeypatch.setattr(export_module, "print_error", lambda *a, **k: None)

    assert (
        export_module.export_maps_command(inp, tmp_path / "out", ["height"]) is False
    )


def test_display_export_results_prints_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    lines: list = []
    row_cells: list = []

    class RecordingTable(export_module.Table):
        def add_row(self, *args, **kwargs):
            row_cells.append(tuple(args))
            return super().add_row(*args, **kwargs)

    monkeypatch.setattr(export_module, "Table", RecordingTable)

    def capture_print(*args, **kwargs):
        for a in args:
            lines.append(str(a))

    monkeypatch.setattr(export_module.console, "print", capture_print)

    ok_path = tmp_path / "in_height.png"
    results = {
        "height": {"success": True, "path": ok_path, "time": 0.12},
        "normal": {"success": False, "path": None, "time": 0.03},
    }
    export_module.display_export_results(results, total_time=1.5)

    flat = " ".join(" ".join(str(c) for c in row) for row in row_cells)
    assert "height" in flat and "normal" in flat
    assert "OK" in flat and "FAIL" in flat
    text = "\n".join(lines)
    assert "Total processing time" in text
    assert "1.50" in text
