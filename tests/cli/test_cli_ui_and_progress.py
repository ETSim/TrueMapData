"""Tests for ``tmd.cli.core.ui`` and ``tmd.cli.utils.progress``."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tmd.cli.core import ui as ui_mod
from tmd.cli.utils import progress as prog_mod


def test_print_messages_smoke() -> None:
    ui_mod.print_warning("w")
    ui_mod.print_error("e")
    ui_mod.print_success("s")
    ui_mod.print_info("i")


def test_print_rich_table_with_rows() -> None:
    ui_mod.print_rich_table(
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}],
        title="T",
        columns=[("a", "cyan"), ("b", "green")],
    )


def test_print_rich_table_fallback_on_table_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):
        raise TypeError("no rich table")

    monkeypatch.setattr(ui_mod, "Table", boom)
    ui_mod.print_rich_table([{"k": "v"}], title="F")


def test_display_metadata() -> None:
    ui_mod.display_metadata({"width": 10, "z": 1.23456789, "vec": (1.0, 2.0)})


def test_format_height_map_summary() -> None:
    assert "not available" in ui_mod.format_height_map_summary(None)
    hm = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    s = ui_mod.format_height_map_summary(hm)
    assert "Dimensions" in s and "Mean" in s


def test_progress_context_rich_path(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyProgress:
        def __init__(self, *a, **k):
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

        def add_task(self, *a, **k):
            return 1

        def update(self, *a, **k):
            pass

        def advance(self, *a, **k):
            pass

    monkeypatch.setattr(ui_mod, "Progress", DummyProgress)
    with ui_mod.ProgressContext("Job", total=3) as ctx:
        ctx.update(advance=1, description="step")


def test_progress_context_no_rich_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui_mod, "Progress", lambda *a, **k: (_ for _ in ()).throw(ImportError("no")))
    with ui_mod.ProgressContext("X") as ctx:
        ctx.update(advance=1, description="y")


def test_display_map_export_info_and_results() -> None:
    ui_mod.display_map_export_info(
        Path("in.tmd"),
        Path("out"),
        ["normal", "ao"],
        {"format": "png", "compress": 3, "fast": True},
    )
    ui_mod.display_map_export_results(
        {
            "normal": {"success": True, "path": str(Path("/tmp/a.png")), "time": 0.1},
            "ao": False,
        }
    )


def test_display_batch_progress() -> None:
    ui_mod.display_batch_progress({"processed": 2, "total": 5, "successful": 2, "failed": 0})


def test_display_single_map_export_result() -> None:
    ui_mod.display_single_map_export_result(True, Path("/tmp/m.png"))
    ui_mod.display_single_map_export_result(False, Path("/tmp/m.png"))


def test_display_tmd_info() -> None:
    d = SimpleNamespace(metadata={"w": 3})
    ui_mod.display_tmd_info(d)
    ui_mod.display_tmd_info(SimpleNamespace())


def test_print_tmd_info_table_custom_console() -> None:
    c = SimpleNamespace(print=lambda *a, **k: None)
    d = SimpleNamespace(height_map=np.zeros((2, 3), dtype=np.float32), metadata={"k": 1})
    ui_mod.print_tmd_info_table(d, console=c)


def test_progress_create_update_close_tqdm() -> None:
    bar = prog_mod.create_progress_bar(3, "x", use_rich=False)
    assert bar["type"] == "tqdm"
    prog_mod.update_progress(bar, n=1)
    prog_mod.close_progress(bar)


def test_progress_iterator_tqdm() -> None:
    out = list(prog_mod.progress_iterator([1, 2, 3], "iter", use_rich=False))
    assert out == [1, 2, 3]


def test_spinner_context_no_rich() -> None:
    with prog_mod.spinner_context("spin", use_rich=False) as sp:
        sp.update("next")


def test_process_with_progress_empty() -> None:
    r = prog_mod.process_with_progress([], lambda x: x, "d")
    assert r["total"] == 0


def test_process_with_progress_handles_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    _orig_create = prog_mod.create_progress_bar

    def tqdm_bar(total: int, description: str, **kwargs):
        return _orig_create(total, description, use_rich=False)

    monkeypatch.setattr(prog_mod, "create_progress_bar", tqdm_bar)
    errors: list[tuple] = []

    def fail(x):
        if x.name == "b":
            raise ValueError("bad")
        return x.name

    def handle(item, exc):
        errors.append((item, exc))

    items = [SimpleNamespace(name="a"), SimpleNamespace(name="b"), SimpleNamespace(name="c")]
    r = prog_mod.process_with_progress(items, fail, "p", error_handler=handle)
    assert r["failed"] == 1 and r["success"] == 2
    assert errors and errors[0][0].name == "b"


def test_file_progress_bar_start_stop() -> None:
    bar = prog_mod.file_progress_bar(1000, description="dl")
    assert bar["type"] == "rich"
    prog_mod.update_progress(bar, n=500)
    prog_mod.close_progress(bar)


def test_create_progress_bar_rich_start_stop() -> None:
    bar = prog_mod.create_progress_bar(2, "richjob", use_rich=True)
    assert bar["type"] == "rich"
    prog_mod.update_progress(bar, n=1, description="step2")
    prog_mod.close_progress(bar)


def test_progress_iterator_rich() -> None:
    out = list(prog_mod.progress_iterator([10, 20], "it", use_rich=True))
    assert out == [10, 20]


def test_spinner_context_rich() -> None:
    with prog_mod.spinner_context("sp", use_rich=True) as sp:
        sp.update("go")
