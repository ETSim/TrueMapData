"""Extra coverage for :class:`TMDProcessor` and :class:`TMD` in :mod:`tmd.core.tmd`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
import numpy as np
import pytest

from tmd.core.tmd import TMD, TMDProcessor
from tmd.exceptions import TMDProcessingError


@pytest.fixture(autouse=True)
def _agg_backend() -> None:
    matplotlib.use("Agg", force=True)


def test_tmd_processor_chain_and_data_methods(tmp_tmd_path: Path) -> None:
    p = TMDProcessor(tmp_tmd_path)
    assert p.set_debug(True) is p
    assert p.set_default_plotter("matplotlib") is p

    with pytest.raises(ValueError):
        p.set_default_plotter("___not_a_plotter___")

    hdr = p.print_file_header()
    assert "width" in hdr and "height" in hdr and isinstance(hdr["version"], int)

    out = p.process()
    assert "metadata" in out and out["height_map"] is not None

    p2 = TMDProcessor(tmp_tmd_path)
    p2.process(force_offset=(0.0, 0.0))

    meta_path = tmp_tmd_path.with_suffix(".meta_out.txt")
    exported = p.export_metadata(meta_path)
    assert Path(exported).exists()

    stats = p.get_stats()
    assert isinstance(stats, dict) and stats

    assert p.get_metadata()
    hm = p.get_height_map()
    assert hm is not None and hm.ndim == 2

    res = p.analyze_defects()
    assert res is not None

    raw = p.load()
    assert raw["height_map"] is not None

    assert "TMDProcessor" in str(p) and "version=" in repr(p)


def test_tmd_processor_plots_save(tmp_path: Path, tmp_tmd_path: Path) -> None:
    p = TMDProcessor(tmp_tmd_path)
    p.process()
    fig2d = p.plot(plotter_strategy="matplotlib", mode="2d", output_path=tmp_path / "p2d.png")
    assert fig2d is not None
    prof = p.plot_profile(row_index=0, plotter_strategy="matplotlib", output_path=tmp_path / "prof.png")
    assert prof is not None
    stats_fig = p.plot_stats(plotter_strategy="matplotlib", output_path=tmp_path / "stats.png")
    assert stats_fig is not None


def test_tmd_crop_valid_and_invalid(small_heightmap: np.ndarray) -> None:
    t = TMD(small_heightmap, {"comment": "c"})
    c = t.crop(0, 0, 2, 2)
    assert c.shape == (2, 2)
    assert "cropped" in (c.metadata.get("comment") or "")
    with pytest.raises(ValueError):
        t.crop(0, 0, 99, 99)


def test_tmd_plot_stats_stub_sequence_plotter(monkeypatch: pytest.MonkeyPatch, small_heightmap: np.ndarray) -> None:
    calls: list = []

    class StubSeq:
        def visualize_statistics(self, stats_data, **kwargs):
            calls.append(stats_data)
            return "stats-ok"

        def save_figure(self, fig, path, **kwargs):
            calls.append(("save", path))

    def fake_create(strategy: str):
        return StubSeq()

    monkeypatch.setattr(
        "tmd.core.tmd.TMDSequencePlotterFactory.create_plotter",
        fake_create,
    )
    t = TMD(small_heightmap, {"comment": "stub"})
    assert t.plot_stats(output_path=Path("stats_out.png")) == "stats-ok"
    assert calls and calls[0]


def test_tmd_processor_plot_failure_wraps(tmp_tmd_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = TMDProcessor(tmp_tmd_path)
    p.process()

    def boom(*a, **k):
        raise RuntimeError("plot fail")

    monkeypatch.setattr("tmd.core.tmd.TMDPlotterFactory.create_plotter", lambda s: MagicMock(plot=boom))
    with pytest.raises(TMDProcessingError):
        p.plot(plotter_strategy="matplotlib")
