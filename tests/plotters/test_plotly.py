"""Exercise Plotly plotters for coverage (no GUI)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plotly.graph_objects")

from tmd.plotters.plotly import PlotlyHeightMapVisualizer, PlotlySequenceVisualizer


@pytest.fixture
def hm() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.random((12, 14), dtype=np.float32)


def test_height_map_modes_and_save(tmp_path, hm: np.ndarray) -> None:
    viz = PlotlyHeightMapVisualizer()
    for mode in ("2d", "3d", "contour", "profile", "slider"):
        fig = viz.plot(hm, mode=mode, show=False, profile_row=min(3, hm.shape[0] - 1))
        assert fig is not None

    fig2 = viz.plot_2d(hm, show=False)
    fig3 = viz.plot_3d(hm, show=False)
    fig4 = viz.plot_profile(hm, profile_row=2, show=False)
    assert fig2 and fig3 and fig4

    bad_row = viz.plot(hm, mode="profile", profile_row=hm.shape[0] + 10, show=False)
    assert bad_row is not None

    sub = viz.plot(hm, partial_range=(0, 6, 0, 7), show=False)
    assert sub is not None

    prof_mm = viz.plot(
        hm,
        mode="profile",
        profile_row=2,
        x_length=10.0,
        x_offset=1.0,
        show_markers=False,
        show_grid=False,
        show=False,
    )
    assert prof_mm is not None

    html_out = tmp_path / "p.html"
    assert viz.save(fig2, str(html_out)) == str(html_out)


def test_sequence_visualizer(tmp_path, hm: np.ndarray) -> None:
    seq = PlotlySequenceVisualizer()
    frames = [hm, hm * 0.9, hm * 0.8]
    many = [hm + i * 0.01 for i in range(10)]

    assert seq.visualize_sequence([], show=False) is not None
    assert seq.visualize_sequence(frames, mode="2d", show=False) is not None
    assert seq.visualize_sequence(frames, mode="3d", show=False) is not None
    assert seq.visualize_sequence(many, n_frames=3, show=False) is not None

    assert seq.create_animation([], show=False) is not None
    assert seq.create_animation(frames, mode="2d", fps=5, show=False) is not None
    assert seq.create_animation(frames, mode="3d", show=False) is not None

    fig_st = seq.visualize_statistics(
        {
            "timestamps": [0, 1, 2],
            "mean": [1.0, 1.1, 1.2],
            "std": [0.1, 0.2, 0.15],
            "min": [0.5, 0.6, 0.55],
            "max": [2.0, 2.1, 2.05],
            "custom": [3.0, 3.1, 3.2],
        },
        show=False,
    )
    assert fig_st is not None

    empty_st = seq.visualize_statistics({}, show=False)
    assert empty_st is not None

    bad_metrics = seq.visualize_statistics({"timestamps": [0]}, metrics=["nope"], show=False)
    assert bad_metrics is not None

    out = tmp_path / "seq.html"
    assert seq.save_figure(fig_st, str(out)) == str(out)
