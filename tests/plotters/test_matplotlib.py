"""Exercise Matplotlib plotters with Agg backend."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tmd.plotters.matplotlib import MatplotlibHeightMapPlotter, MatplotlibSequencePlotter


@pytest.fixture
def hm() -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.random((10, 12), dtype=np.float32)


def test_height_map_plot_modes_and_save(tmp_path, hm: np.ndarray) -> None:
    p = MatplotlibHeightMapPlotter()
    for mode in ("2d", "3d", "contour", "profile"):
        fig = p.plot(hm, mode=mode, show=False, profile_row=4)
        assert fig is not None

    fig_pr = p.plot(hm, mode="profile", profile_row=hm.shape[0] + 99, show=False)
    assert fig_pr is not None

    sub = p.plot(hm, partial_range=(0, 5, 0, 6), mode="2d", show=False)
    assert sub is not None

    out = tmp_path / "m.png"
    assert p.save(fig_pr, str(out), close=True) == str(out)


def test_sequence_plotter(tmp_path, hm: np.ndarray) -> None:
    sp = MatplotlibSequencePlotter()
    frames = [hm, hm * 0.5]
    long_frames = [hm + i * 0.01 for i in range(8)]

    assert sp.visualize_sequence([], show=False) is not None
    assert sp.visualize_sequence(frames, layout="row", show=False) is not None
    assert sp.visualize_sequence(long_frames, n_frames=3, layout="grid", show=False) is not None
    assert sp.visualize_sequence(long_frames, layout="column", frame_indices=[0, 2], show=False) is not None
    assert sp.visualize_sequence(long_frames, frame_indices=[99], show=False) is not None

    anim = sp.create_animation(frames, interval=50, show=False)
    assert anim is not None

    stats_fig = sp.visualize_statistics(
        {"mean": [1.0, 1.1], "std": [0.1, 0.2], "timestamps": [0, 1]},
        style="line",
        show=False,
    )
    assert stats_fig is not None

    empty = sp.visualize_statistics({}, show=False)
    assert empty is not None

    fig3 = sp.visualize_sequence(frames, mode="3d", show=False)
    assert fig3 is not None

    out = tmp_path / "seq_stat.png"
    assert sp.save_figure(stats_fig, str(out)) == str(out)
