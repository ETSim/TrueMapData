"""Exercise Seaborn-backed plotters."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

pytest.importorskip("seaborn")

from tmd.plotters.seaborn import (
    SeabornHeightMapPlotter,
    SeabornProfilePlotter,
    SeabornSequencePlotter,
)


@pytest.fixture
def hm() -> np.ndarray:
    rng = np.random.default_rng(2)
    return rng.random((16, 18), dtype=np.float32)


def test_height_map_plotter(tmp_path, hm: np.ndarray) -> None:
    p = SeabornHeightMapPlotter()
    pth = str(tmp_path / "hm.png")
    fig = p.plot(hm, filename=pth, show=False)
    assert fig is not None

    fig_e = p.plot(hm, enhanced=True, filename=str(tmp_path / "en.png"), show=False)
    assert fig_e is not None

    fig_3 = p.plot(hm, mode="3d", title="t", show=False)
    assert fig_3 is not None

    fig_pr = p.plot_2d(hm, partial_range=(0, 8, 0, 9), show=False)
    assert fig_pr is not None

    out = tmp_path / "save.png"
    assert p.save(fig, str(out), close=False) == str(out)


def test_sequence_and_profile(tmp_path, hm: np.ndarray) -> None:
    sp = SeabornSequencePlotter()
    frames = [hm, hm * 0.9]
    assert sp.visualize_sequence([], show=False) is not None
    many = [hm + i * 0.01 for i in range(10)]
    fig_s = sp.visualize_sequence(many, n_frames=4, show=False)
    assert fig_s is not None

    fig_a = sp.create_animation(frames, show=False)
    assert fig_a is not None

    st = sp.visualize_statistics(
        {"rms": [1.0, 1.1], "timestamps": [0, 1]},
        style="bar",
        show=False,
    )
    assert st is not None
    assert sp.visualize_statistics({}, show=False) is not None

    prof = SeabornProfilePlotter()
    fig_p = prof.plot_profile(np.linspace(0, 1, 20), title="row", fill=False, show_markers=True)
    assert fig_p is not None

    fig_d = prof.plot_height_distribution(hm, kde=False, bins=10, show_stats=False)
    assert fig_d is not None

    fig_pc = prof.plot_profile_comparison(
        [hm[4, :], hm[5, :]],
        labels=["a", "b"],
    )
    assert fig_pc is not None
