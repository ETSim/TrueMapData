"""Tests for :mod:`tmd.plotters.visualization_utils`."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from tmd.plotters.visualization_utils import (
    ColorMapRegistry,
    HeightMapAnalyzer,
    TMDVisualizationUtils,
)


def test_colormap_registry_lists_and_register() -> None:
    names = ColorMapRegistry.get_available_cmaps()
    assert "tmd_height" in names
    ColorMapRegistry.register_cmap("zz_test", ["#000000", "#FFFFFF"])
    assert "zz_test" in ColorMapRegistry.get_available_cmaps()
    mc = ColorMapRegistry.create_matplotlib_cmap("tmd_height")
    assert mc is not None
    pl = ColorMapRegistry.create_plotly_cmap("tmd_height")
    assert isinstance(pl, list) and pl
    assert ColorMapRegistry.create_plotly_cmap("unknown_cmap_xyz") is None
    hx = ColorMapRegistry.height_to_color(0.5, 0.0, 1.0, "tmd_height")
    assert isinstance(hx, str) and hx.startswith("#")


def test_height_map_analyzer_stats() -> None:
    h = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    b = HeightMapAnalyzer.compute_basic_stats(h)
    assert b["min"] <= b["max"]
    adv = HeightMapAnalyzer.compute_advanced_stats(h)
    assert "skewness" in adv


@pytest.mark.parametrize("flat", [False, True])
def test_height_map_analyzer_detect_and_profiles_and_gradient(flat: bool) -> None:
    pytest.importorskip("scipy")
    if flat:
        h = np.ones((10, 10), dtype=np.float32)
    else:
        h = np.random.RandomState(1).rand(24, 24).astype(np.float32)
    mask, nfeat = HeightMapAnalyzer.detect_features(h, threshold=0.4, min_size=2)
    assert mask is not None
    prof = HeightMapAnalyzer.compute_profiles(h)
    assert len(prof["horizontal"]) == h.shape[1]
    grad = HeightMapAnalyzer.compute_gradient(h)
    assert grad["magnitude"].shape == h.shape


class _MatplotlibStubPlotter:
    """Class name must contain ``Matplotlib`` for overlay routing."""

    def plot(self, height_map, **kwargs):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        ax.imshow(height_map)
        return ax.figure


def test_create_overlay_plot_matplotlib_path() -> None:
    h = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    ov = np.zeros_like(h)
    fig = TMDVisualizationUtils.create_overlay_plot(_MatplotlibStubPlotter(), h, ov)
    assert fig is not None


def test_overlay_and_multi_view_plotly() -> None:
    pytest.importorskip("plotly.graph_objects")
    from tmd.plotters.plotly import PlotlyHeightMapVisualizer

    h = np.linspace(0, 1, 25, dtype=np.float32).reshape(5, 5)
    ov = np.ones_like(h) * 0.2
    p = PlotlyHeightMapVisualizer()
    fig = TMDVisualizationUtils.create_overlay_plot(p, h, ov, title="ov")
    assert fig is not None
    mv = TMDVisualizationUtils.create_multi_view_plot(p, h, profile_row=2)
    assert mv is not None


def test_overlay_and_multi_view_matplotlib() -> None:
    from tmd.plotters.matplotlib import MatplotlibHeightMapPlotter

    h = np.linspace(0, 1, 25, dtype=np.float32).reshape(5, 5)
    ov = h * 0.5
    p = MatplotlibHeightMapPlotter()
    fig = TMDVisualizationUtils.create_overlay_plot(p, h, ov)
    assert fig is not None
    mv = TMDVisualizationUtils.create_multi_view_plot(p, h)
    assert mv is not None


def test_overlay_unsupported_plotter_fallback() -> None:
    class _Other:
        NAME = "other"

        def plot(self, height_map, **kwargs):
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()
            ax.imshow(height_map)
            return ax.figure

    h = np.ones((4, 4), dtype=np.float32)
    fig = TMDVisualizationUtils.create_overlay_plot(_Other(), h, h)
    assert fig is not None


def test_multi_view_unknown_plotter() -> None:
    class _X:
        pass

    h = np.ones((4, 4), dtype=np.float32)
    assert TMDVisualizationUtils.create_multi_view_plot(_X(), h) is None


def test_height_map_analyzer_profiles_positions() -> None:
    h = np.arange(12, dtype=np.float32).reshape(3, 4)
    p = HeightMapAnalyzer.compute_profiles(h, x_pos=1, y_pos=2)
    assert len(p["horizontal"]) == h.shape[1]
    assert p["x_pos"] == 1
