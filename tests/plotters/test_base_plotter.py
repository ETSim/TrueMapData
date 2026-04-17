#!/usr/bin/env python3
"""Tests for base plotter classes and TMD plotter factories."""

import pytest
import numpy as np

from tmd.plotters.base import BasePlotter, BaseSequencePlotter
from tmd.plotters.factory import TMDPlotterFactory, TMDSequencePlotterFactory


class MockPlotter(BasePlotter):
    NAME = "mock"

    def plot(self, height_map: np.ndarray, **kwargs):
        return {"type": "plot", "height_map": height_map, "options": kwargs}

    def save(self, plot_obj, filename: str, **kwargs):
        return filename


class MockSequencePlotter(BaseSequencePlotter):
    NAME = "mock_seq"

    def visualize_sequence(self, frames, **kwargs):
        return {"type": "sequence", "frames": frames, "options": kwargs}

    def create_animation(self, frames, **kwargs):
        return {"type": "animation", "frames": frames, "options": kwargs}

    def visualize_statistics(self, stats_data, **kwargs):
        return {"type": "stats", "data": stats_data, "options": kwargs}

    def save_figure(self, fig, filename: str, **kwargs):
        return filename


class FailingMockPlotter(BasePlotter):
    REQUIRED_DEPENDENCIES = ["nonexistent_module_xyz123"]

    def __init__(self):
        super().__init__()

    def plot(self, height_map: np.ndarray, **kwargs):
        return {}

    def save(self, plot_obj, filename: str, **kwargs):
        return filename


class TestBasePlotter:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BasePlotter()

    def test_mock_implementation(self):
        plotter = MockPlotter()
        height_map = np.zeros((10, 10))
        result = plotter.plot(height_map, option1="value1")
        assert result["type"] == "plot"
        assert result["height_map"] is height_map
        assert plotter.save(result, "test.png") == "test.png"


class TestBaseSequencePlotter:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            BaseSequencePlotter()

    def test_mock_sequence_implementation(self):
        plotter = MockSequencePlotter()
        frames = [np.zeros((10, 10)), np.ones((10, 10))]
        out = plotter.visualize_sequence(frames)
        assert out["type"] == "sequence"
        assert plotter.save_figure(out, "x.png") == "x.png"


class TestTMDPlotterFactory:
    def setup_method(self):
        self._plot = TMDPlotterFactory._plotter_registry.copy()
        self._seq = TMDSequencePlotterFactory._plotter_registry.copy()

    def teardown_method(self):
        TMDPlotterFactory._plotter_registry = self._plot
        TMDSequencePlotterFactory._plotter_registry = self._seq

    def test_register_create_list(self):
        TMDPlotterFactory._plotter_registry = {}
        TMDPlotterFactory.register("mockunit", MockPlotter)
        plotter = TMDPlotterFactory.create_plotter("mockunit")
        assert isinstance(plotter, MockPlotter)
        avail = TMDPlotterFactory.list_available_strategies()
        assert avail.get("mockunit") is True
        with pytest.raises(ValueError, match="not registered"):
            TMDPlotterFactory.create_plotter("missing_backend_xyz")

    def test_sequence_factory_register(self):
        TMDSequencePlotterFactory._plotter_registry = {}
        TMDSequencePlotterFactory.register("mocksequnit", MockSequencePlotter)
        sp = TMDSequencePlotterFactory.create_plotter("mocksequnit")
        assert isinstance(sp, MockSequencePlotter)
