#!/usr/bin/env python3
"""Tests for plotter factory helpers and registration."""

import numpy as np

from tmd.plotters.factory import (
    TMDPlotterFactory,
    get_registered_plotters,
    get_available_plotters,
    get_best_plotter,
)
from tmd.plotters.base import BasePlotter


class _TmpPlotter(BasePlotter):
    NAME = "tmpplotter"

    def plot(self, height_map, **kwargs):
        return "tmp"

    def save(self, plot_obj, filename, **kwargs):
        return filename


def test_get_registered_plotters_shape():
    d = get_registered_plotters()
    assert isinstance(d, dict)
    assert all(isinstance(v, bool) for v in d.values())


def test_get_available_plotters_subset_of_keys():
    names = get_available_plotters()
    assert isinstance(names, list)
    reg = get_registered_plotters()
    for n in names:
        assert n in reg


def test_get_best_plotter_returns_plotter_or_none():
    p = get_best_plotter(["matplotlib", "plotly", "seaborn"])
    assert p is None or hasattr(p, "plot")


def test_register_under_isolated_registry():
    orig = TMDPlotterFactory._plotter_registry.copy()
    try:
        TMDPlotterFactory._plotter_registry = dict(orig)
        TMDPlotterFactory.register("tmpplotter", _TmpPlotter)
        inst = TMDPlotterFactory.create_plotter("tmpplotter")
        assert inst.plot(np.zeros((3, 3))) == "tmp"
    finally:
        TMDPlotterFactory._plotter_registry = orig
