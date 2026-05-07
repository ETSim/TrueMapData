"""Extra coverage for tmd.plotters.factory error and edge paths."""

from __future__ import annotations

from typing import ClassVar, List

import pytest

from tmd.plotters import factory as factory_mod
from tmd.plotters.base import BasePlotter, BaseSequencePlotter
from tmd.plotters.factory import (
    TMDPlotterFactory,
    TMDSequencePlotterFactory,
    get_best_plotter,
    get_registered_plotters,
    get_available_plotters,
)


class _PlotterMissingDep(BasePlotter):
    """Plotter class whose dependencies cannot be imported."""

    NAME = "missing_dep"
    REQUIRED_DEPENDENCIES: ClassVar[List[str]] = ["this_module_should_not_exist_xyz123"]

    def plot(self, height_map, **kwargs):  # pragma: no cover - shouldn't be reached
        raise RuntimeError("plot should not be called for unavailable plotter")

    def save(self, plot_obj, filename, **kwargs):  # pragma: no cover
        raise RuntimeError("save should not be called for unavailable plotter")

    def __init__(self) -> None:
        # Force ImportError during instantiation
        raise ImportError("simulated missing dependency")


class _DummyPlotter(BasePlotter):
    NAME = "dummy"

    def plot(self, height_map, **kwargs):
        return "dummy"

    def save(self, plot_obj, filename, **kwargs):
        return filename


class _DummySequencePlotter(BaseSequencePlotter):
    NAME = "dummy_seq"

    def visualize_sequence(self, frames, **kwargs):
        return "seq"

    def create_animation(self, frames, **kwargs):
        return "anim"

    def visualize_statistics(self, stats_data, **kwargs):
        return "stats"

    def save_figure(self, fig, filename, **kwargs):
        return filename


def test_register_rejects_non_plotter_class() -> None:
    class _NotAPlotter:
        pass

    with pytest.raises(TypeError):
        TMDPlotterFactory.register("not_plotter", _NotAPlotter)


def test_create_plotter_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="not registered"):
        TMDPlotterFactory.create_plotter("never_registered_plotter_xyz")


def test_create_plotter_missing_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover the ImportError -> ValueError translation in create_plotter."""
    orig = TMDPlotterFactory._plotter_registry.copy()
    try:
        TMDPlotterFactory._plotter_registry = dict(orig)
        TMDPlotterFactory.register("missing_dep", _PlotterMissingDep)
        with pytest.raises(ValueError, match="dependencies"):
            TMDPlotterFactory.create_plotter("missing_dep")
    finally:
        TMDPlotterFactory._plotter_registry = orig


def test_list_available_strategies_returns_dict_of_bools() -> None:
    res = TMDPlotterFactory.list_available_strategies()
    assert isinstance(res, dict)
    assert all(isinstance(v, bool) for v in res.values())


def test_get_registered_plotters_method_returns_names() -> None:
    names = TMDPlotterFactory.get_registered_plotters()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)


def test_factory_get_available_plotters_method() -> None:
    avail = TMDPlotterFactory.get_available_plotters()
    assert isinstance(avail, list)


def test_get_best_plotter_no_match_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(factory_mod, "get_registered_plotters", lambda: {})
    assert get_best_plotter(preference_order=["nope"]) is None


def test_get_best_plotter_default_order(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        factory_mod,
        "get_registered_plotters",
        lambda: {"plotly": False, "matplotlib": True, "seaborn": False},
    )

    def _fake_create(name):
        return f"created_{name}"

    monkeypatch.setattr(TMDPlotterFactory, "create_plotter", classmethod(lambda cls, name: f"created_{name}"))
    res = get_best_plotter()
    assert res == "created_matplotlib"


def test_sequence_factory_register_rejects_non_sequence_plotter() -> None:
    class _NotASequence:
        pass

    with pytest.raises(TypeError):
        TMDSequencePlotterFactory.register("bogus", _NotASequence)


def test_sequence_factory_create_plotter_unknown_name() -> None:
    with pytest.raises(ValueError, match="not registered"):
        TMDSequencePlotterFactory.create_plotter("missing_seq_plotter_xyz")


def test_sequence_factory_register_and_create() -> None:
    orig = TMDSequencePlotterFactory._plotter_registry.copy()
    try:
        TMDSequencePlotterFactory._plotter_registry = dict(orig)
        TMDSequencePlotterFactory.register("dummy_seq", _DummySequencePlotter)
        inst = TMDSequencePlotterFactory.create_plotter("dummy_seq")
        assert isinstance(inst, _DummySequencePlotter)
    finally:
        TMDSequencePlotterFactory._plotter_registry = orig


def test_sequence_factory_list_available_strategies() -> None:
    res = TMDSequencePlotterFactory.list_available_strategies()
    assert isinstance(res, dict)


def test_sequence_factory_create_plotter_missing_dependencies() -> None:
    class _SeqWithBadDep(BaseSequencePlotter):
        REQUIRED_DEPENDENCIES: ClassVar[List[str]] = ["this_module_should_not_exist_xyz123"]

        def __init__(self) -> None:
            raise ImportError("simulated")

        def visualize_sequence(self, frames, **kwargs):
            return None

        def create_animation(self, frames, **kwargs):
            return None

        def visualize_statistics(self, stats_data, **kwargs):
            return None

        def save_figure(self, fig, filename, **kwargs):
            return filename

    orig = TMDSequencePlotterFactory._plotter_registry.copy()
    try:
        TMDSequencePlotterFactory._plotter_registry = dict(orig)
        TMDSequencePlotterFactory.register("bad_seq", _SeqWithBadDep)
        with pytest.raises(ValueError, match="dependencies"):
            TMDSequencePlotterFactory.create_plotter("bad_seq")
    finally:
        TMDSequencePlotterFactory._plotter_registry = orig
