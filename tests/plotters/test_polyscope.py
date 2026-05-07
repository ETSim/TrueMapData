"""Polyscope plotter: guard when optional dependency is absent."""

from __future__ import annotations

import pytest


def test_polyscope_raises_when_marked_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import tmd.plotters.polyscope as pm

    monkeypatch.setattr(pm, "HAS_POLYSCOPE", False)
    from tmd.plotters.polyscope import PolyscopePlotter

    with pytest.raises(ImportError):
        PolyscopePlotter()
