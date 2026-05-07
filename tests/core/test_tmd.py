"""Tests for :mod:`tmd.core.tmd` (``TMD`` and helpers)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tmd.core import tmd as tmd_module
from tmd.core.tmd import TMD, TMDProcessor, get_registered_plotters, load
from tmd.exceptions import TMDProcessingError
from tmd.utils.utils import TMDUtils


def test_tmd_from_array_metadata_and_dimensions(small_heightmap: np.ndarray) -> None:
    md = {"comment": "unit", "width": 3, "height": 3, "resolution_x": 0.5, "resolution_y": 0.5}
    t = TMD(small_heightmap, md, compute_initial_stats=True)
    np.testing.assert_array_equal(t.height_map, small_heightmap)
    assert t.metadata["comment"] == "unit"
    d = t.dimensions
    assert "width" in d and "height" in d


def test_tmd_empty_array_then_crop_resize_raises(small_heightmap: np.ndarray) -> None:
    t = TMD(small_heightmap, {})
    with pytest.raises(ValueError):
        t.crop(0, 0, 10, 10)
    t2 = t.resize(2, 2)
    assert t2.shape == (2, 2)


def test_tmd_save_load_roundtrip(tmp_path: Path, small_heightmap: np.ndarray) -> None:
    t = TMD(small_heightmap, {"comment": "rt\n"})
    out = tmp_path / "saved.dat"
    saved = t.save(out)
    assert saved.endswith(".tmd")
    loaded = TMD.load(saved)
    np.testing.assert_allclose(loaded.height_map, small_heightmap, rtol=1e-5, atol=1e-5)


def test_tmd_load_module_function(tmp_tmd_path: Path) -> None:
    t = load(tmp_tmd_path)
    assert t.height_map.ndim == 2


def test_tmd_load_missing_file() -> None:
    with pytest.raises((FileNotFoundError, TMDProcessingError)):
        TMD.load("/nonexistent/path/that/does/not/exist.tmd")


def test_tmd_str_repr(small_heightmap: np.ndarray) -> None:
    t = TMD(small_heightmap, {"comment": "x"})
    assert "3x3" in str(t)
    assert "TMD(" in repr(t)


def test_get_registered_plotters() -> None:
    reg = get_registered_plotters()
    assert "available" in reg and "registered" in reg


def test_tmd_analyze_defects(small_heightmap: np.ndarray) -> None:
    t = TMD(small_heightmap, {})
    res = t.analyze_defects()
    assert res is not None


def test_tmd_analyze_defects_empty_raises() -> None:
    t = TMD(np.zeros((0, 0)))
    with pytest.raises(ValueError):
        t.analyze_defects()


def test_tmd_export_metadata(tmp_path: Path, small_heightmap: np.ndarray) -> None:
    t = TMD(small_heightmap, {})
    p = t.export_metadata(tmp_path / "meta.txt")
    assert Path(p).exists()


def test_tmd_set_default_plotter_valid() -> None:
    t = TMD(np.ones((4, 4), dtype=np.float32), {})
    t2 = t.set_default_plotter("matplotlib")
    assert t2 is t


def test_tmd_set_default_plotter_invalid() -> None:
    t = TMD(np.ones((4, 4), dtype=np.float32), {})
    with pytest.raises(ValueError):
        t.set_default_plotter("___not_a_real_backend___")


def test_tmd_plot_smoke(tmp_path: Path, small_heightmap: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)

    t = TMD(small_heightmap, {})
    fig = t.plot(plotter_strategy="matplotlib", mode="2d")
    assert fig is not None


def test_tmd_plot_profile_smoke(tmp_path: Path, small_heightmap: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)

    t = TMD(small_heightmap, {})
    fig = t.plot_profile(row_index=1, plotter_strategy="matplotlib")
    assert fig is not None


def test_tmd_plot_stats_smoke(tmp_path: Path, small_heightmap: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)

    t = TMD(small_heightmap, {})
    fig = t.plot_stats(plotter_strategy="matplotlib")
    assert fig is not None


def test_tmd_processor_requires_existing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.tmd"
    with pytest.raises(FileNotFoundError):
        TMDProcessor(missing)


def test_tmd_processor_process(tmp_tmd_path: Path) -> None:
    p = TMDProcessor(tmp_tmd_path)
    result = p.process()
    assert "height_map" in result and result["height_map"] is not None


def test_tmd_dimension_over_limit_raises() -> None:
    hm = np.zeros((100001, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="too large"):
        TMD(hm, {})


def test_load_function_alias(tmp_tmd_path: Path) -> None:
    assert isinstance(tmd_module.load(tmp_tmd_path), TMD)
