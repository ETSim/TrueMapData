"""Tests for the MATLAB ``.mat`` exporter/importer in ``tmd.compression.mat``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from tmd.compression import (
    ISO_PARAMETER_KEYS,
    MATExporter,
    MATImporter,
    TMDDataIOFactory,
    TMD_FORMAT_TAG,
)
from tmd.compression import mat as mat_mod


@pytest.fixture(scope="module")
def scipy_io():
    sio = pytest.importorskip("scipy.io")
    return sio


def _ramp(shape=(8, 8), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float64)


def test_factory_registers_mat() -> None:
    assert "mat" in TMDDataIOFactory.supported_export_formats()
    assert "mat" in TMDDataIOFactory.supported_import_formats()
    assert isinstance(TMDDataIOFactory.get_exporter("mat"), MATExporter)
    assert isinstance(TMDDataIOFactory.get_importer("mat"), MATImporter)


def test_factory_get_exporter_case_insensitive() -> None:
    assert isinstance(TMDDataIOFactory.get_exporter("MAT"), MATExporter)
    assert isinstance(TMDDataIOFactory.get_importer("Mat"), MATImporter)


def test_iso_parameter_keys_canonical_set() -> None:
    assert set(ISO_PARAMETER_KEYS) == {
        "Sdr",
        "Sdq",
        "Spc",
        "Zp",
        "Spd",
        "Sq",
        "Sku",
        "Ssk",
    }


def test_roundtrip_single_surface(tmp_path: Path, scipy_io) -> None:
    arr = _ramp((6, 9), seed=1)
    out = tmp_path / "single.mat"

    written = MATExporter().export({"height_map": arr}, str(out))
    assert Path(written).exists()

    raw = scipy_io.loadmat(str(out))
    assert "Surface_simulation_1" in raw
    assert "Surface_simulation_2" not in raw

    data = MATImporter().load(str(out))
    assert set(data["surfaces"].keys()) == {1}
    np.testing.assert_array_equal(data["surfaces"][1], arr)
    assert "height_map" in data
    np.testing.assert_array_equal(data["height_map"], arr)
    assert data["tmd_format"] == TMD_FORMAT_TAG


def test_roundtrip_multi_surface_with_parameters(tmp_path: Path) -> None:
    surfaces = {1: _ramp(seed=11), 2: _ramp(seed=22), 3: _ramp(seed=33)}
    parameters = {
        1: {"Sq": 0.123, "Sdr": 4.56, "Sku": 3.0, "Ssk": -0.1},
        2: {"Sq": 0.234, "Sdr": 7.89, "Sku": 2.7, "Ssk": 0.05},
        3: {"Sq": 0.345, "Sdr": 9.0, "Sku": 4.0, "Ssk": 0.2},
    }
    out = tmp_path / "multi.mat"
    MATExporter().export({"surfaces": surfaces, "parameters": parameters}, str(out))

    data = MATImporter().load(str(out))
    assert set(data["surfaces"].keys()) == {1, 2, 3}
    for idx in (1, 2, 3):
        np.testing.assert_allclose(data["surfaces"][idx], surfaces[idx])
        assert data["surfaces"][idx].dtype == np.float64
        for key, value in parameters[idx].items():
            assert pytest.approx(value, rel=1e-9, abs=1e-12) == data["parameters"][idx][key]
    # Multi-surface result must not expose the convenience height_map alias.
    assert "height_map" not in data


def test_metadata_roundtrip(tmp_path: Path) -> None:
    arr = _ramp((4, 5), seed=7)
    metadata = {"mmpp": 0.06, "x_length": 1.0, "y_length": 1.0, "comment": "demo"}
    out = tmp_path / "meta.mat"
    MATExporter().export({"height_map": arr, "metadata": metadata}, str(out))

    data = MATImporter().load(str(out))
    assert pytest.approx(metadata["mmpp"]) == data["metadata"]["mmpp"]
    assert pytest.approx(metadata["x_length"]) == data["metadata"]["x_length"]
    assert data["metadata"]["comment"] == "demo"


def test_importer_handles_real_schema(tmp_path: Path, scipy_io) -> None:
    """Build a .mat exactly like ``surface_from_simulation.mat`` and round-trip it."""
    raw_path = tmp_path / "sim_like.mat"
    surfaces = {1: _ramp(seed=101), 2: _ramp(seed=202)}
    iso_params = {key: float(idx) / 10.0 for idx, key in enumerate(ISO_PARAMETER_KEYS, start=1)}

    payload: Dict[str, Any] = {}
    for idx, arr in surfaces.items():
        payload[f"Surface_simulation_{idx}"] = arr
        payload[f"SurfaceParameters_{idx}"] = mat_mod._struct_from_dict(iso_params)
    scipy_io.savemat(str(raw_path), payload)

    data = MATImporter().load(str(raw_path))
    assert set(data["surfaces"].keys()) == {1, 2}
    np.testing.assert_allclose(data["surfaces"][1], surfaces[1])
    np.testing.assert_allclose(data["surfaces"][2], surfaces[2])
    for idx in (1, 2):
        for key, expected in iso_params.items():
            assert pytest.approx(expected) == data["parameters"][idx][key]


def test_invalid_height_map_dim(tmp_path: Path) -> None:
    out = tmp_path / "bad.mat"
    with pytest.raises(ValueError, match="must be 2D"):
        MATExporter().export({"height_map": np.zeros((4,))}, str(out))


def test_invalid_no_surfaces(tmp_path: Path) -> None:
    out = tmp_path / "empty.mat"
    with pytest.raises(ValueError, match="height_map"):
        MATExporter().export({}, str(out))


def test_empty_surfaces_dict(tmp_path: Path) -> None:
    out = tmp_path / "empty_surfaces.mat"
    with pytest.raises(ValueError, match="empty"):
        MATExporter().export({"surfaces": {}}, str(out))


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MATImporter().load(str(tmp_path / "does_not_exist.mat"))


def test_load_mat_without_surface_keys(tmp_path: Path, scipy_io) -> None:
    out = tmp_path / "no_surfaces.mat"
    scipy_io.savemat(str(out), {"unrelated": np.zeros((2, 2))})
    with pytest.raises(ValueError, match="Surface_simulation"):
        MATImporter().load(str(out))


def test_unknown_parameters_passthrough(tmp_path: Path) -> None:
    arr = _ramp((4, 4), seed=8)
    parameters = {1: {"Sq": 0.5, "custom_metric": 1.25, "label": "tile_a"}}
    out = tmp_path / "extra_params.mat"
    MATExporter().export({"height_map": arr, "parameters": parameters}, str(out))
    data = MATImporter().load(str(out))
    assert pytest.approx(0.5) == data["parameters"][1]["Sq"]
    assert pytest.approx(1.25) == data["parameters"][1]["custom_metric"]
    assert data["parameters"][1]["label"] == "tile_a"


def test_surfaces_as_iterable_indexed_from_one(tmp_path: Path) -> None:
    arrs = [_ramp((3, 3), seed=i) for i in range(2)]
    out = tmp_path / "list.mat"
    MATExporter().export({"surfaces": arrs}, str(out))
    data = MATImporter().load(str(out))
    assert set(data["surfaces"].keys()) == {1, 2}
    np.testing.assert_allclose(data["surfaces"][1], arrs[0])
    np.testing.assert_allclose(data["surfaces"][2], arrs[1])


def test_surfaces_dict_reindexed_contiguously(tmp_path: Path) -> None:
    """Non-contiguous integer keys should be re-emitted as 1..N on disk."""
    arrs = {3: _ramp((3, 3), seed=3), 7: _ramp((3, 3), seed=7)}
    out = tmp_path / "sparse.mat"
    MATExporter().export({"surfaces": arrs}, str(out))
    data = MATImporter().load(str(out))
    assert set(data["surfaces"].keys()) == {1, 2}


def test_parameters_keys_must_be_int_coercible(tmp_path: Path) -> None:
    out = tmp_path / "bad_params.mat"
    with pytest.raises(ValueError, match="parameters key"):
        MATExporter().export(
            {"height_map": _ramp((3, 3)), "parameters": {"abc": {"Sq": 1.0}}},
            str(out),
        )


def test_require_scipy_io_raises_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover the ImportError branch in ``_require_scipy_io``."""
    import builtins
    real_import = builtins.__import__

    def _failing_import(name: str, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError("simulated missing scipy")
        return real_import(name, *args, **kwargs)

    import sys
    saved = {k: v for k, v in sys.modules.items() if k.startswith("scipy")}
    for key in list(saved):
        sys.modules.pop(key, None)
    monkeypatch.setattr(builtins, "__import__", _failing_import)
    try:
        with pytest.raises(ImportError, match="scipy is required"):
            mat_mod._require_scipy_io()
    finally:
        sys.modules.update(saved)


def test_factory_unknown_format_raises() -> None:
    with pytest.raises(ValueError):
        TMDDataIOFactory.get_exporter("not_mat")
