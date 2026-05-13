"""Tests for wear_simulation TMD/MAT I/O helpers."""

from __future__ import annotations

import numpy as np

from tmd.compression import MATImporter
from tmd.sequence.wear_simulation import export_final_state_mat, load_surfaces_for_simulation
from tmd.utils.utils import TMDUtils


def test_load_surfaces_for_simulation_from_tmd_flat_slave(tmp_path) -> None:
    hm = np.linspace(0.0, 0.1, 64, dtype=np.float32).reshape(8, 8)
    tmd_path = tmp_path / "tiny.tmd"
    TMDUtils.write_tmd_file(
        hm,
        str(tmd_path),
        comment="pytest wear_simulation_io\n",
        version=2,
        x_length=2.0,
        y_length=3.0,
    )

    surfaces, stored, meta, mode = load_surfaces_for_simulation(
        tmd_path,
        None,
        second_surface_mode="flat",
        max_grid_dim=256,
        normalize_to_unit_sq=False,
        repo_root=tmp_path,
    )

    assert mode == "flat"
    assert set(surfaces.keys()) == {1, 2}
    assert surfaces[1].shape == (8, 8)
    assert surfaces[2].shape == (8, 8)
    assert np.allclose(surfaces[2], 0.0)
    assert meta["input_a_kind"] == "tmd"
    assert float(meta["a_x_length"]) == 2.0
    assert float(meta["a_y_length"]) == 3.0


def test_export_final_state_mat_round_trip(tmp_path) -> None:
    n = 6
    ms0 = np.random.default_rng(0).random((n, n)).astype(np.float64)
    ss0 = np.zeros_like(ms0)
    final = {
        "MS": ms0 * 0.9,
        "SS": ss0,
        "cumulative_wear_MS": np.abs(ms0 - ms0 * 0.9),
    }
    out = tmp_path / "out.mat"
    export_final_state_mat(
        out,
        ms_initial=ms0,
        ss_initial=ss0,
        final_state=final,
        extra_metadata={"note": "pytest"},
    )

    payload = MATImporter().load(str(out))
    assert "tmd_format" in payload
    surf = payload["surfaces"]
    assert set(surf.keys()) == {1, 2, 3, 4, 5}
    assert surf[1].shape == (n, n)
    assert surf[3].shape == (n, n)


def test_ms_snapshots_to_sequence_adds_frames() -> None:
    from tmd.sequence.wear_simulation import ms_snapshots_to_sequence

    z = np.zeros((4, 4), dtype=np.float64)
    snaps = {
        0: {"MS": z, "SS": z},
        1: {"MS": z + 0.1, "SS": z},
    }
    seq = ms_snapshots_to_sequence(
        snaps,
        name="t",
        pitch_metadata={"a_x_length": 1.0, "a_y_length": 1.0, "a_width": 4.0, "a_height": 4.0},
    )
    assert seq.get_frame_count() == 2
    m0 = seq.get_frame_metadata(0) or {}
    assert m0.get("step") == 0
    assert float(m0["x_length"]) == 1.0
