"""Extra tests for :mod:`tmd.cli.apps.sequence_app`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from tmd.cli.apps import sequence_app as seq_mod
from tmd.core.sequence import TMDSequence
from tmd.utils.utils import TMDUtils


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner(env={"TERM": "dumb"})


def test_spatial_helpers_and_paths() -> None:
    ref = {"width": 100, "height": 100, "x_length": 10.0, "y_length": 10.0}
    sp = seq_mod._spatial_fields_from_reference(ref, nh=50, nw=50)
    assert sp["width"] == 50 and sp["height"] == 50

    ref2 = {"mmpp": 0.1, "width": 10, "height": 10}
    sp2 = seq_mod._spatial_fields_from_reference(ref2, nh=20, nw=20)
    assert sp2["x_length"] == pytest.approx(2.0)

    d = Path("/tmp/aligned")
    p = Path("scan_circle_150mm.tmd")
    assert "150" in str(seq_mod._maps_output_dir(d, p))

    mp = seq_mod._mesh_output_path(Path("/a"), Path("circle_10mm_aligned.tmd"), "stl")
    assert mp.name.endswith(".stl") and "circle_10mm" in mp.name


def test_align_command_mocked(tmp_path: Path, runner: CliRunner, small_heightmap: np.ndarray, monkeypatch) -> None:
    t1 = tmp_path / "a.tmd"
    t2 = tmp_path / "b.tmd"
    TMDUtils.write_tmd_file(small_heightmap, str(t1), comment="1\n", version=2)
    TMDUtils.write_tmd_file(small_heightmap, str(t2), comment="2\n", version=2)
    outd = tmp_path / "aligned"

    monkeypatch.setattr(
        TMDSequence,
        "align_height_maps_opencv",
        lambda self, **_k: {"slices": [], "transforms": []},
    )

    app = seq_mod.create_sequence_app()
    r = runner.invoke(
        app,
        [
            "align",
            str(t1),
            str(t2),
            "--output-dir",
            str(outd),
            "--no-save-json",
        ],
    )
    assert r.exit_code == 0
    assert list(outd.glob("*_aligned.tmd"))


def test_export_command_mocked(tmp_path: Path, runner: CliRunner, small_heightmap: np.ndarray, monkeypatch) -> None:
    ad = tmp_path / "aligned"
    ad.mkdir()
    t = ad / "frame_aligned.tmd"
    TMDUtils.write_tmd_file(small_heightmap, str(t), comment="e\n", version=2)

    monkeypatch.setattr(seq_mod, "export_maps_command", lambda *a, **k: True)
    monkeypatch.setattr(seq_mod, "export_model", lambda *a, **k: True)
    monkeypatch.setattr(seq_mod, "apply_maps_to_mesh", lambda **k: {"obj": "x"})

    app = seq_mod.create_sequence_app()
    tpl = Path(__file__).resolve().parents[3] / "tmd" / "fixtures" / "templates" / "plane" / "plane.obj"
    r = runner.invoke(
        app,
        [
            "export",
            str(ad),
            "--template-mesh",
            str(tpl),
        ],
    )
    assert r.exit_code == 0
