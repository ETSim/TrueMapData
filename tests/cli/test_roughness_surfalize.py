"""CLI roughness (Surfalize): help always; file/batch when surfalize is installed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from tmd.cli.main import app
from tmd.utils.utils import TMDUtils


def _write_minimal_v2_tmd(path: Path) -> None:
    """Canonical v2 TMD bytes that Surfalize and TMDUtils agree on."""
    hm = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    TMDUtils.write_tmd_file(hm, path, version=2, comment="pytest\n")


def test_main_help_lists_roughness() -> None:
    runner = CliRunner()
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "roughness" in (r.stdout or "").lower()


def test_roughness_subhelp() -> None:
    runner = CliRunner()
    r = runner.invoke(app, ["roughness", "--help"])
    assert r.exit_code == 0
    out = (r.stdout or "").lower()
    assert "file" in out and "batch" in out and "sequence" in out


def test_roughness_file_json_smoke_quick(tmp_path: Path) -> None:
    pytest.importorskip("surfalize")

    tmd_path = tmp_path / "synth.tmd"
    _write_minimal_v2_tmd(tmd_path)

    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["roughness", "file", str(tmd_path), "--json", "--quick"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert data.get("file")
    assert "parameters" in data
    assert "Sa" in data["parameters"]


def test_roughness_file_default_includes_all_iso_parameters(tmp_path: Path) -> None:
    """Default (no --quick / --all / --params) uses Surfalize ISO 25178 set."""
    pytest.importorskip("surfalize")
    from surfalize import Surface

    tmd_path = tmp_path / "synth.tmd"
    _write_minimal_v2_tmd(tmd_path)

    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["roughness", "file", str(tmd_path), "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    expected = list(Surface.ISO_PARAMETERS)
    for name in expected:
        assert name in data["parameters"], f"missing ISO parameter {name!r}"


def test_roughness_sequence_json_ordered(tmp_path: Path) -> None:
    pytest.importorskip("surfalize")

    a = tmp_path / "a.tmd"
    b = tmp_path / "b.tmd"
    _write_minimal_v2_tmd(a)
    _write_minimal_v2_tmd(b)

    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(app, ["roughness", "sequence", str(b), str(a), "--quick", "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    rows = json.loads((r.stdout or "").strip())
    assert len(rows) == 2
    assert rows[0]["frame"] == 0 and rows[0]["file"] == b.name
    assert rows[1]["frame"] == 1 and rows[1]["file"] == a.name
    assert "Sa" in rows[0]


def test_roughness_sequence_from_dir_sort_name(tmp_path: Path) -> None:
    pytest.importorskip("surfalize")

    d = tmp_path / "seq"
    d.mkdir()
    _write_minimal_v2_tmd(d / "z.tmd")
    _write_minimal_v2_tmd(d / "m.tmd")

    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(
        app,
        ["roughness", "sequence", "--from-dir", str(d), "--sort-by", "name", "--quick", "--json"],
    )
    assert r.exit_code == 0, r.stdout or str(r.exception)
    rows = json.loads((r.stdout or "").strip())
    assert [row["file"] for row in rows] == ["m.tmd", "z.tmd"]


def test_roughness_batch_csv(tmp_path: Path) -> None:
    pytest.importorskip("surfalize")

    d = tmp_path / "tmds"
    d.mkdir()
    _write_minimal_v2_tmd(d / "a.tmd")
    _write_minimal_v2_tmd(d / "b.tmd")
    out_csv = tmp_path / "out.csv"

    runner = CliRunner(env={"TERM": "dumb"})
    r = runner.invoke(
        app,
        ["roughness", "batch", str(d), "--output", str(out_csv), "--quick"],
    )
    assert r.exit_code == 0, r.stdout or str(r.exception)
    text = out_csv.read_text(encoding="utf-8")
    assert "Sa" in text or "sa" in text.lower()
