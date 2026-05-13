"""CLI coverage for ``tmd.cli.apps.tribology_app`` (axis, contact curve, plot, lubrication)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    sys.modules.setdefault("noise", SimpleNamespace(snoise2=lambda *args, **kwargs: 0.0))
    return CliRunner(env={"TERM": "dumb"})


def _main_app():
    from tmd.cli.main import app

    return app


def test_tribology_subcommands_help(runner: CliRunner) -> None:
    for sub in ("axis", "contact-curve", "plot", "lubrication"):
        r = runner.invoke(_main_app(), ["tribology", sub, "--help"])
        assert r.exit_code == 0, sub


def test_tribology_axis_json_and_text(runner: CliRunner, tmp_tmd_path: Path) -> None:
    r = runner.invoke(_main_app(), ["tribology", "axis", str(tmp_tmd_path), "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert "axis_rad" in data and data.get("file")

    r2 = runner.invoke(_main_app(), ["tribology", "axis", str(tmp_tmd_path)])
    assert r2.exit_code == 0, r2.stdout
    assert "axis_rad" in (r2.stdout or "")


def test_tribology_axis_rejects_non_tmd(runner: CliRunner, tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("nope")
    r = runner.invoke(_main_app(), ["tribology", "axis", str(p)])
    assert r.exit_code == 1


def test_tribology_contact_curve_json_and_csv(
    runner: CliRunner, tmp_tmd_path: Path, tmp_path: Path
) -> None:
    r = runner.invoke(
        _main_app(),
        ["tribology", "contact-curve", str(tmp_tmd_path), "--n", "5", "--json"],
    )
    assert r.exit_code == 0, r.stdout or str(r.exception)
    data = json.loads((r.stdout or "").strip())
    assert "curve" in data and len(data["curve"]) == 5

    out_csv = tmp_path / "curve.csv"
    r2 = runner.invoke(
        _main_app(),
        ["tribology", "contact-curve", str(tmp_tmd_path), "--n", "4", "-o", str(out_csv)],
    )
    assert r2.exit_code == 0, r2.stdout or str(r2.exception)
    text = out_csv.read_text(encoding="utf-8")
    assert "separation" in text and text.count("\n") >= 5


def test_tribology_contact_curve_stdout_csv(runner: CliRunner, tmp_tmd_path: Path) -> None:
    r = runner.invoke(_main_app(), ["tribology", "contact-curve", str(tmp_tmd_path), "--n", "3"])
    assert r.exit_code == 0, r.stdout
    assert "separation" in (r.stdout or "") and "area_fraction" in (r.stdout or "")


def test_tribology_plot_writes_png(runner: CliRunner, tmp_tmd_path: Path, tmp_path: Path) -> None:
    png = tmp_path / "tribo_dash.png"
    r = runner.invoke(
        _main_app(),
        [
            "tribology",
            "plot",
            str(tmp_tmd_path),
            "-o",
            str(png),
            "--n",
            "8",
            "--dpi",
            "72",
            "--no-maps",
            "--include-anomaly-angle",
        ],
    )
    assert r.exit_code == 0, r.stdout or str(r.exception)
    assert png.is_file() and png.stat().st_size > 0


def test_tribology_lubrication_json_mocked(
    runner: CliRunner, tmp_tmd_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tmd.cli.apps import roughness_common as rc

    class _Dummy:
        def level(self):
            return self

    monkeypatch.setattr(rc, "surfalize_surface_class", lambda: object)
    monkeypatch.setattr(rc, "load_surface_for_roughness", lambda path, Surface: _Dummy())
    monkeypatch.setattr(
        rc,
        "roughness_dict",
        lambda surface, names: {n: float(i) for i, n in enumerate(names)},
    )

    r = runner.invoke(_main_app(), ["tribology", "lubrication", str(tmp_tmd_path), "--json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    payload = json.loads((r.stdout or "").strip())
    assert "parameters" in payload and "Vvv" in payload["parameters"]


def test_tribology_lubrication_table_mocked(
    runner: CliRunner, tmp_tmd_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tmd.cli.apps import roughness_common as rc

    class _Dummy:
        def level(self):
            return self

    monkeypatch.setattr(rc, "surfalize_surface_class", lambda: object)
    monkeypatch.setattr(rc, "load_surface_for_roughness", lambda path, Surface: _Dummy())
    monkeypatch.setattr(rc, "roughness_dict", lambda surface, names: {names[0]: 1.23})

    r = runner.invoke(_main_app(), ["tribology", "lubrication", str(tmp_tmd_path), "--no-json"])
    assert r.exit_code == 0, r.stdout or str(r.exception)
    assert "Vvv" in (r.stdout or "") or "1.23" in (r.stdout or "")
