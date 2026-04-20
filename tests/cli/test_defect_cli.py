"""CLI defect detection tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from typer.testing import CliRunner

from tmd.cli.main import app
from tmd.utils.utils import TMDUtils


def _write_defect_fixture(path: Path) -> None:
    size = 96
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(x, y)
    height_map = 0.05 * np.sin(3.0 * np.pi * xx) + 0.05 * np.cos(4.0 * np.pi * yy)

    pit = np.exp(-((xx + 0.35) ** 2 + (yy + 0.40) ** 2) / (2.0 * 0.05**2))
    peak = np.exp(-((xx - 0.25) ** 2 + (yy - 0.20) ** 2) / (2.0 * 0.06**2))
    height_map -= 0.60 * pit
    height_map += 0.75 * peak
    height_map[46:48, 14:84] -= 0.22

    TMDUtils.write_tmd_file(height_map.astype(np.float32), path, version=2, comment="defect-cli\n")


def test_main_help_lists_defect_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "defect" in (result.stdout or "").lower()


def test_defect_file_json(tmp_path: Path) -> None:
    tmd_path = tmp_path / "fixture.tmd"
    _write_defect_fixture(tmd_path)

    runner = CliRunner(env={"TERM": "dumb"})
    result = runner.invoke(app, ["defect", "file", str(tmd_path), "--json"])
    assert result.exit_code == 0, result.stdout or str(result.exception)

    payload = json.loads((result.stdout or "").strip())
    assert payload["file"].endswith("fixture.tmd")
    assert "summary" in payload
    assert "defects" in payload
    assert "pits" in payload["defects"]
    assert "mask" not in payload["defects"]["pits"]
    assert "response" not in payload["defects"]["pits"]


def test_defect_file_json_with_heavy_flags(tmp_path: Path) -> None:
    tmd_path = tmp_path / "fixture.tmd"
    _write_defect_fixture(tmd_path)

    runner = CliRunner(env={"TERM": "dumb"})
    result = runner.invoke(
        app,
        ["defect", "file", str(tmd_path), "--json", "--include-mask", "--include-responses"],
    )
    assert result.exit_code == 0, result.stdout or str(result.exception)
    payload = json.loads((result.stdout or "").strip())
    assert payload["summary"]["total_count"] >= 1


def test_defect_file_exports_mask_and_overlay(tmp_path: Path) -> None:
    tmd_path = tmp_path / "fixture.tmd"
    _write_defect_fixture(tmd_path)
    mask_path = tmp_path / "mask.png"
    overlay_path = tmp_path / "overlay.png"

    runner = CliRunner(env={"TERM": "dumb"})
    result = runner.invoke(
        app,
        [
            "defect",
            "file",
            str(tmd_path),
            "--mask-output",
            str(mask_path),
            "--overlay-output",
            str(overlay_path),
        ],
    )
    assert result.exit_code == 0, result.stdout or str(result.exception)
    assert mask_path.exists()
    assert overlay_path.exists()

    mask = np.array(Image.open(mask_path))
    unique_values = set(np.unique(mask).tolist())
    assert unique_values.issubset({0, 255}), f"Mask should be binary, got values: {sorted(unique_values)}"


def test_defect_batch_writes_csv(tmp_path: Path) -> None:
    fixtures_dir = tmp_path / "fixtures"
    fixtures_dir.mkdir()
    _write_defect_fixture(fixtures_dir / "a.tmd")
    _write_defect_fixture(fixtures_dir / "b.tmd")
    out_csv = tmp_path / "defects.csv"

    runner = CliRunner(env={"TERM": "dumb"})
    result = runner.invoke(
        app,
        ["defect", "batch", str(fixtures_dir), "--output", str(out_csv)],
    )
    assert result.exit_code == 0, result.stdout or str(result.exception)
    text = out_csv.read_text(encoding="utf-8")
    assert "total_count" in text
    assert "directionality_anomalies" in text
