#!/usr/bin/env python3
"""Tests for surface defect detection."""

from __future__ import annotations

import time

import numpy as np
import pytest

from tmd import TMD
from tmd.surface.defects import DefectDetectionConfig, detect_surface_defects


def _base_surface(size: int = 128) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, size)
    y = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(x, y)
    return 0.08 * np.sin(4.0 * np.pi * xx) + 0.06 * np.cos(3.0 * np.pi * yy)


def _surface_with_all_defects(size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(1234)
    surface = _base_surface(size)

    yy, xx = np.ogrid[:size, :size]

    # Pit
    pit = np.exp(-((xx - 28) ** 2 + (yy - 24) ** 2) / (2.0 * 3.0**2))
    surface -= 0.8 * pit

    # Peak
    peak = np.exp(-((xx - 94) ** 2 + (yy - 102) ** 2) / (2.0 * 3.5**2))
    surface += 0.9 * peak

    # Scratch: horizontal narrow trench
    surface[60:63, 20:108] -= 0.28

    # Crack: diagonal sharp groove
    for idx in range(24, 104):
        y = idx
        x = idx - 12
        if 2 <= y < size - 2 and 2 <= x < size - 2:
            surface[y - 1 : y + 2, x - 1 : x + 2] -= 0.35

    # Directionality anomaly: vertically oriented stripe texture patch
    patch = (slice(18, 62), slice(78, 118))
    px = np.linspace(0, 6.0 * np.pi, patch[1].stop - patch[1].start)
    stripe = np.sin(px)[None, :]
    surface[patch] += 0.14 * np.repeat(stripe, patch[0].stop - patch[0].start, axis=0)

    # Mild sensor-like noise
    surface += 0.015 * rng.standard_normal((size, size))
    return surface


def test_detect_surface_defects_returns_typed_schema() -> None:
    surface = _surface_with_all_defects()

    result = detect_surface_defects(surface)

    expected_defect_keys = {
        "pits",
        "peaks",
        "scratches",
        "cracks",
        "directionality_anomalies",
    }
    assert set(result["defects"].keys()) == expected_defect_keys

    assert "mask" not in result
    assert "labels" not in result
    assert "overlay_rgb" not in result

    assert result["summary"]["total_count"] >= 5
    assert 0.0 <= result["summary"]["global_confidence"] <= 1.0

    for defect_name in expected_defect_keys:
        entry = result["defects"][defect_name]
        assert isinstance(entry["count"], int)
        assert 0.0 <= entry["confidence"] <= 1.0
        assert "mask" not in entry
        assert "response" not in entry
        assert "areas" not in entry


def test_detect_surface_defects_full_mode_emits_heavy_outputs() -> None:
    surface = _surface_with_all_defects()

    result = detect_surface_defects(
        surface,
        DefectDetectionConfig(output_mode="full", include_responses=True),
    )

    assert result["mask"].shape == surface.shape
    assert result["labels"].shape == surface.shape
    assert result["overlay_rgb"].shape == (surface.shape[0], surface.shape[1], 3)
    assert np.all(result["overlay_rgb"] >= 0.0)
    assert np.all(result["overlay_rgb"] <= 1.0)
    assert result["defects"]["pits"]["mask"].shape == surface.shape
    assert result["defects"]["pits"]["response"].shape == surface.shape
    assert isinstance(result["defects"]["pits"]["areas"], list)


def test_detect_surface_defects_detects_each_defect_family() -> None:
    surface = _surface_with_all_defects()
    result = detect_surface_defects(surface)

    assert result["defects"]["pits"]["count"] >= 1
    assert result["defects"]["peaks"]["count"] >= 1
    assert result["defects"]["scratches"]["count"] >= 1
    assert result["defects"]["cracks"]["count"] >= 1
    assert result["defects"]["directionality_anomalies"]["count"] >= 1


def test_detect_surface_defects_respects_confidence_threshold() -> None:
    surface = _surface_with_all_defects()

    low = detect_surface_defects(surface, DefectDetectionConfig(min_confidence=0.0))
    high = detect_surface_defects(surface, DefectDetectionConfig(min_confidence=0.75))

    assert high["summary"]["total_count"] <= low["summary"]["total_count"]


def test_detect_surface_defects_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError):
        detect_surface_defects(np.ones((10, 10, 3)))


def test_tmd_api_exposes_analyze_defects() -> None:
    surface = _surface_with_all_defects()
    tmd_data = TMD(surface, {"comment": "defect-api-test"})

    result = tmd_data.analyze_defects(min_confidence=0.1)

    assert result["summary"]["total_count"] >= 1
    assert "cracks" in result["defects"]


def test_tmd_stats_are_lazy_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def _fake_compute_stats(height_map: np.ndarray) -> dict[str, float]:
        calls["count"] += 1
        return {"mean": float(np.mean(height_map))}

    monkeypatch.setattr("tmd.core.tmd.compute_stats", _fake_compute_stats)

    tmd_data = TMD(_surface_with_all_defects(), {"comment": "lazy-stats"})
    assert calls["count"] == 0

    _ = tmd_data.analyze_defects()
    assert calls["count"] == 0

    _ = tmd_data.stats
    assert calls["count"] == 1


def test_summary_mode_runtime_guard_against_full_mode() -> None:
    surface = _surface_with_all_defects(size=320)

    # Warm up SciPy paths to reduce first-call bias.
    detect_surface_defects(surface, DefectDetectionConfig(output_mode="summary"))
    detect_surface_defects(surface, DefectDetectionConfig(output_mode="full"))

    start = time.perf_counter()
    detect_surface_defects(surface, DefectDetectionConfig(output_mode="summary"))
    summary_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    detect_surface_defects(surface, DefectDetectionConfig(output_mode="full"))
    full_elapsed = time.perf_counter() - start

    # Coarse guard: summary mode should not regress to being slower than full mode.
    assert summary_elapsed <= full_elapsed * 1.20
