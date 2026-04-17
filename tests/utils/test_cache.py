#!/usr/bin/env python3
"""Tests for TMD CLI cache helpers."""

from pathlib import Path

from tmd.surface.terrain import TMDTerrain


def test_cache_roundtrip_loads_synthetic_tmd(tmp_path: Path) -> None:
    from tmd.cli.utils.caching import clear_cache, get_cache_stats
    from tmd.cli.core.io import load_tmd_file

    tmd_path = tmp_path / "cached.tmd"
    TMDTerrain.generate_synthetic_tmd(output_path=str(tmd_path), width=24, height=24, pattern="waves")

    clear_cache(expired_only=False)
    before = get_cache_stats()["entry_count"]

    first = load_tmd_file(tmd_path, with_console_status=False, use_cache=True)
    assert first is not None
    mid = get_cache_stats()["entry_count"]
    assert mid >= before

    second = load_tmd_file(tmd_path, with_console_status=False, use_cache=True)
    assert second is not None
