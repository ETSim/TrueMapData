"""Extra coverage for tmd.cli.utils.caching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tmd.cli.utils import caching as cache_mod


@pytest.fixture
def cache_under_tmp(tmp_path: Path) -> cache_mod.TMDCache:
    return cache_mod.TMDCache(cache_dir=tmp_path / "cache", ttl=60)


def _write_dummy_tmd(tmp_path: Path, name: str = "fixture.tmd") -> Path:
    p = tmp_path / name
    p.write_bytes(b"hello-world")
    return p


def test_get_cache_key_stable_for_same_file(tmp_path: Path, cache_under_tmp: cache_mod.TMDCache) -> None:
    f = _write_dummy_tmd(tmp_path)
    k1 = cache_under_tmp._get_cache_key(f)
    k2 = cache_under_tmp._get_cache_key(f)
    assert k1 == k2


def test_get_cache_key_distinct_for_different_paths(
    tmp_path: Path, cache_under_tmp: cache_mod.TMDCache
) -> None:
    a = _write_dummy_tmd(tmp_path, "a.tmd")
    b = _write_dummy_tmd(tmp_path, "b.tmd")
    assert cache_under_tmp._get_cache_key(a) != cache_under_tmp._get_cache_key(b)


def test_get_cache_key_falls_back_when_info_fails(
    tmp_path: Path, cache_under_tmp: cache_mod.TMDCache, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = _write_dummy_tmd(tmp_path)
    from tmd.utils.files import TMDFileUtilities

    def _boom(_path):
        raise OSError("nope")

    monkeypatch.setattr(TMDFileUtilities, "get_file_info", _boom)
    key = cache_under_tmp._get_cache_key(f)
    assert isinstance(key, str) and len(key) > 0


def test_load_index_returns_empty_when_corrupt(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "index.json").write_text("not-json", encoding="utf-8")
    c = cache_mod.TMDCache(cache_dir=cache_dir)
    assert c._index == {}


def test_save_index_writes_file(tmp_path: Path, cache_under_tmp: cache_mod.TMDCache) -> None:
    cache_under_tmp._index = {"abc": {"timestamp": 1.0, "cache_path": "x", "size": 0}}
    cache_under_tmp._save_index()
    idx = (cache_under_tmp.cache_dir / "index.json")
    assert idx.exists()


def test_put_and_get_roundtrip(tmp_path: Path, cache_under_tmp: cache_mod.TMDCache) -> None:
    f = _write_dummy_tmd(tmp_path)
    hm = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    meta = {"x_length": 1.0}
    assert cache_under_tmp.put(f, hm, meta) is True

    got = cache_under_tmp.get(f)
    assert got is not None
    arr, got_meta = got
    assert arr.shape == hm.shape
    assert np.allclose(arr, hm)
    assert got_meta == meta


def test_get_returns_none_for_missing_entry(tmp_path: Path, cache_under_tmp: cache_mod.TMDCache) -> None:
    f = _write_dummy_tmd(tmp_path, "missing.tmd")
    assert cache_under_tmp.get(f) is None


def test_get_evicts_expired_entries(tmp_path: Path, cache_under_tmp: cache_mod.TMDCache) -> None:
    f = _write_dummy_tmd(tmp_path)
    hm = np.zeros((4, 4), dtype=np.float32)
    cache_under_tmp.put(f, hm, {})
    key = cache_under_tmp._get_cache_key(f)
    cache_under_tmp._index[key]["timestamp"] = 0.0
    assert cache_under_tmp.get(f) is None
    assert key not in cache_under_tmp._index


def test_get_evicts_when_cache_file_missing(
    tmp_path: Path, cache_under_tmp: cache_mod.TMDCache
) -> None:
    f = _write_dummy_tmd(tmp_path)
    cache_under_tmp.put(f, np.zeros((2, 2), dtype=np.float32), {})
    key = cache_under_tmp._get_cache_key(f)
    cache_path = Path(cache_under_tmp._index[key]["cache_path"])
    cache_path.unlink()
    assert cache_under_tmp.get(f) is None


def test_remove_cache_entry_idempotent(tmp_path: Path, cache_under_tmp: cache_mod.TMDCache) -> None:
    f = _write_dummy_tmd(tmp_path)
    cache_under_tmp.put(f, np.zeros((2, 2), dtype=np.float32), {})
    key = cache_under_tmp._get_cache_key(f)
    cache_under_tmp._remove_cache_entry(key)
    assert key not in cache_under_tmp._index
    cache_under_tmp._remove_cache_entry(key)
    assert key not in cache_under_tmp._index


def test_clear_expired_only_removes_aged_entries(
    tmp_path: Path,
    cache_under_tmp: cache_mod.TMDCache,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh = _write_dummy_tmd(tmp_path, "fresh.tmd")
    aged = _write_dummy_tmd(tmp_path, "aged.tmd")
    cache_under_tmp.put(fresh, np.zeros((2, 2), dtype=np.float32), {})
    cache_under_tmp.put(aged, np.zeros((2, 2), dtype=np.float32), {})

    aged_key = cache_under_tmp._get_cache_key(aged)
    cache_under_tmp._index[aged_key]["timestamp"] = 0.0

    removed = cache_under_tmp.clear_expired()
    assert removed >= 1
    assert aged_key not in cache_under_tmp._index


def test_clear_all_drops_index_and_files(tmp_path: Path, cache_under_tmp: cache_mod.TMDCache) -> None:
    f = _write_dummy_tmd(tmp_path)
    cache_under_tmp.put(f, np.zeros((4, 4), dtype=np.float32), {})
    n = cache_under_tmp.clear_all()
    assert n == 1
    assert cache_under_tmp._index == {}


def test_get_stats_reports_counts(tmp_path: Path, cache_under_tmp: cache_mod.TMDCache) -> None:
    f = _write_dummy_tmd(tmp_path)
    cache_under_tmp.put(f, np.zeros((4, 4), dtype=np.float32), {})
    stats = cache_under_tmp.get_stats()
    assert stats["entry_count"] >= 1
    assert "total_size_bytes" in stats
    assert "cache_dir" in stats


def test_get_cache_singleton_same_instance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache_mod, "_cache_instance", None)
    monkeypatch.setattr(cache_mod, "DEFAULT_CACHE_DIR", tmp_path / "default_cache")
    c1 = cache_mod.get_cache()
    c2 = cache_mod.get_cache()
    assert c1 is c2


def test_public_api_round_trip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cache_mod, "_cache_instance", None)
    monkeypatch.setattr(cache_mod, "DEFAULT_CACHE_DIR", tmp_path / "default_cache")
    f = _write_dummy_tmd(tmp_path)
    hm = np.full((3, 3), 0.5, dtype=np.float32)
    assert cache_mod.cache_tmd_data(f, hm, {"foo": "bar"}) is True
    got = cache_mod.get_cached_tmd_data(f)
    assert got is not None
    stats = cache_mod.get_cache_stats()
    assert stats["entry_count"] >= 1
    cleared = cache_mod.clear_cache(expired_only=False)
    assert cleared >= 1
