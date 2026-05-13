"""Tests for NVBD mesh writer and exporter."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from tmd.model.base import ExportConfig, MeshData
from tmd.model.formats.nvbd import NVBDExporter, write_nvbd


def _read_nvbd_header(path: Path) -> tuple[int, int, int]:
    data = path.read_bytes()
    assert data[:4] == b"NVBD"
    ver, nv, nf = struct.unpack_from("<III", data, 4)
    assert ver == 1
    return nv, nf, len(data)


def test_write_nvbd_without_normals(tmp_path: Path) -> None:
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = MeshData(verts, faces, normals=None)
    out = tmp_path / "a.nvbd"
    write_nvbd(mesh, str(out))
    nv, nf, nbytes = _read_nvbd_header(out)
    assert nv == 3 and nf == 1
    # header + verts + faces + uint32 normals flag (0)
    expect = 16 + 36 + 12 + 4
    assert nbytes == expect


def test_write_nvbd_with_normals(tmp_path: Path) -> None:
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    norms = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    mesh = MeshData(verts, faces, normals=norms)
    out = tmp_path / "b.nvbd"
    write_nvbd(mesh, str(out))
    nv, nf, nbytes = _read_nvbd_header(out)
    assert nv == 3 and nf == 1
    # same as no-normals case but normals flag is 1 and 3 * float32*3 normals follow
    expect = 16 + 36 + 12 + 4 + 36
    assert nbytes == expect


def test_nvbd_exporter_writes_file(tmp_path: Path) -> None:
    z = np.array([[0.0, 0.1, 0.2], [0.1, 0.2, 0.3], [0.0, 0.0, 0.1]], dtype=np.float32)
    out = tmp_path / "m.nvbd"
    cfg = ExportConfig(triangulation_method="quadtree", max_triangles=500, calculate_normals=False)
    path = NVBDExporter.export(z, str(out), cfg)
    assert path is not None
    assert out.is_file()
    nv, nf, _ = _read_nvbd_header(out)
    assert nv >= 3 and nf >= 1


def test_nvbd_exporter_returns_none_on_mesh_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    z = np.zeros((4, 4), dtype=np.float32)

    @classmethod
    def fake_create(cls, height_map, config):
        raise RuntimeError("mesh failure")

    monkeypatch.setattr(NVBDExporter, "create_mesh_from_heightmap", fake_create)
    out = tmp_path / "bad.nvbd"
    assert NVBDExporter.export(z, str(out), ExportConfig()) is None
