"""Extra PLY exporter coverage."""

from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from tmd.model.base import ExportConfig, MeshData
from tmd.model.formats.ply import PLYExporter, _generate_vertex_colors, write_ascii_ply


def test_write_ascii_ply_header_and_counts(tmp_path: Path, triangle_vertices: np.ndarray, triangle_faces: np.ndarray) -> None:
    mesh = MeshData(triangle_vertices, triangle_faces)
    out = tmp_path / "m.ply"
    write_ascii_ply(mesh, str(out))
    text = out.read_text(encoding="utf-8")
    assert "ply" in text
    assert "format ascii 1.0" in text
    assert "element vertex 3" in text
    assert "element face 1" in text


def test_generate_vertex_colors_matplotlib(triangle_vertices: np.ndarray) -> None:
    hm = np.ones((2, 2), dtype=np.float32)
    rgb = _generate_vertex_colors(triangle_vertices, hm, color_map="viridis")
    assert rgb.shape == (3, 3)


def test_generate_vertex_colors_grayscale_fallback(
    triangle_vertices: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    hm = np.ones((2, 2), dtype=np.float32)
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals=None,
        locals=None,
        fromlist: tuple = (),
        level: int = 0,
    ):
        if name == "matplotlib" and fromlist and "cm" in fromlist:
            raise ImportError("mock matplotlib.cm")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    rgb = _generate_vertex_colors(triangle_vertices, hm)
    assert rgb.shape == (3, 3)


def test_ply_exporter_binary_and_ascii(tmp_path: Path) -> None:
    hm = np.array(
        [[0.0, 0.2], [0.1, 0.3]],
        dtype=np.float32,
    )
    cfg_bin = ExportConfig(x_length=1.0, y_length=1.0, z_scale=1.0, binary=True)
    p_bin = tmp_path / "o.ply"
    assert PLYExporter.export(hm, str(p_bin), cfg_bin) is not None
    assert p_bin.exists()

    cfg_ascii = ExportConfig(x_length=1.0, y_length=1.0, z_scale=1.0, binary=False)
    p_ascii = tmp_path / "o_ascii.ply"
    assert PLYExporter.export(hm, str(p_ascii), cfg_ascii) is not None
    head = p_ascii.read_text(encoding="utf-8", errors="ignore")[:200]
    assert "ascii" in head
