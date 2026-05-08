"""Extra glTF exporter and helper coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tmd.model.base import ExportConfig, MeshData
from tmd.model.formats.gltf import (
    GLTFExporter,
    _add_material,
    _create_gltf_structure,
    _generate_texture_from_heightmap,
    _write_binary_glb,
)


def test_gltf_exporter_invalid_returns_none() -> None:
    cfg = ExportConfig(x_length=1.0, y_length=1.0, z_scale=1.0, binary=False)
    bad = np.zeros((1, 1), dtype=np.float32)
    assert GLTFExporter.export(bad, "out.gltf", cfg) is None


def test_gltf_and_glb_export_roundtrip_files(tmp_path: Path) -> None:
    hm = np.array(
        [
            [0.0, 0.1, 0.2, 0.0],
            [0.1, 0.3, 0.4, 0.1],
            [0.0, 0.2, 0.5, 0.2],
            [0.2, 0.1, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    cfg_json = ExportConfig(
        x_length=1.0,
        y_length=1.0,
        z_scale=1.0,
        binary=False,
        texture=True,
        extra={"embed_textures": True},
    )
    j = tmp_path / "m.gltf"
    assert GLTFExporter.export(hm, str(j), cfg_json) is not None
    assert j.exists() and j.stat().st_size > 0

    cfg_bin = ExportConfig(
        x_length=1.0,
        y_length=1.0,
        z_scale=1.0,
        binary=True,
        texture=False,
    )
    b = tmp_path / "m.glb"
    assert GLTFExporter.export(hm, str(b), cfg_bin) is not None
    assert b.exists() and b.read_bytes()[:4] == b"glTF"


def test_add_material_embed_and_external_and_write_glb(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    mesh = MeshData(verts, faces)
    mesh.ensure_normals()
    mesh.ensure_uvs()

    png_bytes = _generate_texture_from_heightmap(
        np.linspace(0, 1, 9, dtype=np.float32).reshape(3, 3),
        colormap="terrain",
        resolution=(16, 16),
    )
    assert isinstance(png_bytes, (bytes, bytearray)) and len(png_bytes) > 0

    out_gltf = str(tmp_path / "root.gltf")
    gltf = _create_gltf_structure(
        mesh,
        texture_data=None,
        generate_binary=True,
        embed_textures=True,
        material_map_bindings={},
        output_filename=out_gltf,
    )
    idx = _add_material(
        gltf,
        png_bytes,
        binary=True,
        embed_textures=True,
        material_map_bindings={},
        output_filename=out_gltf,
    )
    assert isinstance(idx, int)
    gltf["meshes"][0]["primitives"][0]["material"] = idx

    bump = tmp_path / "n.png"
    bump.write_bytes(png_bytes)
    gltf2 = _create_gltf_structure(
        mesh,
        texture_data=None,
        generate_binary=False,
        embed_textures=False,
        material_map_bindings={"map_bump": str(bump)},
        output_filename=str(tmp_path / "ext.gltf"),
    )
    assert gltf2.get("external_textures") or gltf2.get("materials")

    out_glb = tmp_path / "pack.glb"
    _write_binary_glb(gltf, str(out_glb))
    assert out_glb.exists()
