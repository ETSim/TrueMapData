"""Coverage for helpers and flows in :mod:`tmd.cli.commands.model`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from tmd.cli.commands import model as model_cmd
from tmd.utils.utils import TMDUtils


def test_resolve_map_slots_monkeypatched(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Tmd:
        height_map = np.ones((4, 4), dtype=np.float32)
        metadata = {"comment": "x"}

    written: list = []

    def fake_export_map(_height_map, output_file, _map_type, **_k):
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        Path(output_file).write_bytes(b"x")
        written.append(output_file)

    monkeypatch.setattr(
        "tmd.image.export.exporter.MapExporter.export_map",
        staticmethod(fake_export_map),
    )
    slots = model_cmd._resolve_map_slots(_Tmd(), tmp_path, "stem", compress=0, normalize=False)
    assert "map_kd" in slots
    assert len(written) >= 1


def test_read_template_obj_metrics_and_resolve_paths(tmp_path: Path) -> None:
    # X/Z spans must be positive (flat triangle in XY has span_z == 0 and is rejected).
    obj_text = (
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 0 0 1\n"
        "f 1 2 3\n"
    )
    obj = tmp_path / "tri.obj"
    obj.write_text(obj_text, encoding="utf-8")
    sx, sz, uv = model_cmd._read_template_obj_metrics(obj, obj_units_to_mm=1000.0)
    assert sx > 0 and sz > 0
    assert uv is False

    p = model_cmd._resolve_template_mesh_path(
        template_mesh_path=obj,
        template_plane_dir=None,
        template_kind="plane",
        template_fixtures_dir=None,
    )
    assert p == obj

    builtin = model_cmd._default_template_fixtures_dir()
    assert (builtin / "plane" / "plane.obj").is_file()

    (tmp_path / "plane.obj").write_text(obj_text, encoding="utf-8")
    pp = model_cmd._resolve_template_mesh_path(
        template_mesh_path=None,
        template_plane_dir=tmp_path,
        template_kind="plane",
        template_fixtures_dir=None,
    )
    assert pp == tmp_path / "plane.obj"

    with pytest.raises(ValueError):
        model_cmd._resolve_template_mesh_path(
            template_mesh_path=None,
            template_plane_dir=None,
            template_kind="not_a_builtin",
            template_fixtures_dir=None,
        )


def test_generate_uv_sphere_and_ensure_template(tmp_path: Path) -> None:
    sp = tmp_path / "sphere.obj"
    model_cmd._generate_uv_sphere_obj(sp, rings=4, segments=8, radius=1.0)
    text = sp.read_text(encoding="utf-8")
    assert "v " in text and "f " in text

    with pytest.raises(ValueError):
        model_cmd._generate_uv_sphere_obj(sp, rings=2, segments=8)

    low = tmp_path / "low.obj"
    low.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    out = model_cmd._ensure_template_quality(low, "sphere")
    assert out.read_text(encoding="utf-8").count("\nf ") > 2


def test_require_mm_length_and_resolve_mm_per_pixel() -> None:
    assert model_cmd._require_mm_length({"x_length": 1.0}, "x_length") == 1.0
    with pytest.raises(ValueError):
        model_cmd._require_mm_length({}, "x_length")
    with pytest.raises(ValueError):
        model_cmd._require_mm_length({"x_length": 0.0}, "x_length")

    assert model_cmd._resolve_mm_per_pixel({}, 0.1) == 0.1
    assert model_cmd._resolve_mm_per_pixel({"mmpp": 0.05}, None) == 0.05
    assert model_cmd._resolve_mm_per_pixel({"width": 100, "x_length": 6.0}, None) == 0.06
    assert model_cmd._resolve_mm_per_pixel({}, None, fallback_mm_per_pixel=0.07) == 0.07
    with pytest.raises(ValueError):
        model_cmd._resolve_mm_per_pixel({}, None, fallback_mm_per_pixel=0.0)


def test_apply_maps_to_mesh_mocked_tiling(tmp_path: Path, tmp_tmd_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plane = Path(__file__).resolve().parents[3] / "tmd" / "fixtures" / "templates" / "plane" / "plane.obj"
    slots = {
        "map_kd": str(tmp_path / "textures" / "kd.png"),
        "map_bump": str(tmp_path / "textures" / "bump.png"),
        "map_disp": str(tmp_path / "textures" / "disp.png"),
        "map_pr": str(tmp_path / "textures" / "pr.png"),
    }
    for p in slots.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_bytes(b"\x89PNG\r\n\x1a\n")

    def fake_tiling(*_a, **_k):
        return slots

    monkeypatch.setattr(
        "tmd.image.export.exporter.MapExporter.export_material_binding_maps_with_physical_tiling",
        staticmethod(fake_tiling),
    )
    out_root = tmp_path / "bundle"
    res = model_cmd.apply_maps_to_mesh(
        tmp_tmd_path,
        out_root,
        template_mesh_path=plane,
        output_prefix="pfx",
        application_mode="uv",
        uv_alignment_mode="preserve",
        tmd_mm_per_pixel=0.1,
        max_texture_edge=512,
    )
    assert Path(res["obj"]).exists() and Path(res["mtl"]).exists()
    assert res["application_mode"] == "uv"


def test_batch_export_models_patched_glob(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_heightmap: np.ndarray) -> None:
    t1 = tmp_path / "one.tmd"
    t2 = tmp_path / "two.tmd"
    TMDUtils.write_tmd_file(small_heightmap, str(t1), comment="a\n", version=2)
    TMDUtils.write_tmd_file(small_heightmap, str(t2), comment="b\n", version=2)

    real_glob = Path.glob

    def _glob(self: Path, pattern: str):
        if str(tmp_path) in pattern.replace("/", "\\") and pattern.endswith(".tmd"):
            return iter([t1, t2])
        return real_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", _glob)
    monkeypatch.setattr(model_cmd, "export_model", lambda **_k: True)

    ok = model_cmd.batch_export_models(tmp_path, output_dir=tmp_path / "out", format="stl", max_workers=1)
    assert ok is True


def test_generate_model_command_smoke(tmp_path: Path, small_heightmap: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    tmd_path = tmp_path / "in.tmd"
    TMDUtils.write_tmd_file(small_heightmap, str(tmd_path), comment="g\n", version=2)
    out = tmp_path / "out.stl"

    def fake_convert(**_k):
        verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        return verts, faces

    fake_mod = MagicMock()
    fake_mod.convert_heightmap_to_adaptive_mesh = fake_convert
    monkeypatch.setitem(sys.modules, "tmd.model.adaptive_mesh", fake_mod)

    assert model_cmd.generate_model_command(tmd_path, output_file=out, z_scale=1.0, max_triangles=100) is True
