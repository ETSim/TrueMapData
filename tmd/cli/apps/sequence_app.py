"""CLI for TMD height-map sequences (alignment, crop, batch export)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from tmd import TMD
from tmd.cli.commands.export import export_maps_command
from tmd.cli.commands.model import (
    MeshMethod,
    QualityPreset,
    apply_maps_to_mesh,
    export_model,
    get_quality_params,
)
from tmd.cli.core.ui import console
from tmd.core.sequence import TMDSequence
from tmd.utils.utils import TMDUtils


def _spatial_fields_from_reference(ref_meta: Dict[str, Any], nh: int, nw: int) -> Dict[str, Any]:
    """Update width/height and physical lengths for a cropped aligned patch (square pixels)."""
    out = dict(ref_meta)
    out["width"] = int(nw)
    out["height"] = int(nh)
    mmpp = out.get("mmpp")
    if mmpp is not None:
        m = float(mmpp)
        out["x_length"] = m * nw
        out["y_length"] = m * nh
    else:
        ow = int(out.get("width", nw) or nw)
        oh = int(out.get("height", nh) or nh)
        xl = float(out.get("x_length", 10.0))
        yl = float(out.get("y_length", 10.0))
        out["x_length"] = xl * nw / max(ow, 1)
        out["y_length"] = yl * nh / max(oh, 1)
    return out


def _maps_output_dir(aligned_dir: Path, tmd_path: Path) -> Path:
    """``maps_0mm`` / ``maps_150mm`` when name matches ``circle_<n>mm``; else ``maps_<stem>``."""
    m = re.search(r"circle_(\d+)mm", tmd_path.name)
    if m:
        return aligned_dir / f"maps_{m.group(1)}mm"
    return aligned_dir / f"maps_{tmd_path.stem}"


def _mesh_output_path(aligned_dir: Path, tmd_path: Path, fmt: str) -> Path:
    mesh_dir = aligned_dir / "mesh"
    m = re.match(r"(circle_\d+mm)", tmd_path.stem)
    base = m.group(1) if m else tmd_path.stem.replace("_aligned", "")
    return mesh_dir / f"{base}_mesh.{fmt.lower()}"


def create_sequence_app() -> typer.Typer:
    app = typer.Typer(help="Align and export sequences of TMD height maps")

    @app.command("align")
    def align_command(
        tmd_files: List[Path] = typer.Argument(
            ...,
            help="TMD paths in time / scan order (reference index selects the fixed frame)",
            exists=True,
        ),
        output_dir: Path = typer.Option(
            ...,
            "--output-dir",
            "-o",
            help="Directory for aligned .tmd files and optional JSON sidecar",
        ),
        reference_index: int = typer.Option(
            0,
            "--reference",
            "-r",
            help="Index of the reference frame (others align to it)",
        ),
        method: str = typer.Option(
            "auto",
            "--method",
            "-m",
            help="Primary registration: auto (ORB+RANSAC affine, then phase fallback), "
            "affine_ransac, or phase_correlation (translation only)",
        ),
        refine_phase: bool = typer.Option(
            True,
            "--refine-phase/--no-refine-phase",
            help="After primary method: sub-pixel phase correlation on each non-reference frame",
        ),
        second_full_pass: bool = typer.Option(
            False,
            "--second-pass/--no-second-pass",
            help="Run a full second alignment on the first-pass result (TextureFriction-style; slow on large rasters)",
        ),
        orb_nfeatures: int = typer.Option(
            2500,
            "--orb-nfeatures",
            help="ORB keypoint budget (auto / affine_ransac); increase for large, textured scans",
        ),
        min_inliers: int = typer.Option(
            6,
            "--min-inliers",
            help="Minimum RANSAC inliers to accept an affine model (auto / affine_ransac)",
        ),
        ransac_reproj_threshold: float = typer.Option(
            3.0,
            "--ransac-reproj-threshold",
            help="RANSAC reprojection threshold in pixels (auto / affine_ransac)",
        ),
        crop: bool = typer.Option(
            True,
            "--crop/--no-crop",
            help="After warping, crop to valid overlap (intersection of in-bounds pixels)",
        ),
        margin: int = typer.Option(
            0,
            "--margin",
            help="Shrink crop box by this many pixels on each side (after overlap)",
        ),
        version: int = typer.Option(2, "--version", "-v", help="TMD file version to write (1 or 2)"),
        save_json: bool = typer.Option(
            True,
            "--save-json/--no-save-json",
            help="Write alignment_info.json with crop slices and per-frame transforms",
        ),
        registration_channel: str = typer.Option(
            "gradient",
            "--registration-channel",
            help=(
                "Image channel for matching: height (raw), gradient (Sobel magnitude; "
                "recommended for periodic GelSight domes), or detail (Gaussian high-pass)"
            ),
        ),
    ) -> None:
        """
        Load several TMDs as a sequence, align them with OpenCV, optionally crop to overlap,
        and write one aligned .tmd per input (same stem + ``_aligned.tmd``).
        """
        if len(tmd_files) < 2:
            console.print("[red]Need at least two TMD files.[/]")
            raise typer.Exit(1)
        if registration_channel not in ("height", "gradient", "detail"):
            console.print(
                "[red]--registration-channel must be one of: height, gradient, detail[/]"
            )
            raise typer.Exit(1)
        if not (0 <= reference_index < len(tmd_files)):
            console.print("[red]reference index out of range[/]")
            raise typer.Exit(1)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        seq = TMDSequence(name="cli_sequence")
        ref_meta: Dict[str, Any] = {}
        stems: List[str] = []

        with console.status("[cyan]Loading TMD frames…[/]"):
            for i, p in enumerate(tmd_files):
                p = Path(p)
                stems.append(p.stem)
                data = TMD.load(p)
                if i == reference_index:
                    ref_meta = dict(data.metadata or {})
                seq.add_frame(
                    data.height_map.copy(),
                    timestamp=p.stem,
                    metadata=dict(data.metadata or {}),
                )

        console.print(
            f"[cyan]Aligning[/] {len(tmd_files)} frames "
            f"(ref={reference_index}, method={method!r}, channel={registration_channel!r}, "
            f"crop={crop}, margin={margin}, refine_phase={refine_phase}, second_pass={second_full_pass})…"
        )
        try:
            info = seq.align_height_maps_opencv(
                reference_index=reference_index,
                method=method,
                crop=crop,
                margin=margin,
                phase_refine=refine_phase,
                second_full_pass=second_full_pass,
                orb_nfeatures=orb_nfeatures,
                min_inliers=min_inliers,
                ransac_reproj_threshold=ransac_reproj_threshold,
                registration_channel=registration_channel,
            )
        except ImportError as e:
            console.print(f"[red]{e}[/]")
            raise typer.Exit(1) from e
        except Exception as e:
            console.print(f"[red]Alignment failed:[/] {e}")
            raise typer.Exit(1) from e

        nh, nw = seq.frames[0].shape
        spatial = _spatial_fields_from_reference(ref_meta, nh, nw)
        comment = "Aligned TMD seq\n"

        written: List[str] = []
        for stem, arr in zip(stems, seq.frames):
            out_path = output_dir / f"{stem}_aligned.tmd"
            TMDUtils.write_tmd_file(
                arr,
                out_path,
                comment=comment,
                x_length=float(spatial.get("x_length", 10.0)),
                y_length=float(spatial.get("y_length", 10.0)),
                x_offset=float(spatial.get("x_offset", 0.0)),
                y_offset=float(spatial.get("y_offset", 0.0)),
                version=version,
            )
            written.append(str(out_path))

        if save_json:
            json_path = output_dir / "alignment_info.json"

            def _json_default(o: Any) -> Any:
                if isinstance(o, slice):
                    return {"start": o.start, "stop": o.stop, "step": o.step}
                raise TypeError

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2, default=_json_default)
            console.print(f"[green]Wrote[/] {json_path}")

        console.print(f"[green]Aligned shape:[/] {nw}×{nh}")
        for w in written:
            console.print(f"[green]Saved[/] {w}")

    @app.command("export")
    def export_command(
        aligned_dir: Path = typer.Argument(
            ...,
            exists=True,
            file_okay=False,
            help="Directory containing ``*_aligned.tmd`` files (e.g. output of ``sequence align``)",
        ),
        maps_enable: bool = typer.Option(True, "--maps/--no-maps", help="Run ``maps all`` per aligned file"),
        mesh_enable: bool = typer.Option(True, "--mesh/--no-mesh", help="Generate a mesh per aligned file"),
        map_fast: bool = typer.Option(
            True,
            "--map-fast/--full-maps",
            help="When --map-fast, omit parallax_ao from the default map set",
        ),
        compress: int = typer.Option(0, "--compress", "-c", help="Map PNG compression 0–100"),
        normalize: bool = typer.Option(
            False,
            "--normalize",
            "-n",
            help="Normalize height before map export",
        ),
        mesh_format: str = typer.Option("stl", "--mesh-format", help="Mesh format (stl, obj, ply, …)"),
        mesh_quality: QualityPreset = typer.Option(QualityPreset.DRAFT, "--mesh-quality"),
        mesh_method: MeshMethod = typer.Option(MeshMethod.QUADTREE, "--mesh-method"),
        scale: float = typer.Option(5.0, "--scale", help="Mesh height (z) scale"),
        max_subdivisions: int = typer.Option(12, "--max-subdivisions", help="Quadtree depth cap for mesh"),
        save_heightmap: bool = typer.Option(
            False,
            "--save-heightmap/--no-save-heightmap",
            help="Also write *_heightmap.png next to each mesh (STL)",
        ),
        template_mesh: Path = typer.Option(
            None,
            "--template-mesh",
            help="Optional template OBJ to apply maps onto (separate apply-on-mesh flow).",
        ),
        template_kind: str = typer.Option(
            "plane",
            "--template-kind",
            help="Built-in template kind for apply flow: plane, sphere, cube.",
        ),
        template_fixtures_dir: Path = typer.Option(
            None,
            "--template-fixtures-dir",
            help="Override built-in template fixtures root directory.",
        ),
        template_plane_dir: Path = typer.Option(
            None,
            "--template-plane-dir",
            help="Template dir containing plane.obj for apply-on-mesh flow.",
        ),
        apply_mode: str = typer.Option("uv", "--apply-mode", help="Apply mode: uv or displace"),
        uv_alignment_mode: str = typer.Option(
            "preserve",
            "--uv-alignment-mode",
            help="UV behavior: preserve (default) or remap_bbox.",
        ),
        obj_units_to_mm: float = typer.Option(
            1000.0,
            "--obj-units-to-mm",
            help="Template OBJ units to millimeters conversion factor (meters=1000).",
        ),
        tmd_mm_per_pixel: Optional[float] = typer.Option(
            None,
            "--tmd-mm-per-pixel",
            help="Override mm-per-pixel physical scale used for atlas/tile sizing.",
        ),
    ) -> None:
        """
        For every ``*_aligned.tmd`` in a directory (sorted by name): export all maps into
        ``maps_<n>mm`` (GelSight-style names) or ``maps_<stem>``, and write meshes under ``mesh/``.
        """
        aligned_dir = Path(aligned_dir)
        files = sorted(aligned_dir.glob("*_aligned.tmd"))
        if not files:
            console.print(f"[red]No *_aligned.tmd files in {aligned_dir}[/]")
            raise typer.Exit(1)

        console.print(f"[cyan]Found {len(files)} aligned TMD(s) in {aligned_dir}[/]")

        for fp in files:
            if maps_enable:
                out_maps = _maps_output_dir(aligned_dir, fp)
                out_maps.mkdir(parents=True, exist_ok=True)
                console.print(f"\n[cyan]Maps[/] {fp.name} → {out_maps}")
                ok = export_maps_command(
                    fp,
                    out_maps,
                    None,
                    compress=compress,
                    format="png",
                    normalize=normalize,
                    fast=map_fast,
                )
                if not ok:
                    raise typer.Exit(1)

            if mesh_enable:
                if template_mesh or template_plane_dir:
                    bundle_dir = aligned_dir / "mesh" / fp.stem
                    console.print(f"\n[cyan]Apply[/] {fp.name} → {bundle_dir}")
                    apply_maps_to_mesh(
                        tmd_file=fp,
                        output_root=bundle_dir,
                        template_mesh_path=template_mesh,
                        template_plane_dir=template_plane_dir,
                        template_kind=template_kind,
                        template_fixtures_dir=template_fixtures_dir,
                        output_prefix=fp.stem.replace("_aligned", ""),
                        application_mode=apply_mode,
                        uv_alignment_mode=uv_alignment_mode,
                        obj_units_to_mm=obj_units_to_mm,
                        tmd_mm_per_pixel=tmd_mm_per_pixel,
                        compress=compress,
                        normalize=normalize,
                    )
                else:
                    mesh_out = _mesh_output_path(aligned_dir, fp, mesh_format)
                    mesh_out.parent.mkdir(parents=True, exist_ok=True)
                    qp = get_quality_params(mesh_quality)
                    qp["max_subdivisions"] = max_subdivisions
                    console.print(f"\n[cyan]Mesh[/] {fp.name} → {mesh_out}")
                    ok = export_model(
                        input_file=fp,
                        output_file=mesh_out,
                        format=mesh_format,
                        method=mesh_method.value,
                        scale=scale,
                        binary=True,
                        coordinate_system="right-handed",
                        optimize=True,
                        save_heightmap=save_heightmap,
                        colormap="terrain",
                        base_height=0.0,
                        **qp,
                    )
                    if not ok:
                        raise typer.Exit(1)

        console.print(f"\n[green]Sequence export finished for {len(files)} file(s).[/]")

    return app
