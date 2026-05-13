"""Typer CLI for wear-oriented surface metrics (`tmd-wear`)."""

from __future__ import annotations

import csv
import io
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import typer

from tmd import TMD
from tmd.cli.apps.roughness_app import _roughness_rows_for_paths, _surfalize_imports
from tmd.cli.core import check_dependencies
from tmd.cli.core.ui import console
from tmd.core.sequence import TMDSequence
from tmd.image.core.image_utils import save_image
from tmd.surface.metrics import bearing_analysis, debris_pocket_score, shear_proxy_uint8
from tmd.sequence.wear_analysis import (
    append_trajectory_derivatives_batch,
    scratch_evolution_pair,
    scratch_series_metrics,
    slip_axis_metrics,
    ssk_trend_heuristic,
    wear_incremental_series,
    wear_series_vs_reference,
)
from tmd.surface.defects import detect_surface_defects
from tmd.surface.types import DefectDetectionConfig

_PARAM_SET_TRAJECTORY = "Sa,Sq,Sp,Sv,Sz,Ssk,Sku"


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def _pixel_pitch_mm(data: TMD) -> Tuple[float, float]:
    hm = data.height_map
    h, w = hm.shape
    meta = data.metadata or {}
    width = int(meta.get("width", w))
    height = int(meta.get("height", h))
    xl = float(meta.get("x_length", 10.0))
    yl = float(meta.get("y_length", 10.0))
    dx = xl / max(width, 1)
    dy = yl / max(height, 1)
    return dx, dy


def _ordered_tmd_paths(
    paths: Optional[List[Path]],
    from_dir: Optional[Path],
    pattern: str,
    sort_by: str,
) -> List[Path]:
    if from_dir is not None and paths:
        console.print("[red]Use either explicit PATHS or --from-dir, not both.[/]")
        raise typer.Exit(1)
    ordered: List[Path] = []
    if from_dir is not None:
        base = Path(from_dir)
        found = [p for p in base.glob(pattern) if p.is_file() and p.suffix.lower() == ".tmd"]
        if sort_by == "mtime":
            found.sort(key=lambda p: p.stat().st_mtime)
        elif sort_by == "name":
            found.sort(key=lambda p: p.name.lower())
        else:
            console.print("[red]--sort-by must be 'name' or 'mtime'[/]")
            raise typer.Exit(1)
        ordered = found
    else:
        ordered = [Path(p) for p in (paths or [])]

    if not ordered:
        console.print("[red]Provide at least one .tmd path, or use --from-dir[/]")
        raise typer.Exit(1)
    for p in ordered:
        if not p.is_file():
            console.print(f"[red]Not a file: {p}[/]")
            raise typer.Exit(1)
        if p.suffix.lower() != ".tmd":
            console.print(f"[red]Not a .tmd file: {p}[/]")
            raise typer.Exit(1)
    return ordered


def _align_stack_if_requested(
    seq: TMDSequence,
    *,
    align: str,
    reference_index: int,
) -> Optional[Dict[str, Any]]:
    """Run in-place alignment when ``align`` is not ``none``; return alignment info dict or None."""
    key = (align or "none").strip().lower().replace("_", "-")
    if key in ("", "none", "off", "false", "no"):
        return None
    if key in ("phase-fft", "phasefft", "fft"):
        return seq.align_height_maps_phase_fft(reference_index=reference_index)
    if key in ("opencv", "cv", "ocv"):
        return seq.align_height_maps_opencv(reference_index=reference_index)
    if key in ("sift", "sift-height", "texture-friction", "texture-friction-height"):
        return seq.align_height_maps_sift(reference_index=reference_index)
    if key in ("sift-normals", "sift_normals", "texture-friction-normals"):
        return seq.align_height_maps_from_normals(reference_index=reference_index)
    console.print(
        f"[red]Unknown --align value: {align!r} (use none, phase-fft, opencv, sift, sift-normals)[/]"
    )
    raise typer.Exit(1)


def _sequence_from_tmd_paths(ordered: List[Path], *, name: str = "wear-stack") -> TMDSequence:
    seq = TMDSequence(name=name)
    for p in ordered:
        t = TMD.load(p, compute_initial_stats=False)
        seq.add_frame(np.asarray(t.height_map, dtype=np.float64), p.name, t.metadata or {})
    return seq


def create_wear_app() -> typer.Typer:
    app = typer.Typer(
        help="Wear-oriented surface metrics (Abbott curve, wear volume, hazard maps, …)",
        add_completion=False,
    )

    @app.callback()
    def _wear_callback() -> None:
        check_dependencies(auto_install=False, exit_on_failure=True)

    bearing = typer.Typer(help="Abbott–Firestone / material ratio (plane remove + bearing curve)")
    app.add_typer(bearing, name="bearing")

    @bearing.command("curve")
    def bearing_curve_cmd(
        path: Path = typer.Argument(..., exists=True, help="Path to a .tmd file"),
        json_out: bool = typer.Option(False, "--json", help="Print JSON to stdout"),
        samples: int = typer.Option(256, "--samples", help="Number of depth samples along Abbott curve"),
        rmr_at_depth: Optional[str] = typer.Option(
            None,
            "--rmr-at-depth",
            help="Comma-separated cut depths (same units as height map, typically mm)",
        ),
        output_csv: Optional[Path] = typer.Option(
            None,
            "--output-csv",
            "-o",
            help="Write depth,rmr_percent CSV instead of stdout table",
        ),
    ) -> None:
        """Plane-remove heights and sample material ratio vs depth from the peak."""
        data = TMD.load(path, compute_initial_stats=False)
        z = data.height_map
        qdepths: Optional[List[float]] = None
        if rmr_at_depth and rmr_at_depth.strip():
            qdepths = [float(x.strip()) for x in rmr_at_depth.split(",") if x.strip()]
        payload = bearing_analysis(z, n_depth_samples=samples, rmr_query_depths=qdepths)
        payload["file"] = str(path.resolve())

        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            depths = payload["depths"]
            rmr = payload["rmr_percent"]
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["depth", "rmr_percent"])
                for d, r in zip(depths, rmr):
                    w.writerow([d, r])
            console.print(f"[green]Wrote[/] {output_csv}")
            return

        if json_out:
            typer.echo(json.dumps(_sanitize_json(payload), indent=2, allow_nan=False))
            return

        console.print(f"[cyan]Bearing curve[/] {path.name} — depths: {len(payload['depths'])} samples")
        if "rmr_at_depth" in payload:
            console.print(f"[dim]{payload['rmr_at_depth']}[/]")

    @app.command("roughness-track")
    def roughness_track_cmd(
        paths: Optional[List[Path]] = typer.Argument(None, help=".tmd files in frame order"),
        from_dir: Optional[Path] = typer.Option(None, "--from-dir", exists=True, file_okay=False),
        pattern: str = typer.Option("*.tmd", "--pattern"),
        sort_by: str = typer.Option("name", "--sort-by"),
        output: Optional[Path] = typer.Option(None, "--output", "-o"),
        json_out: bool = typer.Option(False, "--json"),
        level: bool = typer.Option(True, "--level/--no-level"),
    ) -> None:
        """Surfalize roughness trajectory + Sp/Sv ratio and valley_share (requires surfalize)."""
        ordered = _ordered_tmd_paths(paths, from_dir, pattern, sort_by)
        _surfalize_imports()
        base_rows = _roughness_rows_for_paths(
            ordered,
            level=level,
            quick=False,
            params=_PARAM_SET_TRAJECTORY,
            all_params=False,
            include_frame_index=True,
            include_full_path=True,
        )
        rows = append_trajectory_derivatives_batch(base_rows)
        rows_out: List[Dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["file"] = Path(str(d.get("path", d.get("file", "")))).name if d.get("path") else d.get("file")
            rows_out.append(d)

        hint = ssk_trend_heuristic(rows)
        if json_out:
            typer.echo(json.dumps(_sanitize_json({"rows": rows_out, "ssk_trend_hint": hint}), indent=2, allow_nan=False))
            return

        try:
            import pandas as pd
        except ImportError as e:
            console.print("[red]roughness-track tabular output requires pandas.[/]")
            raise typer.Exit(1) from e

        df = pd.DataFrame(rows_out)
        if output is not None:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            console.print(f"[green]Wrote[/] {out_path}")
        else:
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            sys.stdout.write(buf.getvalue())
        console.print(f"[dim]{hint}[/]")

    @app.command("hazard-map")
    def hazard_map_cmd(
        path: Path = typer.Argument(..., exists=True),
        out: Path = typer.Option(..., "--out", "-o", help="Output PNG path"),
        window: int = typer.Option(7, "--window", help="Odd window size for local RMS"),
        normalize: str = typer.Option("p98", "--normalize", help="none | minmax | p98"),
    ) -> None:
        """Relative shear-hazard proxy map (|grad|·|Laplace|·local RMS), saved as PNG."""
        if normalize not in ("none", "minmax", "p98"):
            console.print("[red]--normalize must be one of: none, minmax, p98[/]")
            raise typer.Exit(1)
        if window < 3 or window % 2 == 0:
            console.print("[red]--window must be an odd integer >= 3[/]")
            raise typer.Exit(1)
        data = TMD.load(path, compute_initial_stats=False)
        _, u8 = shear_proxy_uint8(data.height_map, window=window, normalize=normalize)  # type: ignore[arg-type]
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        ok = save_image(u8, str(out), normalize=False, format="png")
        if not ok:
            console.print(f"[red]Failed to write[/] {out}")
            raise typer.Exit(1)
        console.print(f"[green]Wrote[/] {out}")

    @app.command("debris-risk")
    def debris_risk_cmd(
        path: Path = typer.Argument(..., exists=True),
        json_out: bool = typer.Option(False, "--json"),
        score_png: Optional[Path] = typer.Option(None, "--score-png", help="Optional heatmap PNG"),
    ) -> None:
        """Heuristic debris-pocket score from pits + valleys + low gradient."""
        data = TMD.load(path, compute_initial_stats=False)
        score, meta = debris_pocket_score(data.height_map)
        payload = {"file": str(path.resolve()), "summary": meta}
        if score_png is not None:
            u8 = (np.clip(score, 0.0, 1.0) * 255.0).astype(np.uint8)
            score_png = Path(score_png)
            score_png.parent.mkdir(parents=True, exist_ok=True)
            if not save_image(u8, str(score_png), normalize=False, format="png"):
                console.print(f"[red]Failed to write[/] {score_png}")
                raise typer.Exit(1)
            payload["score_png"] = str(score_png.resolve())
        if json_out:
            typer.echo(json.dumps(_sanitize_json(payload), indent=2, allow_nan=False))
            return
        console.print(f"[cyan]Debris pocket heuristic[/] {path.name}")
        for k, v in meta.items():
            console.print(f"  {k}: {v}")

    @app.command("volume-series")
    def volume_series_cmd(
        paths: Optional[List[Path]] = typer.Argument(None, help="Aligned .tmd files (same grid)"),
        from_dir: Optional[Path] = typer.Option(None, "--from-dir", exists=True, file_okay=False),
        pattern: str = typer.Option("*.tmd", "--pattern"),
        sort_by: str = typer.Option("name", "--sort-by"),
        reference: int = typer.Option(
            0,
            "--reference",
            "-r",
            help="Reference frame index; loss = z_ref - z_i (positive = lower than ref)",
        ),
        incremental: bool = typer.Option(
            False,
            "--incremental",
            help="Use consecutive-frame loss z_{i-1}-z_i instead of vs reference",
        ),
        signed: bool = typer.Option(False, "--signed", help="Include signed volume vs reference"),
        top_fraction: float = typer.Option(
            0.10,
            "--top-fraction",
            help="Localization: top fraction of pixels by positive loss",
        ),
        cycles: Optional[str] = typer.Option(
            None,
            "--cycles",
            help="Comma-separated cycle counts (must match number of frames)",
        ),
        times: Optional[str] = typer.Option(
            None,
            "--times",
            help="Comma-separated times (must match number of frames)",
        ),
        align: str = typer.Option(
            "none",
            "--align",
            help="Stack alignment before metrics: none | phase-fft | opencv | sift | sift-normals (OpenCV; same shape)",
        ),
        output: Optional[Path] = typer.Option(None, "--output", "-o"),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Wear volume and localization on a stack (optionally align with --align first)."""
        ordered = _ordered_tmd_paths(paths, from_dir, pattern, sort_by)
        seq = _sequence_from_tmd_paths(ordered)
        align_info = _align_stack_if_requested(seq, align=align, reference_index=reference)
        frames = [np.asarray(f, dtype=np.float64) for f in seq.frames]
        dx, dy = _pixel_pitch_mm(TMD.load(ordered[0], compute_initial_stats=False))

        if incremental:
            rows = wear_incremental_series(frames, dx=dx, dy=dy, top_fraction=top_fraction)
        else:
            rows = wear_series_vs_reference(
                frames,
                reference_index=reference,
                dx=dx,
                dy=dy,
                top_fraction=top_fraction,
                signed=signed,
            )

        if cycles and cycles.strip():
            cvals = [float(x.strip()) for x in cycles.split(",") if x.strip()]
            if len(cvals) != len(rows):
                console.print("[red]--cycles length must match number of frames[/]")
                raise typer.Exit(1)
            for r, c in zip(rows, cvals):
                r["cycle"] = c
        if times and times.strip():
            tvals = [float(x.strip()) for x in times.split(",") if x.strip()]
            if len(tvals) != len(rows):
                console.print("[red]--times length must match number of frames[/]")
                raise typer.Exit(1)
            for r, t in zip(rows, tvals):
                r["time"] = t

        # Wear rate: incremental mode uses step volume; vs-ref uses delta of cumulative-style volume vs ref
        if incremental:
            for i, r in enumerate(rows):
                if i == 0:
                    continue
                dv = float(r.get("volume_positive_incremental", 0.0))
                if "cycle" in r and rows[i - 1].get("cycle") is not None and r.get("cycle") is not None:
                    dc = float(r["cycle"]) - float(rows[i - 1]["cycle"])
                    r["wear_rate_dV_dcycle"] = dv / dc if dc != 0 else float("nan")
                if "time" in r and rows[i - 1].get("time") is not None and r.get("time") is not None:
                    dt = float(r["time"]) - float(rows[i - 1]["time"])
                    r["wear_rate_dV_dt"] = dv / dt if dt != 0 else float("nan")
        elif len(rows) > 1:
            for i in range(1, len(rows)):
                dv = float(rows[i].get("volume_positive_loss", 0.0)) - float(
                    rows[i - 1].get("volume_positive_loss", 0.0)
                )
                rows[i]["delta_volume_positive_loss"] = dv
                if "cycle" in rows[i] and rows[i - 1].get("cycle") is not None and rows[i].get("cycle") is not None:
                    dc = float(rows[i]["cycle"]) - float(rows[i - 1]["cycle"])
                    rows[i]["wear_rate_dV_dcycle"] = dv / dc if dc != 0 else float("nan")
                if "time" in rows[i] and rows[i - 1].get("time") is not None and rows[i].get("time") is not None:
                    dt = float(rows[i]["time"]) - float(rows[i - 1]["time"])
                    rows[i]["wear_rate_dV_dt"] = dv / dt if dt != 0 else float("nan")

        payload = {"loss_convention": "z_ref - z_i (positive => current lower than reference)", "rows": rows}
        if align_info is not None:
            payload["alignment"] = align_info
        if json_out:
            typer.echo(json.dumps(_sanitize_json(payload), indent=2, allow_nan=False))
            return
        try:
            import pandas as pd
        except ImportError as e:
            console.print("[red]volume-series tabular output requires pandas.[/]")
            raise typer.Exit(1) from e

        df = pd.DataFrame(rows)
        if output is not None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(Path(output), index=False)
            console.print(f"[green]Wrote[/] {output}")
        else:
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            sys.stdout.write(buf.getvalue())

    @app.command("scratch-evolve")
    def scratch_evolve_cmd(
        paths: Optional[List[Path]] = typer.Argument(None, help="Aligned .tmd files (same grid)"),
        from_dir: Optional[Path] = typer.Option(None, "--from-dir", exists=True, file_okay=False),
        pattern: str = typer.Option("*.tmd", "--pattern"),
        sort_by: str = typer.Option("name", "--sort-by"),
        reference: int = typer.Option(
            0,
            "--reference",
            "-r",
            help="Reference frame index for optional stack alignment",
        ),
        align: str = typer.Option(
            "none",
            "--align",
            help="Stack alignment before scratch masks: none | phase-fft | opencv | sift | sift-normals",
        ),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Sequential scratch-mask evolution (defects standard mode)."""
        ordered = _ordered_tmd_paths(paths, from_dir, pattern, sort_by)
        seq = _sequence_from_tmd_paths(ordered, name="scratch-stack")
        align_info = _align_stack_if_requested(seq, align=align, reference_index=reference)
        cfg = DefectDetectionConfig(output_mode="standard")
        masks: List[np.ndarray] = []
        for i, _p in enumerate(ordered):
            fr = seq.get_frame(i)
            if fr is None:
                raise RuntimeError("missing frame")
            res = detect_surface_defects(np.asarray(fr, dtype=np.float32), cfg)
            m = res["defects"]["scratches"].get("mask")
            if m is None:
                console.print("[red]scratch mask missing; ensure standard output_mode[/]")
                raise typer.Exit(1)
            masks.append(np.asarray(m, dtype=bool))
        evolutions: List[Dict[str, Any]] = []
        for i in range(1, len(masks)):
            ev = scratch_evolution_pair(masks[i - 1], masks[i])
            ev["file_before"] = ordered[i - 1].name
            ev["file_after"] = ordered[i].name
            evolutions.append(ev)
        series = scratch_series_metrics(masks)
        payload: Dict[str, Any] = {"pairs": evolutions, "series": series}
        if align_info is not None:
            payload["alignment"] = align_info
        if json_out:
            typer.echo(json.dumps(_sanitize_json(payload), indent=2, allow_nan=False))
            return
        console.print(f"[cyan]Scratch evolution[/] {len(evolutions)} pair(s)")
        for ev in evolutions:
            console.print(f"  {ev['file_before']} → {ev['file_after']}: growth={ev['growth_pixels']}")

    @app.command("slip-axis")
    def slip_axis_cmd(
        path: Path = typer.Argument(..., exists=True),
        json_out: bool = typer.Option(False, "--json"),
        use_directionality_mask: bool = typer.Option(
            False,
            "--use-directionality-mask",
            help="Mask gradient tensor using defect directionality_anomalies mask",
        ),
    ) -> None:
        """Preferred slip axis heuristics (gradient tensor + PSD wedges)."""
        data = TMD.load(path, compute_initial_stats=False)
        dmask: Optional[np.ndarray] = None
        if use_directionality_mask:
            cfg = DefectDetectionConfig(output_mode="standard")
            res = detect_surface_defects(data.height_map.astype(np.float32), cfg)
            raw = res["defects"]["directionality_anomalies"].get("mask")
            if raw is None:
                console.print("[red]directionality mask missing[/]")
                raise typer.Exit(1)
            dmask = np.asarray(raw, dtype=bool)
        metrics = slip_axis_metrics(data.height_map, direction_mask=dmask)
        payload = {"file": str(path.resolve()), **metrics}
        if json_out:
            typer.echo(json.dumps(_sanitize_json(payload), indent=2, allow_nan=False))
            return
        for k, v in payload.items():
            if k != "file":
                console.print(f"  {k}: {v}")

    @app.command("slip-axis-series")
    def slip_axis_series_cmd(
        paths: Optional[List[Path]] = typer.Argument(None, help=".tmd files in frame order (same grid)"),
        from_dir: Optional[Path] = typer.Option(None, "--from-dir", exists=True, file_okay=False),
        pattern: str = typer.Option("*.tmd", "--pattern"),
        sort_by: str = typer.Option("name", "--sort-by"),
        reference: int = typer.Option(
            0,
            "--reference",
            "-r",
            help="Reference frame index for optional stack alignment",
        ),
        align: str = typer.Option(
            "none",
            "--align",
            help="Stack alignment before metrics: none | phase-fft | opencv | sift | sift-normals",
        ),
        use_directionality_mask: bool = typer.Option(
            False,
            "--use-directionality-mask",
            help="Mask gradient tensor using defect directionality_anomalies mask",
        ),
        json_out: bool = typer.Option(False, "--json"),
    ) -> None:
        """Per-frame slip-axis heuristics (gradient tensor + PSD wedges) on a stack."""
        ordered = _ordered_tmd_paths(paths, from_dir, pattern, sort_by)
        seq = _sequence_from_tmd_paths(ordered, name="slip-stack")
        align_info = _align_stack_if_requested(seq, align=align, reference_index=reference)
        cfg = DefectDetectionConfig(output_mode="standard")
        frames_out: List[Dict[str, Any]] = []
        for i, p in enumerate(ordered):
            fr = seq.get_frame(i)
            if fr is None:
                raise RuntimeError("missing frame")
            z = np.asarray(fr, dtype=np.float64)
            dmask: Optional[np.ndarray] = None
            if use_directionality_mask:
                res = detect_surface_defects(z.astype(np.float32), cfg)
                raw = res["defects"]["directionality_anomalies"].get("mask")
                if raw is None:
                    console.print("[red]directionality mask missing[/]")
                    raise typer.Exit(1)
                dmask = np.asarray(raw, dtype=bool)
            metrics = slip_axis_metrics(z, direction_mask=dmask)
            frames_out.append({"frame_index": i, "file": p.name, **metrics})
        payload: Dict[str, Any] = {"frames": frames_out}
        if align_info is not None:
            payload["alignment"] = align_info
        if json_out:
            typer.echo(json.dumps(_sanitize_json(payload), indent=2, allow_nan=False))
            return
        console.print(f"[cyan]Slip-axis series[/] {len(frames_out)} frame(s)")
        for row in frames_out:
            console.print(f"  [{row['frame_index']}] {row['file']}")

    return app
