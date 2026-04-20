"""CLI commands for surface defect detection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import typer
from rich.table import Table

from tmd import TMD
from tmd.cli.core.ui import console
from tmd.image.core.image_utils import save_image
from tmd.surface.types import DefectDetectionConfig


def _sanitize_json_values(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_json_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_values(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def _payload_from_result(path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    defects = {}
    for name, entry in result["defects"].items():
        defects[name] = {
            "count": int(entry["count"]),
            "confidence": float(entry["confidence"]),
            "areas": [int(a) for a in entry.get("areas", [])],
        }
    return {
        "file": str(path.resolve()),
        "summary": {
            "total_count": int(result["summary"]["total_count"]),
            "global_confidence": float(result["summary"]["global_confidence"]),
            "class_counts": {k: int(v) for k, v in result["summary"]["class_counts"].items()},
        },
        "defects": defects,
    }


def _config_from_options(
    *,
    gaussian_sigma: float,
    zscore_threshold: float,
    min_area: int,
    min_confidence: float,
    directionality_window: int,
    directionality_angle_threshold_deg: float,
) -> DefectDetectionConfig:
    return DefectDetectionConfig(
        gaussian_sigma=gaussian_sigma,
        zscore_threshold=zscore_threshold,
        min_area=min_area,
        min_confidence=min_confidence,
        directionality_window=directionality_window,
        directionality_angle_threshold_deg=directionality_angle_threshold_deg,
    )


def create_defect_app() -> typer.Typer:
    app = typer.Typer(help="Detect pits, peaks, scratches, cracks and directionality anomalies")

    @app.command("file")
    def file_command(
        path: Path = typer.Argument(..., help="Path to a .tmd file", exists=True, readable=True),
        json_out: bool = typer.Option(False, "--json", help="Print JSON instead of rich table"),
        gaussian_sigma: float = typer.Option(1.0, "--gaussian-sigma", help="Gaussian smoothing sigma"),
        zscore_threshold: float = typer.Option(1.8, "--zscore-threshold", help="Z-score threshold"),
        min_area: int = typer.Option(6, "--min-area", help="Minimum connected defect area in pixels"),
        min_confidence: float = typer.Option(
            0.0, "--min-confidence", help="Drop classes below this confidence [0,1]"
        ),
        directionality_window: int = typer.Option(
            11, "--directionality-window", help="Window size for local directionality analysis"
        ),
        directionality_angle_threshold_deg: float = typer.Option(
            22.5, "--directionality-angle-threshold-deg", help="Angular anomaly threshold in degrees"
        ),
        mask_output: Optional[Path] = typer.Option(
            None, "--mask-output", help="Output path for combined defect mask image"
        ),
        overlay_output: Optional[Path] = typer.Option(
            None, "--overlay-output", help="Output path for RGB overlay image"
        ),
        include_mask: bool = typer.Option(
            False,
            "--include-mask",
            help="Compute and include class masks in analysis (slower)",
        ),
        include_overlay: bool = typer.Option(
            False,
            "--include-overlay",
            help="Compute combined labels/overlay arrays in analysis (slower)",
        ),
        include_responses: bool = typer.Option(
            False,
            "--include-responses",
            help="Compute and include per-class response arrays (slower)",
        ),
    ) -> None:
        """Run defect detection on a single TMD file."""
        if path.suffix.lower() != ".tmd":
            console.print("[red]Input must be a .tmd file.[/]")
            raise typer.Exit(1)

        config = _config_from_options(
            gaussian_sigma=gaussian_sigma,
            zscore_threshold=zscore_threshold,
            min_area=min_area,
            min_confidence=min_confidence,
            directionality_window=directionality_window,
            directionality_angle_threshold_deg=directionality_angle_threshold_deg,
        )

        # Fast default: summary mode for CLI unless heavy outputs are explicitly requested.
        output_mode = "summary"
        if include_mask or include_responses:
            output_mode = "standard"
        if include_overlay or mask_output is not None or overlay_output is not None:
            output_mode = "full"

        tmd_data = TMD.load(path, compute_initial_stats=False)
        result = tmd_data.analyze_defects(
            **{
                **config.__dict__,
                "output_mode": output_mode,
                "include_responses": include_responses,
            }
        )
        payload = _payload_from_result(path, result)

        if mask_output is not None:
            if "mask" not in result:
                console.print("[red]Mask output requested but no mask data was generated.[/]")
                raise typer.Exit(1)
            mask_output.parent.mkdir(parents=True, exist_ok=True)
            mask_img = ((result["mask"] > 0).astype(np.uint8) * 255).astype(np.uint8, copy=False)
            ok = save_image(mask_img, str(mask_output), normalize=False, format="png")
            if not ok:
                console.print(f"[red]Failed to save mask image:[/] {mask_output}")
                raise typer.Exit(1)

        if overlay_output is not None:
            if "overlay_rgb" not in result:
                console.print("[red]Overlay output requested but no overlay data was generated.[/]")
                raise typer.Exit(1)
            overlay_output.parent.mkdir(parents=True, exist_ok=True)
            ok = save_image(result["overlay_rgb"], str(overlay_output), normalize=False, format="png")
            if not ok:
                console.print(f"[red]Failed to save overlay image:[/] {overlay_output}")
                raise typer.Exit(1)

        if json_out:
            typer.echo(json.dumps(_sanitize_json_values(payload), indent=2, allow_nan=False))
            return

        table = Table(title=f"Defect Detection — {path.name}")
        table.add_column("Defect class", style="cyan")
        table.add_column("Count", justify="right")
        table.add_column("Confidence", justify="right")
        for name in (
            "pits",
            "peaks",
            "scratches",
            "cracks",
            "directionality_anomalies",
        ):
            entry = payload["defects"][name]
            table.add_row(name, str(entry["count"]), f"{entry['confidence']:.3f}")
        table.add_row("total", str(payload["summary"]["total_count"]), f"{payload['summary']['global_confidence']:.3f}")
        console.print(table)

    @app.command("batch")
    def batch_command(
        directory: Path = typer.Argument(
            ...,
            help="Directory containing .tmd files",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
        pattern: str = typer.Option("*.tmd", "--pattern", help="Glob pattern"),
        recursive: bool = typer.Option(False, "--recursive", "-r", help="Search subdirectories"),
        output: Optional[Path] = typer.Option(
            None,
            "--output",
            "-o",
            help="Write CSV output file (default: print CSV to stdout)",
        ),
        json_out: bool = typer.Option(False, "--json", help="Print JSON list instead of CSV"),
        min_confidence: float = typer.Option(
            0.0, "--min-confidence", help="Drop classes below this confidence [0,1]"
        ),
    ) -> None:
        """Run defect detection on many TMD files."""
        iterator = directory.rglob(pattern) if recursive else directory.glob(pattern)
        paths = sorted(p for p in iterator if p.is_file() and p.suffix.lower() == ".tmd")
        if not paths:
            console.print(f"[yellow]No .tmd matched {pattern!r} under {directory}[/]")
            raise typer.Exit(0)

        rows: List[Dict[str, Any]] = []
        config = DefectDetectionConfig(min_confidence=min_confidence)
        for path in paths:
            try:
                tmd_data = TMD.load(path, compute_initial_stats=False)
                result = tmd_data.analyze_defects(
                    **{
                        **config.__dict__,
                        "output_mode": "summary",
                        "include_responses": False,
                    }
                )
                row = _payload_from_result(path, result)
                counts = row["summary"]["class_counts"]
                rows.append(
                    {
                        "file": path.name,
                        "total_count": int(row["summary"]["total_count"]),
                        "global_confidence": float(row["summary"]["global_confidence"]),
                        "pits": int(counts["pits"]),
                        "peaks": int(counts["peaks"]),
                        "scratches": int(counts["scratches"]),
                        "cracks": int(counts["cracks"]),
                        "directionality_anomalies": int(counts["directionality_anomalies"]),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "file": path.name,
                        "total_count": 0,
                        "global_confidence": 0.0,
                        "pits": 0,
                        "peaks": 0,
                        "scratches": 0,
                        "cracks": 0,
                        "directionality_anomalies": 0,
                        "__error__": str(exc),
                    }
                )

        if json_out:
            typer.echo(json.dumps(_sanitize_json_values(rows), indent=2, allow_nan=False))
            return

        fieldnames = [
            "file",
            "total_count",
            "global_confidence",
            "pits",
            "peaks",
            "scratches",
            "cracks",
            "directionality_anomalies",
            "__error__",
        ]
        lines = []
        import io

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        lines = buffer.getvalue()

        if output is None:
            typer.echo(lines, nl=False)
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(lines, encoding="utf-8")
            console.print(f"[green]Wrote[/] {output} ({len(rows)} rows)")

    return app
