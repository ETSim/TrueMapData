"""CLI for tribology-oriented metrics (numpy core; lubrication subcommand uses Surfalize / GPL)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer

from tmd import TMD
from tmd.cli.apps import roughness_common as rc
from tmd.cli.core.ui import console
from tmd.surface.metrics import (
    bearing_area_curve,
    preferred_slip_axis,
    save_tribology_dashboard_png,
)


def create_tribology_app() -> typer.Typer:
    app = typer.Typer(help="Tribology metrics from TMD height maps")

    @app.command("axis")
    def axis_command(
        path: Path = typer.Argument(..., help="Path to a .tmd file", exists=True, readable=True),
        json_out: bool = typer.Option(False, "--json", help="Print JSON"),
        gaussian_sigma: float = typer.Option(0.0, help="Pre-smooth height with Gaussian sigma (pixels)"),
        plane_removal: str = typer.Option(
            "none",
            "--plane-removal",
            help="Large-scale plane: none | mean | median | surfalize (Surfalize least-squares; GPL)",
        ),
    ) -> None:
        """Preferred slip / texture axis from gradients, PSD lay, and directionality anomalies."""
        path = Path(path)
        if path.suffix.lower() != ".tmd":
            console.print("[red]Input must be a .tmd file.[/]")
            raise typer.Exit(1)
        data = TMD.load(str(path))
        hm = np.asarray(data.height_map, dtype=np.float32)
        meta = data.metadata or {}
        out = preferred_slip_axis(
            hm,
            meta,
            gaussian_sigma=gaussian_sigma,
            plane_removal=plane_removal,
        )
        payload = {"file": str(path.resolve()), **out}
        if json_out:
            typer.echo(json.dumps(rc.sanitize_json_values(payload), indent=2, allow_nan=False))
        else:
            for k, v in payload.items():
                console.print(f"  {k}: {v}")

    @app.command("contact-curve")
    def contact_curve_command(
        path: Path = typer.Argument(..., help="Path to a .tmd file", exists=True, readable=True),
        n: int = typer.Option(50, "--n", help="Number of separation samples"),
        z_reference: str = typer.Option("mean", help="'mean' or 'median' leveling"),
        plane_removal: str = typer.Option(
            "none",
            "--plane-removal",
            help="Pre-pass: none | mean | median | surfalize before z_reference cut",
        ),
        json_out: bool = typer.Option(False, "--json", help="Print JSON instead of CSV"),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write CSV to this path"),
    ) -> None:
        """Bearing / geometric contact fraction vs separation (leveled height)."""
        path = Path(path)
        if path.suffix.lower() != ".tmd":
            console.print("[red]Input must be a .tmd file.[/]")
            raise typer.Exit(1)
        data = TMD.load(str(path))
        hm = np.asarray(data.height_map, dtype=np.float32)
        meta = data.metadata or {}
        curve = bearing_area_curve(
            hm,
            n=n,
            z_reference=z_reference,
            metadata=meta,
            plane_removal=plane_removal,
        )
        rows = [
            {
                "separation": float(s),
                "area_fraction": float(a),
                "dA_dd": float(d),
            }
            for s, a, d in zip(
                curve["separations"],
                curve["area_fraction"],
                curve["dA_dd"],
            )
        ]
        if json_out:
            typer.echo(
                json.dumps(
                    rc.sanitize_json_values({"file": str(path.resolve()), "curve": rows}),
                    indent=2,
                    allow_nan=False,
                )
            )
            return
        if output is not None:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", newline="", encoding="utf-8") as text_io:
                w = csv.DictWriter(text_io, fieldnames=["separation", "area_fraction", "dA_dd"])
                w.writeheader()
                for row in rows:
                    w.writerow(row)
            console.print(f"[green]Wrote[/] {out_path}")
        else:
            w = csv.DictWriter(sys.stdout, fieldnames=["separation", "area_fraction", "dA_dd"])
            w.writeheader()
            for row in rows:
                w.writerow(row)

    @app.command("plot")
    def plot_command(
        path: Path = typer.Argument(..., help="Path to a .tmd file", exists=True, readable=True),
        output: Path = typer.Option(..., "--output", "-o", help="Output PNG path"),
        plane_removal: str = typer.Option(
            "none",
            "--plane-removal",
            help="Large-scale plane: none | mean | median | surfalize",
        ),
        z_reference: str = typer.Option("mean", help="'mean' or 'median' after plane_removal"),
        n: int = typer.Option(64, "--n", help="Samples along bearing curve"),
        dpi: float = typer.Option(150.0, "--dpi", help="Figure DPI"),
        gaussian_sigma: float = typer.Option(0.0, help="Pre-smooth for slip-axis panel (pixels)"),
        no_maps: bool = typer.Option(False, "--no-maps", help="Omit shear / debris / summit panels"),
        include_anomaly_angle: bool = typer.Option(
            False,
            "--include-anomaly-angle",
            help="Fuse directionality-anomaly cue (slower; matches default axis JSON)",
        ),
    ) -> None:
        """Multi-panel PNG: leveled height, bearing curve, slip axis, tribology proxy maps."""
        path = Path(path)
        if path.suffix.lower() != ".tmd":
            console.print("[red]Input must be a .tmd file.[/]")
            raise typer.Exit(1)
        data = TMD.load(str(path))
        hm = np.asarray(data.height_map, dtype=np.float32)
        meta = data.metadata or {}
        save_tribology_dashboard_png(
            hm,
            meta,
            title=f"Tribology — {path.name}",
            output_path=output,
            plane_removal=plane_removal,
            z_reference=z_reference,
            curve_n=n,
            gaussian_sigma=gaussian_sigma,
            dpi=dpi,
            include_proxy_maps=not no_maps,
            include_anomaly_angle=include_anomaly_angle,
        )
        console.print(f"[green]Wrote[/] {Path(output).resolve()}")

    @app.command("lubrication")
    def lubrication_command(
        path: Path = typer.Argument(..., help="Path to a .tmd file", exists=True, readable=True),
        level: bool = typer.Option(True, "--level/--no-level", help="Plane leveling before ISO volumes"),
        json_out: bool = typer.Option(True, "--json/--no-json", help="Print JSON (default on)"),
    ) -> None:
        """ISO 25178 functional volume / lubrication-related parameters (Surfalize; GPL-3.0)."""
        path = Path(path)
        if path.suffix.lower() != ".tmd":
            console.print("[red]Input must be a .tmd file.[/]")
            raise typer.Exit(1)
        Surface = rc.surfalize_surface_class()
        surface = rc.load_surface_for_roughness(path, Surface)
        if level:
            surface = surface.level()
        names = list(rc.LUBRICATION_PARAMETER_NAMES)
        values = rc.roughness_dict(surface, names)
        payload = {
            "file": str(path.resolve()),
            "parameters": values,
            "note": "Values from Surfalize (GPL-3.0). See ISO 25178 functional volume definitions.",
        }
        if json_out:
            typer.echo(json.dumps(rc.sanitize_json_values(payload), indent=2, allow_nan=False))
        else:
            table = typer.rich.table.Table(title=f"Lubrication-related ISO — {path.name}")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", justify="right")
            for k in names:
                if k in values:
                    table.add_row(k, f"{values[k]!s}")
            console.print(table)

    return app
