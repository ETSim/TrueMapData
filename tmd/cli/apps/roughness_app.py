"""CLI for ISO 25178 areal roughness via optional Surfalize (GPL-3.0)."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import typer
from rich.table import Table

from tmd.cli.apps import roughness_common as rc
from tmd.cli.core.ui import console

# Backward-compatible aliases (see tests/cli/apps/test_roughness_app_extra.py)
_surface_from_truemap_tmd = rc.surface_from_truemap_tmd
_load_surface_for_roughness = rc.load_surface_for_roughness
_parse_params = rc.parse_roughness_params
_roughness_dict = rc.roughness_dict
_sanitize_json_values = rc.sanitize_json_values
_ISO_PARAMETERS_FALLBACK = rc._ISO_PARAMETERS_FALLBACK
_QUICK_PARAMS = rc._QUICK_PARAMS
_iso_parameter_names = rc.iso_parameter_names
_surfalize_imports = rc.surfalize_surface_class


def _roughness_rows_for_paths(
    ordered_paths,
    *,
    level: bool,
    quick: bool,
    params: Optional[str],
    all_params: bool,
    include_frame_index: bool,
    include_full_path: bool,
):
    return rc.roughness_rows_for_paths(
        ordered_paths,
        level=level,
        quick=quick,
        params=params,
        all_params=all_params,
        include_frame_index=include_frame_index,
        include_full_path=include_full_path,
        load_surface=_load_surface_for_roughness,
    )


def create_roughness_app() -> typer.Typer:
    app = typer.Typer(help="Areal roughness (ISO 25178) via Surfalize — optional dependency")

    @app.command("file")
    def file_command(
        path: Path = typer.Argument(..., help="Path to a .tmd file", exists=True, readable=True),
        level: bool = typer.Option(True, "--level/--no-level", help="Plane leveling before parameters"),
        json_out: bool = typer.Option(False, "--json", help="Print JSON instead of a table"),
        quick: bool = typer.Option(
            False,
            "--quick",
            help="Only Sa,Sq,Sz,Ssk,Sku (fast); default is full ISO 25178 set from Surfalize",
        ),
        params: Optional[str] = typer.Option(
            None,
            "--params",
            "-p",
            help="Comma-separated names; overrides default / --quick",
        ),
        all_params: bool = typer.Option(
            False,
            "--all",
            help="Every parameter Surfalize supports (ISO plus extras such as periodic texture metrics)",
        ),
    ) -> None:
        """Compute roughness on one TMD file."""
        path = Path(path)
        if path.suffix.lower() != ".tmd":
            console.print("[red]Input must be a .tmd file.[/]")
            raise typer.Exit(1)

        Surface = rc.surfalize_surface_class()
        surface = rc.load_surface_for_roughness(path, Surface)
        if level:
            surface = surface.level()

        names = rc.parse_roughness_params(params, all_params, quick, Surface)
        values = rc.roughness_dict(surface, names)
        payload = {"file": str(path.resolve()), "parameters": values}

        if json_out:
            typer.echo(json.dumps(rc.sanitize_json_values(payload), indent=2, allow_nan=False))
        else:
            table = Table(title=f"Roughness — {path.name}")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", justify="right")
            for k in sorted(values.keys(), key=str.lower):
                table.add_row(k, f"{values[k]!s}")
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
            help="CSV path (default: print CSV to stdout)",
        ),
        level: bool = typer.Option(True, "--level/--no-level", help="Plane leveling before parameters"),
        quick: bool = typer.Option(
            False,
            "--quick",
            help="Only Sa,Sq,Sz,Ssk,Sku per file; default is full ISO 25178 set from Surfalize",
        ),
        params: Optional[str] = typer.Option(
            None,
            "--params",
            "-p",
            help="Comma-separated names; overrides default / --quick",
        ),
        all_params: bool = typer.Option(
            False,
            "--all",
            help="Every parameter Surfalize supports (ISO plus extras)",
        ),
        parallel: bool = typer.Option(
            False,
            "--parallel",
            help="Use Surfalize multiprocessing",
        ),
    ) -> None:
        """Roughness for every matching .tmd under a directory."""
        directory = Path(directory)
        it: Iterable[Path] = directory.rglob(pattern) if recursive else directory.glob(pattern)
        paths = sorted(p for p in it if p.is_file() and p.suffix.lower() == ".tmd")

        if not paths:
            console.print(f"[yellow]No .tmd matched {pattern!r} under {directory}[/]")
            raise typer.Exit(0)

        try:
            import pandas as pd
        except ImportError as e:
            console.print("[red]batch requires pandas (install surfalize or pandas).[/]")
            raise typer.Exit(1) from e

        rows = _roughness_rows_for_paths(
            paths,
            level=level,
            quick=quick,
            params=params,
            all_params=all_params,
            include_frame_index=False,
            include_full_path=False,
        )
        df = pd.DataFrame(rows)
        if parallel:
            console.print(
                "[yellow]Note:[/] `--parallel` is ignored; batch uses a sequential "
                "path compatible with all TMD files this library can read."
            )

        if output is not None:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            console.print(f"[green]Wrote[/] {out_path} ({len(df)} rows)")
        else:
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            sys.stdout.write(buf.getvalue())

    @app.command("sequence")
    def sequence_command(
        paths: Optional[List[Path]] = typer.Argument(
            None,
            help="TMD files in frame order (time / scan order after alignment); frame 0 = first path",
        ),
        from_dir: Optional[Path] = typer.Option(
            None,
            "--from-dir",
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="Instead of paths: all matching .tmd in this folder (see --sort-by)",
        ),
        pattern: str = typer.Option("*.tmd", "--pattern", help="Glob with --from-dir (default *.tmd)"),
        sort_by: str = typer.Option(
            "name",
            "--sort-by",
            help="With --from-dir: sort files by 'name' or 'mtime'",
        ),
        output: Optional[Path] = typer.Option(
            None,
            "--output",
            "-o",
            help="CSV path (default: print CSV to stdout)",
        ),
        json_out: bool = typer.Option(False, "--json", help="Print JSON array instead of CSV"),
        level: bool = typer.Option(True, "--level/--no-level", help="Plane leveling before parameters"),
        quick: bool = typer.Option(
            False,
            "--quick",
            help="Only Sa,Sq,Sz,Ssk,Sku per frame; default is full ISO 25178 set",
        ),
        params: Optional[str] = typer.Option(
            None,
            "--params",
            "-p",
            help="Comma-separated names; overrides default / --quick",
        ),
        all_params: bool = typer.Option(
            False,
            "--all",
            help="Every parameter Surfalize supports (ISO plus extras)",
        ),
        no_path: bool = typer.Option(
            False,
            "--no-path",
            help="Omit full path column (CSV/JSON still include file name)",
        ),
    ) -> None:
        """Roughness vs frame index for an ordered sequence (e.g. aligned TMDs over time)."""
        path_list = list(paths) if paths else []
        ordered: List[Path] = []
        if from_dir is not None:
            if path_list:
                console.print("[red]Use either explicit PATHS or --from-dir, not both.[/]")
                raise typer.Exit(1)
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
            ordered = [Path(p) for p in path_list]

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

        rows = _roughness_rows_for_paths(
            ordered,
            level=level,
            quick=quick,
            params=params,
            all_params=all_params,
            include_frame_index=True,
            include_full_path=not no_path,
        )

        if json_out:
            typer.echo(json.dumps(rc.sanitize_json_values(rows), indent=2, allow_nan=False))
            return

        try:
            import pandas as pd
        except ImportError as e:
            console.print("[red]sequence CSV requires pandas (install surfalize or pandas).[/]")
            raise typer.Exit(1) from e

        df = pd.DataFrame(rows)
        if output is not None:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_path, index=False)
            console.print(f"[green]Wrote[/] {out_path} ({len(df)} frames)")
        else:
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            sys.stdout.write(buf.getvalue())

    return app
