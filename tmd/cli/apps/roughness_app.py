"""CLI for ISO 25178 areal roughness via optional Surfalize (GPL-3.0)."""

from __future__ import annotations

import gc
import io
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import typer

from tmd.cli.core.ui import console

_INSTALL_HINT = (
    "Surfalize is not installed. Roughness commands need it (GPL-3.0). Install with:\n"
    '  pip install "truemapdata[roughness]"\n'
    "or:\n"
    "  pip install surfalize"
)

# Fallback if Surfalize lacks Surface.ISO_PARAMETERS (older installs).
_ISO_PARAMETERS_FALLBACK: Tuple[str, ...] = (
    "Sa",
    "Sq",
    "Sp",
    "Sv",
    "Sz",
    "Ssk",
    "Sku",
    "Sdr",
    "Sdq",
    "Sal",
    "Str",
    "Sk",
    "Spk",
    "Svk",
    "Smr1",
    "Smr2",
    "Sxp",
    "Vmp",
    "Vmc",
    "Vvv",
    "Vvc",
)
_QUICK_PARAMS: Tuple[str, ...] = ("Sa", "Sq", "Sz", "Ssk", "Sku")


def _iso_parameter_names(Surface: Any) -> List[str]:
    names = getattr(Surface, "ISO_PARAMETERS", None)
    if names is not None:
        return list(names)
    return list(_ISO_PARAMETERS_FALLBACK)


def _surfalize_imports() -> Any:
    try:
        from surfalize import Surface
    except ImportError as e:
        console.print(f"[red]{_INSTALL_HINT}[/]")
        raise typer.Exit(1) from e
    return Surface


def _surface_from_truemap_tmd(path: Path, Surface: Any) -> Any:
    """Build Surfalize Surface from any TMD this library can read (bypasses Surfalize file parser)."""
    from tmd import TMD

    data = TMD.load(path)
    # float32 keeps peak RAM ~half of float64 for large GelSight maps (e.g. 24M px).
    hm = np.ascontiguousarray(data.height_map, dtype=np.float32)
    meta = data.metadata or {}
    h, w = hm.shape
    width = int(meta.get("width") or w)
    height = int(meta.get("height") or h)
    xl = float(meta.get("x_length", 1.0))
    yl = float(meta.get("y_length", 1.0))
    # x_length / y_length are mm over the raster; Surfalize step_* is µm / pixel.
    step_x = (xl / max(width, 1)) * 1000.0
    step_y = (yl / max(height, 1)) * 1000.0
    slim: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            slim[str(k)] = v
    return Surface(hm, step_x, step_y, metadata=slim or None)


def _load_surface_for_roughness(path: Path, Surface: Any) -> Any:
    """Prefer Surfalize native reader; fall back to TrueMapData for headers Surfalize cannot decode."""
    from surfalize.exceptions import CorruptedFileError

    try:
        return Surface.load(str(path))
    except (UnicodeDecodeError, CorruptedFileError):
        return _surface_from_truemap_tmd(path, Surface)


def _parse_params(
    params: Optional[str],
    all_params: bool,
    quick: bool,
    Surface: Any,
) -> Optional[List[str]]:
    if all_params:
        return None
    if params is not None and params.strip() != "":
        return [p.strip() for p in params.split(",") if p.strip()]
    if quick:
        return list(_QUICK_PARAMS)
    return _iso_parameter_names(Surface)


def _roughness_dict(surface: Any, names: Optional[Sequence[str]]) -> Dict[str, Any]:
    with np.errstate(divide="ignore", invalid="ignore"):
        if names is None:
            raw = surface.roughness_parameters()
        else:
            raw = surface.roughness_parameters(list(names))
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)) or v is None:
            out[k] = v
        elif hasattr(v, "item"):
            try:
                out[k] = float(v.item())
            except (TypeError, ValueError):
                out[k] = str(v)
        else:
            out[k] = str(v)
    return out


def _sanitize_json_values(obj: Any) -> Any:
    """Replace NaN/Inf so json.dumps is strict-JSON safe."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_json_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_values(v) for v in obj]
    return obj


def _roughness_rows_for_paths(
    ordered_paths: Sequence[Path],
    *,
    level: bool,
    quick: bool,
    params: Optional[str],
    all_params: bool,
    include_frame_index: bool,
    include_full_path: bool,
) -> List[Dict[str, Any]]:
    """Run roughness on paths in list order (for sequences / time series)."""
    Surface = _surfalize_imports()
    names = _parse_params(params, all_params, quick, Surface)
    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(ordered_paths):
        row: Dict[str, Any] = {}
        if include_frame_index:
            row["frame"] = i
        surf: Any = None
        try:
            surf = _load_surface_for_roughness(p, Surface)
            if level:
                surf = surf.level()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                vals = _roughness_dict(surf, names)
            row["file"] = p.name
            if include_full_path:
                row["path"] = str(p.resolve())
            row.update(vals)
        except Exception as exc:
            row["file"] = p.name
            if include_full_path:
                row["path"] = str(p.resolve())
            row["__error__"] = str(exc)
        finally:
            surf = None
            gc.collect()
        rows.append(row)
    return rows


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

        Surface = _surfalize_imports()
        surface = _load_surface_for_roughness(path, Surface)
        if level:
            surface = surface.level()

        names = _parse_params(params, all_params, quick, Surface)
        values = _roughness_dict(surface, names)
        payload: Dict[str, Any] = {"file": str(path.resolve()), "parameters": values}

        if json_out:
            typer.echo(json.dumps(_sanitize_json_values(payload), indent=2, allow_nan=False))
        else:
            table = typer.rich.table.Table(title=f"Roughness — {path.name}")
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
            typer.echo(json.dumps(_sanitize_json_values(rows), indent=2, allow_nan=False))
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
