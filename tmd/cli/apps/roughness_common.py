"""Shared Surfalize surface loading and roughness helpers (GPL-3.0 when Surfalize is used)."""

from __future__ import annotations

import gc
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import typer

_INSTALL_HINT = (
    "Surfalize is not installed. Install with:\n"
    '  pip install "truemapdata[roughness]"\n'
    "or:\n"
    '  pip install "surfalize>=0.16.0"\n'
    "(Current Surfalize releases need Python 3.10 or newer.)"
)

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

LUBRICATION_PARAMETER_NAMES: Tuple[str, ...] = (
    "Vvv",
    "Vvc",
    "Vmp",
    "Vmc",
    "Smr1",
    "Smr2",
    "Sxp",
)


def iso_parameter_names(Surface: Any) -> List[str]:
    names = getattr(Surface, "ISO_PARAMETERS", None)
    if names is not None:
        return list(names)
    return list(_ISO_PARAMETERS_FALLBACK)


def surfalize_surface_class() -> Any:
    try:
        from surfalize import Surface
    except ImportError as e:
        from tmd.cli.core.ui import console

        console.print(f"[red]{_INSTALL_HINT}[/]")
        raise typer.Exit(1) from e
    return Surface


def surface_from_truemap_tmd(path: Path, Surface: Any) -> Any:
    """Build Surfalize Surface from any TMD this library can read (bypasses Surfalize file parser)."""
    from tmd import TMD

    data = TMD.load(path)
    hm = np.ascontiguousarray(data.height_map, dtype=np.float32)
    meta = data.metadata or {}
    h, w = hm.shape
    width = int(meta.get("width") or w)
    height = int(meta.get("height") or h)
    xl = float(meta.get("x_length", 1.0))
    yl = float(meta.get("y_length", 1.0))
    step_x = (xl / max(width, 1)) * 1000.0
    step_y = (yl / max(height, 1)) * 1000.0
    slim: Dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            slim[str(k)] = v
    return Surface(hm, step_x, step_y, metadata=slim or None)


def load_surface_for_roughness(path: Path, Surface: Any) -> Any:
    """Prefer Surfalize native reader; fall back to TrueMapData for headers Surfalize cannot decode."""
    from surfalize.exceptions import CorruptedFileError

    try:
        return Surface.load(str(path))
    except (UnicodeDecodeError, CorruptedFileError):
        return surface_from_truemap_tmd(path, Surface)


def parse_roughness_params(
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
    return iso_parameter_names(Surface)


def roughness_dict(surface: Any, names: Optional[Sequence[str]]) -> Dict[str, Any]:
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


def sanitize_json_values(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_json_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json_values(v) for v in obj]
    return obj


def roughness_rows_for_paths(
    ordered_paths: Sequence[Path],
    *,
    level: bool,
    quick: bool,
    params: Optional[str],
    all_params: bool,
    include_frame_index: bool,
    include_full_path: bool,
    load_surface: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Run roughness on paths in list order (for sequences / time series)."""
    Surface = surfalize_surface_class()
    names = parse_roughness_params(params, all_params, quick, Surface)
    loader = load_surface if load_surface is not None else load_surface_for_roughness
    rows: List[Dict[str, Any]] = []
    for i, p in enumerate(ordered_paths):
        row: Dict[str, Any] = {}
        if include_frame_index:
            row["frame"] = i
        surf: Any = None
        try:
            surf = loader(p, Surface)
            if level:
                surf = surf.level()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                vals = roughness_dict(surf, names)
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
