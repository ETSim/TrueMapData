#!/usr/bin/env python3
"""
MAT (MATLAB) Exporter/Importer for TMD Data.

This module provides concrete implementations for exporting and importing
TMD data in the MATLAB ``.mat`` format. The on-disk schema mirrors the
multi-surface convention used by ``surface_from_simulation.mat``:

* ``Surface_simulation_i`` (``i = 1..N``) -- 2D height-map arrays.
* ``SurfaceParameters_i`` -- a struct with the canonical ISO 25178 areal
  roughness fields (``Sdr, Sdq, Spc, Zp, Spd, Sq, Sku, Ssk``); unknown keys
  are passed through verbatim.
* ``metadata`` -- optional struct with arbitrary top-level info such as
  ``mmpp``, ``x_length``, ``y_length``.
* ``tmd_format`` / ``tmd_version`` -- format identifiers for forward
  compatibility.

The exporter accepts either a single-surface dict (``{"height_map": arr}``)
or a multi-surface dict (``{"surfaces": {1: arr1, ...}}``) and always writes
the multi-surface shape on disk. The importer returns a uniform dictionary
and, when ``N == 1``, also exposes ``height_map`` for convenience.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np

from .base import TMDDataExporter, TMDDataImporter

logger = logging.getLogger(__name__)

ISO_PARAMETER_KEYS: Tuple[str, ...] = (
    "Sdr",
    "Sdq",
    "Spc",
    "Zp",
    "Spd",
    "Sq",
    "Sku",
    "Ssk",
)

TMD_FORMAT_TAG = "tmd_surface_v1"
TMD_FORMAT_VERSION = "1.0"

_SURFACE_KEY_RE = re.compile(r"^(Surface_simulation|SurfaceParameters)_(\d+)$")


def _require_scipy_io():
    """Lazy-import scipy.io and raise a friendly error if unavailable."""
    try:
        import scipy.io as sio  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "scipy is required for .mat IO; install scipy to use MATExporter/MATImporter"
        ) from exc
    return sio


def _coerce_height_map(arr: Any, *, where: str) -> np.ndarray:
    """Coerce ``arr`` to a 2D float64 ndarray or raise ``ValueError``."""
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.asarray(arr)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError(f"{where} could not be converted to ndarray: {exc}") from exc
    if arr.ndim != 2:
        raise ValueError(f"{where} must be 2D, got {arr.ndim}D shape={arr.shape}")
    return np.asarray(arr, dtype=np.float64)


def _collect_surfaces(data: Mapping[str, Any]) -> Dict[int, np.ndarray]:
    """Extract a ``{index: 2D ndarray}`` mapping from input data.

    Accepts:
    - ``{"height_map": arr}`` -> ``{1: arr}``.
    - ``{"surfaces": {i: arr, ...}}`` (any int-like key) -> indexed mapping.
    - ``{"surfaces": [arr, ...]}`` -> 1-based indexed mapping.
    """
    if "surfaces" in data and data["surfaces"] is not None:
        surfaces = data["surfaces"]
        result: Dict[int, np.ndarray] = {}
        if isinstance(surfaces, Mapping):
            for raw_key, arr in surfaces.items():
                try:
                    idx = int(raw_key)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"surfaces key {raw_key!r} is not coercible to int"
                    ) from exc
                result[idx] = _coerce_height_map(arr, where=f"surfaces[{idx}]")
        elif isinstance(surfaces, Iterable):
            for i, arr in enumerate(surfaces, start=1):
                result[i] = _coerce_height_map(arr, where=f"surfaces[{i}]")
        else:
            raise ValueError(
                "surfaces must be a Mapping[int, ndarray] or Iterable[ndarray]"
            )
        if not result:
            raise ValueError("surfaces is empty; at least one heightmap is required")
        return result

    if "height_map" in data and data["height_map"] is not None:
        return {1: _coerce_height_map(data["height_map"], where="height_map")}

    raise ValueError(
        "MATExporter input must provide either 'height_map' or 'surfaces'"
    )


def _collect_parameters(
    data: Mapping[str, Any], indices: Iterable[int]
) -> Dict[int, Dict[str, Any]]:
    """Return ``{index: parameter_dict}`` for each surface index.

    Missing entries map to an empty dict (will be written as an empty struct).
    """
    raw = data.get("parameters") or {}
    result: Dict[int, Dict[str, Any]] = {}
    if isinstance(raw, Mapping):
        # Mapping keyed by int (or int-coercible).
        normalised: Dict[int, Mapping[str, Any]] = {}
        for raw_key, value in raw.items():
            try:
                normalised[int(raw_key)] = value or {}
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"parameters key {raw_key!r} is not coercible to int"
                ) from exc
        for idx in indices:
            result[idx] = dict(normalised.get(idx, {}))
    elif isinstance(raw, Iterable):
        sequence = list(raw)
        for offset, idx in enumerate(indices):
            result[idx] = dict(sequence[offset]) if offset < len(sequence) else {}
    else:
        raise ValueError("parameters must be a Mapping or Iterable of dicts")
    return result


def _struct_from_dict(d: Mapping[str, Any]) -> np.ndarray:
    """Build a 1x1 numpy structured array from ``d`` for ``savemat``.

    The dict must be non-empty; scipy cannot serialise a structured array
    that has no fields, so callers are expected to skip empty dicts.
    """
    if not d:
        raise ValueError("_struct_from_dict requires a non-empty mapping")
    dtype = [(str(k), object) for k in d.keys()]
    arr = np.zeros((1, 1), dtype=dtype)
    for k, v in d.items():
        arr[0, 0][str(k)] = v
    return arr


def _normalise_loaded_value(value: Any) -> Any:
    """Best-effort normalisation of a value loaded via ``simplify_cells=True``.

    - ndarray of shape (1,) -> python scalar
    - ndarray of shape () (0-d) -> python scalar
    - dict of struct fields -> recursed
    """
    if isinstance(value, np.ndarray):
        if value.dtype == object and value.size == 1:
            return _normalise_loaded_value(value.item())
        if value.ndim == 0:
            try:
                return value.item()
            except Exception:  # pragma: no cover - defensive
                return value
        if value.shape == (1,):
            try:
                return value.item()
            except Exception:  # pragma: no cover - defensive
                return value
        return value
    if isinstance(value, dict):
        return {str(k): _normalise_loaded_value(v) for k, v in value.items()}
    return value


def _dict_from_struct(struct: Any) -> Dict[str, Any]:
    """Convert a value loaded via ``simplify_cells=True`` to a plain dict."""
    if struct is None:
        return {}
    if isinstance(struct, np.ndarray):
        if struct.size == 0:
            return {}
        if struct.dtype.names:
            entry = struct.flat[0] if struct.size > 1 else struct[()]
            return {name: _normalise_loaded_value(entry[name]) for name in struct.dtype.names}
        if struct.size == 1:
            return _dict_from_struct(struct.item())
    if isinstance(struct, Mapping):
        return {str(k): _normalise_loaded_value(v) for k, v in struct.items()}
    return {}


def _pack_savemat_dict(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the keyword dict passed to ``scipy.io.savemat``."""
    surfaces = _collect_surfaces(data)
    indices = sorted(surfaces.keys())
    parameters = _collect_parameters(data, indices)
    metadata = data.get("metadata") or {}

    out: Dict[str, Any] = {}
    for new_idx, original_idx in enumerate(indices, start=1):
        out[f"Surface_simulation_{new_idx}"] = surfaces[original_idx]
        # Only emit SurfaceParameters_i when the dict is non-empty; scipy
        # cannot serialise a struct array with zero fields.
        params = parameters[original_idx]
        if params:
            out[f"SurfaceParameters_{new_idx}"] = _struct_from_dict(params)

    if metadata:
        out["metadata"] = _struct_from_dict(dict(metadata))
    out["tmd_format"] = TMD_FORMAT_TAG
    out["tmd_version"] = TMD_FORMAT_VERSION
    return out


def _unpack_loadmat_dict(matdict: Mapping[str, Any]) -> Dict[str, Any]:
    """Parse a ``scipy.io.loadmat(simplify_cells=True)`` dict into the uniform schema."""
    surface_indices: Dict[int, np.ndarray] = {}
    parameter_indices: Dict[int, Dict[str, Any]] = {}
    extras: Dict[str, Any] = {}

    for key, value in matdict.items():
        if key.startswith("__"):
            continue
        match = _SURFACE_KEY_RE.match(key)
        if match:
            kind, idx_text = match.group(1), match.group(2)
            idx = int(idx_text)
            if kind == "Surface_simulation":
                surface_indices[idx] = _coerce_height_map(value, where=key)
            else:
                parameter_indices[idx] = _dict_from_struct(value)
            continue
        if key == "metadata":
            extras["metadata"] = _dict_from_struct(value)
        elif key in {"tmd_format", "tmd_version"}:
            extras[key] = _normalise_loaded_value(value)
        else:
            extras[key] = _normalise_loaded_value(value)

    if not surface_indices:
        raise ValueError(
            "MAT file does not contain any 'Surface_simulation_i' entries"
        )

    sorted_indices = sorted(surface_indices.keys())
    surfaces = {i: surface_indices[i] for i in sorted_indices}
    parameters = {i: parameter_indices.get(i, {}) for i in sorted_indices}

    result: Dict[str, Any] = {
        "surfaces": surfaces,
        "parameters": parameters,
        "metadata": extras.get("metadata", {}),
        "tmd_format": extras.get("tmd_format", TMD_FORMAT_TAG),
        "tmd_version": extras.get("tmd_version", TMD_FORMAT_VERSION),
    }

    if len(surfaces) == 1:
        only_idx = sorted_indices[0]
        result["height_map"] = surfaces[only_idx]

    return result


def _export_mat(data: Mapping[str, Any], output_path: str) -> str:
    sio = _require_scipy_io()
    output_path = os.path.abspath(output_path)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = _pack_savemat_dict(data)
    try:
        sio.savemat(output_path, payload, do_compression=True, format="5")
    except Exception as exc:
        logger.error("Error exporting MAT: %s", exc)
        raise
    logger.info("Data exported to MAT file: %s", output_path)
    return output_path


def _load_mat(file_path: str) -> Dict[str, Any]:
    sio = _require_scipy_io()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"MAT file not found: {path}")
    try:
        matdict = sio.loadmat(str(path), simplify_cells=True)
    except Exception as exc:
        logger.error("Error loading MAT file %s: %s", path, exc)
        raise
    logger.info("Data loaded from MAT file: %s", path)
    return _unpack_loadmat_dict(matdict)


class MATExporter(TMDDataExporter):
    """Export TMD data to MATLAB ``.mat`` format."""

    def export(self, data: Dict[str, Any], output_path: str) -> str:
        return _export_mat(data, output_path)


class MATImporter(TMDDataImporter):
    """Load TMD data from MATLAB ``.mat`` files."""

    def load(self, file_path: str) -> Dict[str, Any]:
        return _load_mat(file_path)


__all__ = [
    "ISO_PARAMETER_KEYS",
    "TMD_FORMAT_TAG",
    "TMD_FORMAT_VERSION",
    "MATExporter",
    "MATImporter",
]
