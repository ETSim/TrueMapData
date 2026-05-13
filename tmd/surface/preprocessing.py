"""Light height-map preprocessing helpers shared by notebooks and pipelines.

These two functions are small enough to be inlined, but they show up repeatedly
in tribology / wear notebooks where the same capture has to be downsampled to a
working resolution and rescaled so that scale-sensitive demo parameters behave
the same regardless of source units.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import ndimage


__all__ = ["downsample_to_max_dim", "normalize_to_unit_sq"]


def downsample_to_max_dim(arr: np.ndarray, max_dim: int, order: int = 1) -> np.ndarray:
    """Shrink ``arr`` so its longest axis is at most ``max_dim``.

    Returns a copy at ``float64``. When ``max_dim`` is ``None`` or non-positive,
    or when the input already fits, the array is returned unchanged (still cast
    to ``float64`` to match the downsample branch).

    Parameters
    ----------
    arr:
        2D height field or generic 2D numpy array.
    max_dim:
        Maximum size of the longest axis after downsampling. ``None`` or
        non-positive values disable the operation.
    order:
        ``scipy.ndimage.zoom`` interpolation order (default ``1`` = bilinear).
    """
    arr = np.asarray(arr, dtype=np.float64)
    if max_dim is None or int(max_dim) <= 0:
        return arr
    longest = max(arr.shape)
    if longest <= int(max_dim):
        return arr
    factor = float(max_dim) / float(longest)
    return ndimage.zoom(arr, zoom=factor, order=int(order)).astype(np.float64)


def normalize_to_unit_sq(arr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Subtract the mean and divide by the RMS so the result has unit Sq.

    Returns ``(normalized, original_sq)`` where ``original_sq`` is the RMS of the
    mean-centered input (the scale factor that was divided out). When the
    centered input is numerically flat (``Sq <= 1e-12``) the function returns
    the centered array and ``1.0`` so callers can use the value as a no-op
    scale.
    """
    arr = np.asarray(arr, dtype=np.float64)
    centered = arr - float(np.mean(arr))
    sq = float(np.sqrt(np.mean(centered**2)))
    if sq <= 1e-12:
        return centered, 1.0
    return centered / sq, sq
