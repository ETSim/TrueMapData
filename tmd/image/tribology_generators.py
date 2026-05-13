"""Thin ``MapGenerator`` wrappers for tribology proxy maps (implementation in ``tmd.surface.metrics``)."""

from __future__ import annotations

import numpy as np

from tmd.surface.metrics.tribology_maps import debris_pocket_map_01, shear_hazard_map_01

from .maps.base_generator import MapGenerator


class ShearHazardMapGenerator(MapGenerator):
    """Interfacial shear hazard proxy (slope × |mean curvature| × local Sq)."""

    def __init__(
        self,
        roughness_sigma: float = 10.0,
        sq_window: int = 7,
        gaussian_sigma: float = 0.0,
        plane_removal: str = "none",
        **kwargs,
    ):
        super().__init__(
            roughness_sigma=roughness_sigma,
            sq_window=sq_window,
            gaussian_sigma=gaussian_sigma,
            plane_removal=plane_removal,
            **kwargs,
        )

    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        params = self._get_params(**kwargs)
        metadata = kwargs.get("metadata") or {}
        return shear_hazard_map_01(
            height_map,
            metadata,
            roughness_sigma=float(params["roughness_sigma"]),
            sq_window=int(params["sq_window"]),
            gaussian_sigma=float(params["gaussian_sigma"]),
            plane_removal=str(params.get("plane_removal", "none")),
        )


class DebrisPocketMapGenerator(MapGenerator):
    """Third-body pocket heuristic score (normalized)."""

    def __init__(
        self,
        valley_percentile: float = 10.0,
        slope_percentile: float = 40.0,
        pit_dilate: int = 3,
        plane_removal: str = "none",
        **kwargs,
    ):
        super().__init__(
            valley_percentile=valley_percentile,
            slope_percentile=slope_percentile,
            pit_dilate=pit_dilate,
            plane_removal=plane_removal,
            **kwargs,
        )

    def generate(self, height_map: np.ndarray, **kwargs) -> np.ndarray:
        params = self._get_params(**kwargs)
        metadata = kwargs.get("metadata") or {}
        return debris_pocket_map_01(
            height_map,
            metadata,
            valley_percentile=float(params["valley_percentile"]),
            slope_percentile=float(params["slope_percentile"]),
            pit_dilate=int(params["pit_dilate"]),
            plane_removal=str(params.get("plane_removal", "none")),
        )
