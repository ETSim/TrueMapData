"""Typed contracts for surface analysis outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, TypedDict

import numpy as np

DefectOutputMode = Literal["summary", "standard", "full"]


@dataclass(frozen=True)
class DefectDetectionConfig:
    """Configuration for defect detection on 2D height maps."""

    gaussian_sigma: float = 1.0
    zscore_threshold: float = 1.8
    min_area: int = 6
    min_confidence: float = 0.0
    directionality_window: int = 11
    directionality_angle_threshold_deg: float = 22.5
    output_mode: DefectOutputMode = "summary"
    include_responses: bool = False


class DefectClassResultBase(TypedDict):
    """Per-defect-class outputs."""

    count: int
    confidence: float


class DefectClassResult(DefectClassResultBase, total=False):
    mask: np.ndarray
    response: np.ndarray
    areas: List[int]


class DefectSummary(TypedDict):
    """Top-level aggregate statistics."""

    total_count: int
    global_confidence: float
    class_counts: Dict[str, int]


class DefectAnalysisResultBase(TypedDict):
    """Canonical defect analysis response payload."""

    defects: Dict[str, DefectClassResult]
    summary: DefectSummary


class DefectAnalysisResult(DefectAnalysisResultBase, total=False):
    mask: np.ndarray
    labels: np.ndarray
    overlay_rgb: np.ndarray
