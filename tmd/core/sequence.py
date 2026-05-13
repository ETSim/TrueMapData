"""
TMDSequence: Core class for managing sequences of height maps.

This module defines the TMDSequence class that supports adding frames,
applying transformations, computing statistics, and exporting the sequence
to various formats using a centralized factory-based approach.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from tmd.core.tmd import TMD, TMDProcessingError
from tmd.utils.files import TMDFileUtilities
from tmd.surface.processing import threshold_height_map
from tmd.surface.types import DefectDetectionConfig
from tmd.sequence.alignment import (
    NormalMapSequenceAlignmentConfig,
    align_height_map_sequence_phase_fft,
    align_height_map_sequence_sift,
    align_normal_map_sequence,
    height_maps_to_normal_bgr_uint8,
    warp_scalar_sequence_with_affine_crop,
)
from tmd.sequence.factory import SequenceExporterFactory
from tmd.surface.transformations import align_height_map_sequence_opencv

logger = logging.getLogger(__name__)


def _reference_first_permutation(n: int, reference_index: int) -> Tuple[List[int], List[int]]:
    """``perm[j]`` = original frame index at algorithm position ``j``; ``inv[i]`` = position of original ``i``."""
    if not (0 <= reference_index < n):
        raise ValueError(f"reference_index must be in [0, {n - 1}], got {reference_index}")
    perm = [reference_index] + [i for i in range(n) if i != reference_index]
    inv = [0] * n
    for j, orig in enumerate(perm):
        inv[orig] = j
    return perm, inv


class TMDSequence:
    """
    Class representing a sequence of TMD files.

    Provides methods for adding frames (from TMD files or arrays),
    managing timestamps and transformations, computing statistics, and exporting
    the sequence to different file formats using a factory-based approach.
    """

    def __init__(self, name: str = "Unnamed Sequence"):
        """
        Initialize a new TMDSequence.
        
        Args:
            name: Name of the sequence for identification
        """
        self.name = name
        self.frames: List[np.ndarray] = []
        self.frame_timestamps: List[Any] = []
        self.metadata: Dict[str, Any] = {}
        self.transformations: Dict[int, Dict[str, Any]] = {}
        self.frame_metadata: List[Dict[str, Any]] = []
        self.tmd_objects: List[Optional[TMD]] = []  # Store TMD objects if available

    def add_frame(
        self,
        height_map: np.ndarray,
        timestamp: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        transformation: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add a single frame to the sequence.
        
        Args:
            height_map: 2D numpy array with height map data
            timestamp: Timestamp identifier for the frame
            metadata: Associated metadata dictionary
            transformation: Dictionary of transformations to apply
            
        Returns:
            Index of the added frame, or -1 if failed
        """
        if height_map is None or height_map.size == 0:
            logger.warning("Attempted to add empty height map to sequence")
            return -1

        frame_data = height_map.copy()
        if timestamp is None:
            timestamp = f"Frame {len(self.frames) + 1}"
        if metadata is None:
            metadata = {}

        self.frames.append(frame_data)
        self.frame_timestamps.append(timestamp)
        self.frame_metadata.append(metadata)
        self.transformations[len(self.frames) - 1] = transformation if transformation else {}
        self.tmd_objects.append(None)  # No TMD object for raw frame
        return len(self.frames) - 1

    def add_tmd_file(self, filepath: Union[str, Path], timestamp: Any = None) -> int:
        """
        Add a single TMD file to the sequence.
        
        Args:
            filepath: Path to the TMD file
            timestamp: Optional timestamp (defaults to filename)
            
        Returns:
            Index of the added frame, or -1 if failed
        """
        try:
            # Use the TMD class for better integration
            tmd_obj = TMD(filepath)
            
            if timestamp is None:
                timestamp = Path(filepath).stem
                
            # Get the height map and metadata from the TMD object
            height_map = tmd_obj.height_map
            metadata = tmd_obj.metadata
            
            # Add the frame
            frame_idx = self.add_frame(height_map, timestamp, metadata)
            
            # Store the TMD object reference
            if frame_idx >= 0:
                self.tmd_objects[frame_idx] = tmd_obj
                logger.info(f"Added TMD file '{filepath}' as frame {frame_idx}")
            
            return frame_idx
            
        except (FileNotFoundError, TMDProcessingError) as e:
            logger.error(f"Error adding TMD file '{filepath}': {str(e)}")
            return -1
        except Exception as e:
            logger.error(f"Unexpected error adding TMD file '{filepath}': {str(e)}")
            return -1

    def add_frames_from_folder(
        self, 
        folder_path: Union[str, Path], 
        extension: str = "tmd",
        sort_method: str = "name",
        recursive: bool = True
    ) -> int:
        """
        Add all TMD files from a folder to the sequence.
        
        Args:
            folder_path: Path to the folder containing TMD files
            extension: File extension to match (default: "tmd")
            sort_method: How to sort files ("name", "time", "none")
            recursive: Whether to search subdirectories
            
        Returns:
            Number of successfully added frames
        """
        try:
            # Get list of all files with matching extension
            file_list = TMDFileUtilities.list_files_with_extension(
                str(folder_path), extension, recursive=recursive
            )
            
            if not file_list:
                logger.warning(f"No files with extension '.{extension}' found in {folder_path}")
                return 0
                
            # Sort files if requested
            if sort_method.lower() == "name":
                file_list.sort()
            elif sort_method.lower() == "time":
                file_list.sort(key=lambda x: Path(x).stat().st_mtime)
                
            # Add each file to the sequence
            count = 0
            for filepath in file_list:
                result = self.add_tmd_file(filepath)
                if result >= 0:
                    count += 1
                    
            logger.info(f"Added {count} frames from folder {folder_path}")
            return count
            
        except Exception as e:
            logger.error(f"Error adding frames from folder '{folder_path}': {str(e)}")
            return 0

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        """Get a specific frame by index."""
        if 0 <= index < len(self.frames):
            return self.frames[index]
        logger.warning(f"Invalid frame index: {index}")
        return None

    def get_frame_count(self) -> int:
        """Get the total number of frames in the sequence."""
        return len(self.frames)

    def get_timestamp(self, index: int) -> Optional[Any]:
        """Get the timestamp for a specific frame."""
        if 0 <= index < len(self.frame_timestamps):
            return self.frame_timestamps[index]
        logger.warning(f"Invalid frame index: {index}")
        return None

    def get_all_timestamps(self) -> List[Any]:
        """Get all frame timestamps."""
        return self.frame_timestamps.copy()

    def get_all_frames(self) -> List[np.ndarray]:
        """Get all frames in the sequence."""
        return self.frames.copy()

    def get_frame_metadata(self, index: int) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific frame."""
        if 0 <= index < len(self.frame_metadata):
            return self.frame_metadata[index]
        logger.warning(f"Invalid frame index: {index}")
        return None

    def get_tmd_object(self, index: int) -> Optional[TMD]:
        """Get the original TMD object for a frame if available."""
        if 0 <= index < len(self.tmd_objects):
            return self.tmd_objects[index]
        return None

    def get_transformation(self, index: int) -> Optional[Dict[str, Any]]:
        """Get transformation parameters for a specific frame."""
        return self.transformations.get(index, {})

    def set_transformation(self, index: int, transformation: Dict[str, Any]) -> bool:
        """Set transformation parameters for a specific frame."""
        if 0 <= index < len(self.frames):
            self.transformations[index] = transformation
            return True
        logger.warning(f"Invalid frame index: {index}")
        return False

    def align_height_maps_opencv(
        self,
        reference_index: int = 0,
        method: str = "auto",
        crop: bool = True,
        margin: int = 0,
        phase_refine: bool = False,
        second_full_pass: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Align all frames in this sequence to a reference height map using OpenCV (2D only).

        Replaces ``self.frames`` with aligned (and optionally cropped) arrays. Alignment
        metadata is stored under ``self.metadata["alignment"]`` and is separate from
        per-frame ``transformations`` used by :meth:`apply_transformations`.

        Args:
            reference_index: Index of the frame to use as reference.
            method: ``"auto"``, ``"affine_ransac"``, or ``"phase_correlation"``.
            crop: If True, crop to the intersection of valid overlap after warping.
            margin: Inward shrink of the crop box in pixels (after overlap is computed).
            phase_refine: If True, apply a second phase-correlation pass (sub-pixel) after
                the primary method for each non-reference frame.
            second_full_pass: If True, run a full second registration pass on the aligned
                stack (see :func:`tmd.surface.transformations.align_height_map_sequence_opencv`).
            **kwargs: Optional tuning (e.g. ``min_inliers``, ``ransac_reproj_threshold``,
                ``orb_nfeatures``, ``ratio_test``, ``upsample_factor`` for phase path).

        Returns:
            The alignment info dict from :func:`tmd.surface.transformations.align_height_map_sequence_opencv`.

        Raises:
            ImportError: If OpenCV is not installed.
            ValueError: If there are no frames or ``reference_index`` is invalid.
        """
        if not self.frames:
            raise ValueError("Cannot align an empty sequence")
        aligned, info = align_height_map_sequence_opencv(
            self.frames,
            reference_index=reference_index,
            method=method,
            crop=crop,
            margin=margin,
            phase_refine=phase_refine,
            second_full_pass=second_full_pass,
            **kwargs,
        )
        self.frames = aligned
        self.metadata["alignment"] = info
        return info

    def align_height_maps_phase_fft(self, reference_index: int = 0) -> Dict[str, Any]:
        """
        Align frames to a reference using NumPy phase correlation and ``np.roll``.

        Same shape for every frame is required. For affine drift or mixed resolutions,
        prefer :meth:`align_height_maps_opencv` first.

        Stores result under ``self.metadata["alignment"]`` (overwrites any prior entry).
        """
        if not self.frames:
            raise ValueError("Cannot align an empty sequence")
        aligned, info = align_height_map_sequence_phase_fft(
            [np.asarray(f, dtype=np.float64) for f in self.frames],
            reference_index=reference_index,
        )
        self.frames = aligned
        self.metadata["alignment"] = info
        return info

    def align_height_maps_sift(
        self,
        reference_index: int = 0,
        *,
        two_pass: bool = True,
        crop: bool = True,
        config: Optional[NormalMapSequenceAlignmentConfig] = None,
    ) -> Dict[str, Any]:
        """
        Align height frames using the TextureFriction ``align.ipynb`` **height** path
        (SIFT / ECC cumulative affines, scalar ``warpAffine``, optional two-pass, valid crop).

        The underlying registration always uses **algorithm frame 0** as the fixed
        target. ``reference_index`` selects which *original* frame becomes that anchor
        by reordering internally, then restoring original frame order afterward.

        Requires OpenCV. Metadata is stored under ``self.metadata["alignment"]``.
        """
        if not self.frames:
            raise ValueError("Cannot align an empty sequence")
        n = len(self.frames)
        perm, inv = _reference_first_permutation(n, reference_index)
        permuted = [np.asarray(self.frames[i]) for i in perm]
        aligned_p, info = align_height_map_sequence_sift(
            permuted, two_pass=two_pass, crop=crop, config=config
        )
        self.frames = [aligned_p[inv[i]] for i in range(n)]
        info["reference_index"] = int(reference_index)
        info["permutation_to_algorithm_order"] = [int(x) for x in perm]
        self.metadata["alignment"] = info
        return info

    def align_height_maps_from_normals(
        self,
        reference_index: int = 0,
        *,
        two_pass: bool = True,
        crop: bool = True,
        config: Optional[NormalMapSequenceAlignmentConfig] = None,
        normal_strength: float = 1.0,
        normal_normalize: bool = False,
    ) -> Dict[str, Any]:
        """
        TextureFriction ``align.ipynb`` **primary** path: derive BGR normals from heights,
        run two-pass SIFT alignment on normals (tangent rotation on warp), then apply the
        same affines + crop to the height maps.

        ``reference_index`` is handled like :meth:`align_height_maps_sift` (reorder so that
        frame becomes algorithm index ``0``, then unpermute).

        Requires OpenCV. Metadata is stored under ``self.metadata["alignment"]``.
        """
        if not self.frames:
            raise ValueError("Cannot align an empty sequence")
        n = len(self.frames)
        perm, inv = _reference_first_permutation(n, reference_index)
        permuted = [np.asarray(self.frames[i]) for i in perm]
        dtypes = [np.asarray(self.frames[i]).dtype for i in perm]
        bgrs = height_maps_to_normal_bgr_uint8(
            permuted, strength=float(normal_strength), normalize=bool(normal_normalize)
        )
        _, ninfo = align_normal_map_sequence(bgrs, two_pass=two_pass, crop=crop, config=config)
        transforms = [np.asarray(p["affine_2x3"], dtype=np.float32) for p in ninfo["per_frame"]]
        fs = ninfo["full_size"]
        full_wh = (int(fs["width"]), int(fs["height"]))
        cr = ninfo.get("crop_region")
        if cr is None:
            crop_xywh = (0, 0, full_wh[0], full_wh[1])
        else:
            crop_xywh = (int(cr["x"]), int(cr["y"]), int(cr["width"]), int(cr["height"]))
        h32 = [np.asarray(f, dtype=np.float32) for f in permuted]
        warped_perm = warp_scalar_sequence_with_affine_crop(
            h32, transforms, full_wh, crop_xywh, border_value=0.0
        )
        out_perm: List[np.ndarray] = []
        for w, dt in zip(warped_perm, dtypes):
            if np.issubdtype(dt, np.integer):
                out_perm.append(
                    np.clip(np.round(w), np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)
                )
            else:
                out_perm.append(w.astype(dt, copy=False))
        self.frames = [out_perm[inv[i]] for i in range(n)]
        info: Dict[str, Any] = {**ninfo, "registration_source": "normals"}
        info["reference_index"] = int(reference_index)
        info["permutation_to_algorithm_order"] = [int(x) for x in perm]
        self.metadata["alignment"] = info
        return info

    def sequential_wear_metrics(
        self,
        *,
        dx_mm: Optional[float] = None,
        dy_mm: Optional[float] = None,
        reference_index: int = 0,
        signed: bool = False,
        top_fraction: float = 0.10,
        align_before: Optional[Literal["phase_fft", "opencv", "sift", "sift_normals"]] = None,
        align_opencv_kwargs: Optional[Dict[str, Any]] = None,
        align_sift_kwargs: Optional[Dict[str, Any]] = None,
        include_slip_axis_series: bool = False,
        slip_axis_use_directionality_mask: bool = False,
        include_scratch_series: bool = False,
        scratch_defect_config: Optional[DefectDetectionConfig] = None,
    ) -> Dict[str, Any]:
        """
        Per-frame wear volumes and/or optional slip-axis and scratch-evolution series.

        All frames must share the same 2D shape. When ``align_before`` is set, the
        corresponding alignment runs **first** (in-place on ``self.frames``); summary
        is stored in ``self.metadata["alignment"]`` and echoed under the return key
        ``alignment`` when alignment ran.

        Volume tables (``vs_reference``, ``incremental``) require both ``dx_mm`` and
        ``dy_mm``. Slip-axis and scratch series do not require pixel pitch.

        Args:
            dx_mm: Pixel pitch in mm along X for volume tables (optional if only slip/scratch).
            dy_mm: Pixel pitch in mm along Y for volume tables.
            reference_index: Reference frame for FFT/OpenCV alignment and ``vs_reference``.
            signed: Passed to :func:`~tmd.sequence.wear_analysis.wear_series_vs_reference`.
            top_fraction: Localization index top fraction for volume rows.
            align_before: If ``\"phase_fft\"``, ``\"opencv\"``, ``\"sift\"`` (SIFT on height), or
                ``\"sift_normals\"`` (SIFT on normals from height, then warp heights), align before metrics.
            align_opencv_kwargs: Extra kwargs for :meth:`align_height_maps_opencv` when
                ``align_before=\"opencv\"``.
            align_sift_kwargs: Optional kwargs for SIFT paths (``two_pass``, ``crop``, ``config``,
                and for ``sift_normals`` also ``normal_strength``, ``normal_normalize``).
            include_slip_axis_series: If True, add ``slip_axis_series`` (per-frame dicts).
            slip_axis_use_directionality_mask: If True, mask gradients using defect
                ``directionality_anomalies`` (requires defect detection per frame).
            include_scratch_series: If True, add ``scratch_series`` from defect scratch masks.
            scratch_defect_config: Optional :class:`~tmd.surface.types.DefectDetectionConfig`;
                defaults to ``output_mode=\"standard\"`` when scratch series is requested.
        """
        if not self.frames:
            raise ValueError("Cannot compute wear metrics on an empty sequence")

        has_volumes = dx_mm is not None and dy_mm is not None
        if (dx_mm is None) ^ (dy_mm is None):
            raise ValueError("dx_mm and dy_mm must both be set for volume wear tables, or both omitted")

        if not has_volumes and not include_slip_axis_series and not include_scratch_series and not align_before:
            raise ValueError(
                "sequential_wear_metrics: set dx_mm and dy_mm and/or include_slip_axis_series, "
                "include_scratch_series, or align_before"
            )

        import tmd.sequence.wear_analysis as wa

        out: Dict[str, Any] = {}
        align_info: Optional[Dict[str, Any]] = None
        if align_before == "phase_fft":
            align_info = self.align_height_maps_phase_fft(reference_index=reference_index)
        elif align_before == "opencv":
            ocv_kw = dict(align_opencv_kwargs or {})
            align_info = self.align_height_maps_opencv(reference_index=reference_index, **ocv_kw)
        elif align_before == "sift":
            sk = dict(align_sift_kwargs or {})
            cfg = sk.pop("config", None)
            align_info = self.align_height_maps_sift(
                reference_index=reference_index,
                two_pass=bool(sk.pop("two_pass", True)),
                crop=bool(sk.pop("crop", True)),
                config=cfg,
            )
            if sk:
                raise ValueError(f"Unexpected keys in align_sift_kwargs: {sorted(sk)}")
        elif align_before == "sift_normals":
            sk = dict(align_sift_kwargs or {})
            cfg = sk.pop("config", None)
            n_strength = float(sk.pop("normal_strength", 1.0))
            n_norm = bool(sk.pop("normal_normalize", False))
            align_info = self.align_height_maps_from_normals(
                reference_index=reference_index,
                two_pass=bool(sk.pop("two_pass", True)),
                crop=bool(sk.pop("crop", True)),
                config=cfg,
                normal_strength=n_strength,
                normal_normalize=n_norm,
            )
            if sk:
                raise ValueError(f"Unexpected keys in align_sift_kwargs: {sorted(sk)}")
        elif align_before is not None:
            raise ValueError(
                "align_before must be None, 'phase_fft', 'opencv', 'sift', or 'sift_normals'"
            )

        if align_info is not None:
            out["alignment"] = align_info

        f = [np.asarray(x, dtype=np.float64) for x in self.frames]

        if has_volumes:
            out["vs_reference"] = wa.wear_series_vs_reference(
                f,
                reference_index=reference_index,
                dx=float(dx_mm),
                dy=float(dy_mm),
                top_fraction=top_fraction,
                signed=signed,
            )
            out["incremental"] = wa.wear_incremental_series(
                f, dx=float(dx_mm), dy=float(dy_mm), top_fraction=top_fraction
            )

        if include_slip_axis_series:
            from tmd.surface.defects import detect_surface_defects

            slip_cfg = DefectDetectionConfig(output_mode="standard")
            slip_rows: List[Dict[str, Any]] = []
            for i, frame in enumerate(self.frames):
                z = np.asarray(frame, dtype=np.float64)
                dmask: Optional[np.ndarray] = None
                if slip_axis_use_directionality_mask:
                    res = detect_surface_defects(z.astype(np.float32), slip_cfg)
                    raw = res["defects"]["directionality_anomalies"].get("mask")
                    if raw is None:
                        raise ValueError(
                            "directionality mask missing; use DefectDetectionConfig(output_mode='standard')"
                        )
                    dmask = np.asarray(raw, dtype=bool)
                metrics = wa.slip_axis_metrics(z, direction_mask=dmask)
                slip_rows.append({"frame_index": i, **metrics})
            out["slip_axis_series"] = slip_rows

        if include_scratch_series:
            from tmd.surface.defects import detect_surface_defects

            cfg = scratch_defect_config or DefectDetectionConfig(output_mode="standard")
            masks: List[np.ndarray] = []
            for frame in self.frames:
                res = detect_surface_defects(np.asarray(frame, dtype=np.float32), cfg)
                scratch_entry = res["defects"]["scratches"]
                m = scratch_entry.get("mask")
                if m is None:
                    raise ValueError(
                        "scratch mask missing; use DefectDetectionConfig(output_mode='standard' or 'full')"
                    )
                masks.append(np.asarray(m, dtype=bool))
            out["scratch_series"] = wa.scratch_series_metrics(masks)

        return out

    def apply_transformations(self) -> List[np.ndarray]:
        """
        Apply all defined transformations to frames.
        
        Returns:
            List of transformed frame arrays
        """
        transformed_frames = []
        for i, frame in enumerate(self.frames):
            transform = self.get_transformation(i) or {}
            transformed = frame.copy()
            
            # Apply scaling if specified
            if 'scaling' in transform:
                scaling = transform['scaling']
                if isinstance(scaling, (list, tuple)) and len(scaling) >= 3:
                    transformed = transformed * scaling[2]  # Z-scale
                elif isinstance(scaling, (int, float)):
                    transformed = transformed * scaling
                    
            # Apply thresholding if specified
            if 'threshold' in transform:
                threshold = transform['threshold']
                if isinstance(threshold, dict):
                    transformed = threshold_height_map(
                        transformed,
                        min_height=threshold.get('min'),
                        max_height=threshold.get('max'),
                        replacement=threshold.get('replacement'),
                    )
                    
            # Apply offset if specified
            if 'offset' in transform:
                offset = transform['offset']
                if isinstance(offset, (int, float)):
                    transformed = transformed + offset
                    
            transformed_frames.append(transformed)
        return transformed_frames

    def calculate_statistics(self) -> Dict[str, List[Any]]:
        """
        Calculate statistics for all frames in the sequence.
        
        Returns:
            Dictionary of statistical measures across all frames
        """
        stats = {
            'timestamps': self.frame_timestamps.copy(),
            'min': [],
            'max': [],
            'mean': [],
            'median': [],
            'std': [],
            'range': [],
            'sum': [],
            'valid_pixels': []
        }
        
        transformed_frames = self.apply_transformations()
        for frame in transformed_frames:
            # Handle NaN values appropriately
            valid_mask = ~np.isnan(frame)
            valid_data = frame[valid_mask]
            
            if valid_data.size > 0:
                stats['min'].append(float(np.min(valid_data)))
                stats['max'].append(float(np.max(valid_data)))
                stats['mean'].append(float(np.mean(valid_data)))
                stats['median'].append(float(np.median(valid_data)))
                stats['std'].append(float(np.std(valid_data)))
                stats['range'].append(float(np.max(valid_data) - np.min(valid_data)))
                stats['sum'].append(float(np.sum(valid_data)))
                stats['valid_pixels'].append(int(np.sum(valid_mask)))
            else:
                # Handle empty/all-NaN frames
                stats['min'].append(float('nan'))
                stats['max'].append(float('nan'))
                stats['mean'].append(float('nan'))
                stats['median'].append(float('nan'))
                stats['std'].append(float('nan'))
                stats['range'].append(float('nan'))
                stats['sum'].append(float('nan'))
                stats['valid_pixels'].append(0)
                
        return stats

    def to_dict(
        self,
        *,
        include_derived: bool = False,
        wear_dx_mm: Optional[float] = None,
        wear_dy_mm: Optional[float] = None,
        wear_reference_index: int = 0,
        wear_signed: bool = False,
        wear_top_fraction: float = 0.10,
        wear_align_before: Optional[Literal["phase_fft", "opencv", "sift", "sift_normals"]] = None,
        wear_align_opencv_kwargs: Optional[Dict[str, Any]] = None,
        wear_align_sift_kwargs: Optional[Dict[str, Any]] = None,
        wear_include_slip_axis_series: bool = False,
        wear_slip_axis_use_directionality_mask: bool = False,
        wear_include_scratch_series: bool = False,
        wear_scratch_defect_config: Optional[DefectDetectionConfig] = None,
    ) -> Dict[str, Any]:
        """
        Convert the sequence into a dictionary representation suitable for export.

        Args:
            include_derived: If True, attach ``derived`` with per-frame statistics and
                optional ``wear`` when any wear-related option is set (see below).
            wear_dx_mm: Pixel pitch in mm along X for volume wear tables (optional if only
                slip/scratch/alignment outputs are requested).
            wear_dy_mm: Pixel pitch in mm along Y for volume wear tables.
            wear_reference_index: Reference frame index for alignment and ``vs_reference``.
            wear_signed: Passed through to ``wear_series_vs_reference``.
            wear_top_fraction: Localization top fraction for volume rows.
            wear_align_before: If set, align frames before computing wear (see
                :meth:`sequential_wear_metrics`).
            wear_align_opencv_kwargs: Extra kwargs for OpenCV alignment when ``wear_align_before=\"opencv\"``.
            wear_align_sift_kwargs: Optional kwargs for ``sift`` / ``sift_normals`` alignment
                (see :meth:`sequential_wear_metrics`).
            wear_include_slip_axis_series: Include per-frame ``slip_axis_metrics`` rows.
            wear_slip_axis_use_directionality_mask: Use directionality defect mask for slip axis.
            wear_include_scratch_series: Include ``scratch_series_metrics`` table.
            wear_scratch_defect_config: Optional defect config for scratch masks.

        Returns:
            Dictionary containing sequence data (and optional ``derived`` block).
        """
        base: Dict[str, Any] = {
            "name": self.name,
            "metadata": self.metadata,
            "frames": self.frames,
            "timestamps": self.frame_timestamps,
            "frame_metadata": self.frame_metadata,
            "transformations": self.transformations,
        }
        if not include_derived:
            return base
        derived: Dict[str, Any] = {"statistics": self.calculate_statistics()}
        if (wear_dx_mm is None) ^ (wear_dy_mm is None):
            raise ValueError("wear_dx_mm and wear_dy_mm must both be set or both omitted")
        wear_pitch_ok = wear_dx_mm is not None and wear_dy_mm is not None
        want_wear = bool(self.frames) and (
            wear_pitch_ok
            or wear_include_slip_axis_series
            or wear_include_scratch_series
            or wear_align_before is not None
        )
        if want_wear:
            derived["wear"] = self.sequential_wear_metrics(
                dx_mm=wear_dx_mm,
                dy_mm=wear_dy_mm,
                reference_index=wear_reference_index,
                signed=wear_signed,
                top_fraction=wear_top_fraction,
                align_before=wear_align_before,
                align_opencv_kwargs=wear_align_opencv_kwargs,
                align_sift_kwargs=wear_align_sift_kwargs,
                include_slip_axis_series=wear_include_slip_axis_series,
                slip_axis_use_directionality_mask=wear_slip_axis_use_directionality_mask,
                include_scratch_series=wear_include_scratch_series,
                scratch_defect_config=wear_scratch_defect_config,
            )
        base["derived"] = derived
        return base

    # -------------------------------------------------------------------------
    # Simplified Export Methods using the Centralized Factory
    # -------------------------------------------------------------------------
    
    def export(self, output_path: str, format_type: str, **kwargs) -> Optional[str]:
        """
        Generic export method using the SequenceExporterFactory.
        
        Args:
            output_path: Path to the output file
            format_type: Export format type ('gif', 'video', 'powerpoint', etc.)
            **kwargs: Format-specific export options
            
        Returns:
            Path to the exported file if successful, None otherwise
        """
        # Apply transformations to get the frames to export
        frames = self.apply_transformations()
        
        if not frames:
            logger.error("No frames available to export")
            return None
            
        # Add sequence metadata if not provided
        if 'title' not in kwargs and format_type.lower() in ['powerpoint', 'pptx']:
            kwargs['title'] = self.name
            
        # Add timestamps if available and not provided
        if 'timestamps' not in kwargs and self.frame_timestamps:
            kwargs['timestamps'] = self.frame_timestamps
            
        # Use the factory to perform the export
        return SequenceExporterFactory.export_sequence(
            frames, output_path, format_type, **kwargs
        )
    
    def export_to_gif(self, output_path: str, fps: float = 10.0, **kwargs) -> Optional[str]:
        """
        Export the sequence to an animated GIF.
        
        Args:
            output_path: Path for the output GIF file
            fps: Frames per second (default: 10.0)
            **kwargs: Additional export options
            
        Returns:
            Path to the exported GIF if successful, None otherwise
        """
        # Use the factory's specialized method
        frames = self.apply_transformations()
        return SequenceExporterFactory.export_gif(frames, output_path, fps, **kwargs)
        
    def export_to_video(self, output_path: str, fps: float = 30.0, **kwargs) -> Optional[str]:
        """
        Export the sequence to a video file (MP4).
        
        Args:
            output_path: Path for the output video file
            fps: Frames per second (default: 30.0)
            **kwargs: Additional export options
            
        Returns:
            Path to the exported video if successful, None otherwise
        """
        # Use the factory's specialized method
        frames = self.apply_transformations()
        return SequenceExporterFactory.export_video(frames, output_path, fps, **kwargs)
        
    def export_to_powerpoint(self, output_path: str, **kwargs) -> Optional[str]:
        """
        Export the sequence to a PowerPoint presentation.
        
        Args:
            output_path: Path for the output PPTX file
            **kwargs: Additional export options
            
        Returns:
            Path to the exported presentation if successful, None otherwise
        """
        # Use the factory's specialized method
        frames = self.apply_transformations()
        
        # Add sequence name as title if not provided
        if 'title' not in kwargs:
            kwargs['title'] = self.name
            
        return SequenceExporterFactory.export_powerpoint(frames, output_path, **kwargs)
    
    def export_frames_as_images(self, 
                               output_dir: str, 
                               format_type: str = 'png', 
                               **kwargs) -> List[str]:
        """
        Export individual frames as separate image files.
        
        Args:
            output_dir: Directory where images should be saved
            format_type: Image format ('png', 'jpg', 'tif', etc.)
            **kwargs: Additional export options
            
        Returns:
            List of paths to saved image files
        """
        frames = self.apply_transformations()
        
        # Add timestamps if available and not provided
        if 'timestamps' not in kwargs and self.frame_timestamps:
            kwargs['timestamps'] = self.frame_timestamps
            
        # Use base filename from sequence name if not provided
        if 'base_filename' not in kwargs:
            kwargs['base_filename'] = self.name.replace(' ', '_').lower()
            
        return SequenceExporterFactory.export_frames_as_images(
            frames, output_dir, format_type, **kwargs
        )
    
    def get_supported_export_formats(self) -> List[str]:
        """
        Get a list of supported export formats.
        
        Returns:
            List of supported format names
        """
        return SequenceExporterFactory.supported_formats()
    
    # -------------------------------------------------------------------------
    # Data Storage Methods
    # -------------------------------------------------------------------------
    
    def save_to_npz(self, filepath: str) -> bool:
        """
        Save the sequence to a compressed NPZ file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to dictionary
            data = self.to_dict()
            
            # NPZ doesn't handle dictionaries directly, so convert each frame
            for i, frame in enumerate(data['frames']):
                data[f'frame_{i}'] = frame
                
            # Remove the frames list to avoid duplication
            data.pop('frames')
            
            # Save to NPZ file
            np.savez_compressed(filepath, **data)
            logger.info(f"Sequence saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving sequence: {e}")
            return False

    @classmethod
    def load_from_npz(cls, filepath: str) -> Optional['TMDSequence']:
        """
        Load a sequence from a compressed NPZ file.
        
        Args:
            filepath: Path to the NPZ file
            
        Returns:
            TMDSequence object or None if loading failed
        """
        try:
            # Load the NPZ file
            data = np.load(filepath, allow_pickle=True)
            
            # Create a new sequence with the saved name
            sequence = cls(name=str(data['name']))
            
            # Load metadata if available
            if 'metadata' in data:
                sequence.metadata = data['metadata'].item() if data['metadata'].dtype == np.object_ else {}
                
            # Get timestamps and transformations
            timestamps = data['timestamps'] if 'timestamps' in data else []
            transformations = data['transformations'].item() if 'transformations' in data else {}
            
            # Get frame metadata if available
            frame_metadata = data['frame_metadata'] if 'frame_metadata' in data else []
            
            # Find all frame keys (exclude e.g. frame_metadata by requiring frame_<digits>)
            frame_re = re.compile(r"^frame_(\d+)$")
            frame_keys = [k for k in data.keys() if frame_re.match(str(k))]

            # Add each frame to the sequence
            for key in sorted(frame_keys, key=lambda k: int(frame_re.match(str(k)).group(1))):
                idx = int(frame_re.match(str(key)).group(1))
                
                # Get timestamp for this frame
                timestamp = timestamps[idx] if idx < len(timestamps) else None
                
                # Get transformation for this frame
                transform = transformations.get(str(idx), {}) if isinstance(transformations, dict) else {}
                
                # Get metadata for this frame
                metadata = frame_metadata[idx] if idx < len(frame_metadata) else {}
                
                # Add the frame to the sequence
                sequence.add_frame(data[key], timestamp=timestamp, 
                                   metadata=metadata, transformation=transform)
                
            logger.info(f"Sequence loaded from {filepath} with {len(frame_keys)} frames")
            return sequence
            
        except Exception as e:
            logger.error(f"Error loading sequence from NPZ: {e}")
            return None
            
    def __str__(self) -> str:
        """Return a string representation of the TMDSequence."""
        return f"TMDSequence('{self.name}', {len(self.frames)} frames)"
    
    def __repr__(self) -> str:
        """Return a detailed string representation for debugging."""
        return (
            f"TMDSequence(name={self.name!r}, frames={len(self.frames)}, "
            f"metadata_keys={list(self.metadata.keys())})"
        )