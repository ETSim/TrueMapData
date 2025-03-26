"""
Heightmap alignment module for TMD.

This module provides utilities for aligning height maps based on their
principal orientations and centroids, using rotation and translation.
"""

import logging
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import cv2
from scipy.ndimage import rotate, shift
from scipy.signal import correlate2d

# Setup logging
logger = logging.getLogger(__name__)

def compute_centroid(height_map: np.ndarray) -> Tuple[float, float]:
    """
    Compute the centroid (center of mass) of a height map.
    
    Args:
        height_map: 2D array of height values
        
    Returns:
        Tuple (cx, cy) of centroid coordinates
    """
    # Create coordinate grids
    rows, cols = height_map.shape
    y_coords, x_coords = np.mgrid[:rows, :cols]
    
    # Handle potential NaN values
    valid_mask = ~np.isnan(height_map)
    if not np.any(valid_mask):
        return cols / 2, rows / 2  # Default to center if all values are NaN
    
    # Use height values as weights
    weights = height_map.copy()
    weights[~valid_mask] = 0  # Set NaNs to zero
    
    # Avoid division by zero
    total_weight = np.sum(weights)
    if total_weight == 0:
        return cols / 2, rows / 2  # Default to center if all weights are zero
    
    # Calculate weighted centroid
    cx = np.sum(weights * x_coords) / total_weight
    cy = np.sum(weights * y_coords) / total_weight
    
    return cx, cy

def compute_principal_orientation(height_map: np.ndarray) -> float:
    """
    Compute the principal orientation of a height map using weighted PCA.
    
    Args:
        height_map: 2D array of height values
        
    Returns:
        Angle in degrees of the principal orientation
    """
    # Get centroid
    cx, cy = compute_centroid(height_map)
    
    # Create coordinate grids
    rows, cols = height_map.shape
    y_coords, x_coords = np.mgrid[:rows, :cols]
    
    # Shift coordinates to be relative to centroid
    x_centered = x_coords - cx
    y_centered = y_coords - cy
    
    # Handle NaN values
    valid_mask = ~np.isnan(height_map)
    if not np.any(valid_mask):
        return 0.0  # Default angle if all values are NaN
    
    # Use height values as weights
    weights = height_map.copy()
    weights[~valid_mask] = 0  # Set NaNs to zero
    
    # Calculate weighted covariance matrix elements
    total_weight = np.sum(weights)
    if total_weight == 0:
        return 0.0  # Default angle if all weights are zero
    
    sigma_xx = np.sum(weights * x_centered**2) / total_weight
    sigma_yy = np.sum(weights * y_centered**2) / total_weight
    sigma_xy = np.sum(weights * x_centered * y_centered) / total_weight
    
    # Construct covariance matrix
    cov_matrix = np.array([[sigma_xx, sigma_xy], [sigma_xy, sigma_yy]])
    
    # Compute eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Get the index of the largest eigenvalue
        idx = np.argmax(eigenvalues)
        
        # Get the corresponding eigenvector
        principal_vector = eigenvectors[:, idx]
        
        # Calculate the angle in degrees
        angle = np.degrees(np.arctan2(principal_vector[1], principal_vector[0]))
        
        return angle
    except np.linalg.LinAlgError:
        logger.warning("Failed to compute eigenvectors for principal orientation")
        return 0.0

def align_heightmaps(
    reference: np.ndarray,
    target: np.ndarray,
    method: str = 'principal_orientation',
    normalize: bool = True,
    interpolation_order: int = 1
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Align a target height map to a reference height map using rotation and translation.
    
    Args:
        reference: Reference height map
        target: Target height map to be aligned
        method: Alignment method ('principal_orientation' or 'phase_correlation')
        normalize: Whether to normalize height maps before alignment
        interpolation_order: Order of spline interpolation for rotation (0-5)
        
    Returns:
        Tuple of (aligned_target, transformation_parameters)
    """
    # Make copies to avoid modifying originals
    ref = reference.copy()
    tgt = target.copy()
    
    # Normalize height maps if requested
    if normalize:
        for hmap in [ref, tgt]:
            valid_mask = ~np.isnan(hmap)
            if np.any(valid_mask):
                hmap_min = np.nanmin(hmap)
                hmap_range = np.nanmax(hmap) - hmap_min
                if hmap_range > 0:
                    hmap[valid_mask] = (hmap[valid_mask] - hmap_min) / hmap_range
    
    # Ensure consistent shapes
    if ref.shape != tgt.shape:
        logger.warning(f"Height maps have different shapes: {ref.shape} vs {tgt.shape}")
        # Resize target to match reference
        try:
            tgt = cv2.resize(tgt, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_CUBIC)
        except Exception as e:
            logger.error(f"Failed to resize height map: {e}")
            return target, {"rotation": 0.0, "translation_x": 0.0, "translation_y": 0.0}
    
    if method == 'principal_orientation':
        # Compute principal orientation for both height maps
        theta_ref = compute_principal_orientation(ref)
        theta_tgt = compute_principal_orientation(tgt)
        
        # Calculate rotation angle to align
        rotation_angle = theta_ref - theta_tgt
        
        # Rotate target to align with reference
        tgt_rotated = rotate(tgt, rotation_angle, reshape=False, order=interpolation_order)
        
        # Compute centroids
        cx_ref, cy_ref = compute_centroid(ref)
        cx_tgt, cy_tgt = compute_centroid(tgt_rotated)
        
        # Calculate translation
        tx = cx_ref - cx_tgt
        ty = cy_ref - cy_tgt
        
        # Apply translation
        tgt_aligned = shift(tgt_rotated, (ty, tx), order=interpolation_order)
        
        # Store transformation parameters
        transformation = {
            "rotation": rotation_angle,
            "translation_x": tx,
            "translation_y": ty,
            "method": "principal_orientation"
        }
        
    elif method == 'phase_correlation':
        # Handle NaN values by replacing with zeros
        ref_clean = ref.copy()
        tgt_clean = tgt.copy()
        ref_clean[np.isnan(ref_clean)] = 0
        tgt_clean[np.isnan(tgt_clean)] = 0
        
        try:
            # Use OpenCV's phase correlation
            shifts, _ = cv2.phaseCorrelate(
                ref_clean.astype(np.float32),
                tgt_clean.astype(np.float32)
            )
            tx, ty = shifts
            
            # Apply translation
            tgt_aligned = shift(tgt, (ty, tx), order=interpolation_order)
            
            # Store transformation parameters
            transformation = {
                "rotation": 0.0,
                "translation_x": tx,
                "translation_y": ty,
                "method": "phase_correlation"
            }
            
        except Exception as e:
            logger.warning(f"Phase correlation failed: {e}. Using fallback correlation.")
            # Fallback: use cross-correlation
            correlation = correlate2d(ref_clean, tgt_clean, mode='same')
            y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
            
            # Calculate translation (centered at the middle)
            tx = x - ref.shape[1] // 2
            ty = y - ref.shape[0] // 2
            
            # Apply translation
            tgt_aligned = shift(tgt, (ty, tx), order=interpolation_order)
            
            # Store transformation parameters
            transformation = {
                "rotation": 0.0,
                "translation_x": tx,
                "translation_y": ty,
                "method": "cross_correlation"
            }
    
    else:
        logger.warning(f"Unknown alignment method: {method}. No alignment performed.")
        tgt_aligned = tgt
        transformation = {
            "rotation": 0.0,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "method": "none"
        }
    
    return tgt_aligned, transformation

def align_sequence_to_reference(
    sequence: List[np.ndarray],
    reference_idx: int = 0,
    method: str = 'principal_orientation'
) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
    """
    Align all frames in a sequence to a reference frame.
    
    Args:
        sequence: List of height maps
        reference_idx: Index of the reference frame
        method: Alignment method
        
    Returns:
        Tuple of (aligned_sequence, transformation_parameters)
    """
    if not sequence:
        return [], []
    
    # Ensure reference index is valid
    if reference_idx < 0 or reference_idx >= len(sequence):
        logger.warning(f"Invalid reference index: {reference_idx}. Using first frame.")
        reference_idx = 0
    
    # Get reference frame
    reference = sequence[reference_idx]
    
    # Initialize results
    aligned_sequence = [reference.copy()]  # Reference frame is already aligned
    transformations = [{"rotation": 0.0, "translation_x": 0.0, "translation_y": 0.0, "method": method}]
    
    # Align all other frames
    for i, frame in enumerate(sequence):
        if i == reference_idx:
            continue  # Skip reference frame
        
        aligned_frame, transformation = align_heightmaps(reference, frame, method=method)
        aligned_sequence.append(aligned_frame)
        transformations.append(transformation)
    
    return aligned_sequence, transformations

def evaluate_alignment_quality(
    reference: np.ndarray, 
    aligned: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate the quality of height map alignment.
    
    Args:
        reference: Reference height map
        aligned: Aligned height map
        
    Returns:
        Dictionary of quality metrics
    """
    # Handle NaN values
    valid_mask = ~(np.isnan(reference) | np.isnan(aligned))
    if not np.any(valid_mask):
        return {
            "mse": float('inf'),
            "rmse": float('inf'),
            "mae": float('inf'),
            "correlation": 0.0,
            "valid_ratio": 0.0
        }
    
    # Extract valid values
    ref_valid = reference[valid_mask]
    aligned_valid = aligned[valid_mask]
    
    # Calculate metrics
    mse = np.mean((ref_valid - aligned_valid) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(ref_valid - aligned_valid))
    
    # Calculate correlation
    try:
        correlation = np.corrcoef(ref_valid, aligned_valid)[0, 1]
    except Exception as e:
        correlation = 0.0
    
    # Calculate ratio of valid pixels
    valid_ratio = np.sum(valid_mask) / valid_mask.size
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "correlation": float(correlation),
        "valid_ratio": float(valid_ratio)
    }

"""
Alignment module for height map sequences.

This module provides functionality to align height map sequences in time,
correcting for frame rate differences, missing frames, or temporal offsets.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

# Set up logger
logger = logging.getLogger(__name__)


def align_sequences(
    reference: List[np.ndarray],
    target: List[np.ndarray],
    method: str = "dtw",
    max_offset: Optional[int] = None,
    similarity_metric: str = "correlation",
    window_size: Optional[int] = None
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Align a target sequence to match a reference sequence.
    
    Args:
        reference: Reference sequence of height maps
        target: Target sequence to be aligned
        method: Alignment method ("dtw", "cross_correlation", "uniform")
        max_offset: Maximum frame offset to consider
        similarity_metric: Metric for comparing frames
        window_size: Window size for alignment algorithms
        
    Returns:
        Tuple of (aligned_sequence, alignment_info)
    """
    # Check input sequences
    if not reference or not target:
        raise ValueError("Empty sequence provided")
    
    # Pick alignment method
    if method == "uniform":
        return _align_uniform(reference, target)
    elif method == "cross_correlation":
        return _align_by_cross_correlation(reference, target, max_offset, similarity_metric)
    elif method == "dtw":
        return _align_by_dtw(reference, target, window_size, similarity_metric)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def _align_uniform(
    reference: List[np.ndarray],
    target: List[np.ndarray]
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Uniform alignment by resampling the target sequence.
    
    Args:
        reference: Reference sequence of height maps
        target: Target sequence to be aligned
        
    Returns:
        Tuple of (aligned_sequence, alignment_info)
    """
    # Simple linear resampling
    ref_len = len(reference)
    target_len = len(target)
    
    # Create indices for sampling target frames
    if target_len == 1:
        # Special case: repeat single target frame
        aligned = [target[0]] * ref_len
    else:
        indices = np.linspace(0, target_len - 1, ref_len)
        
        # Get frames with interpolation
        aligned = []
        for idx in indices:
            # Get integer indices for interpolation
            idx_low = int(np.floor(idx))
            idx_high = int(np.ceil(idx))
            
            if idx_low == idx_high:
                # Exact match
                aligned.append(target[idx_low])
            else:
                # Linear interpolation
                weight_high = idx - idx_low
                weight_low = 1.0 - weight_high
                
                # Interpolate frames
                frame = target[idx_low] * weight_low + target[idx_high] * weight_high
                aligned.append(frame)
    
    # Return aligned sequence and info
    info = {
        "method": "uniform",
        "original_length": target_len,
        "aligned_length": ref_len,
        "scale_factor": ref_len / target_len if target_len > 0 else 1.0
    }
    
    return aligned, info


def _align_by_cross_correlation(
    reference: List[np.ndarray],
    target: List[np.ndarray],
    max_offset: Optional[int] = None,
    similarity_metric: str = "correlation"
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Align sequences using cross-correlation to find optimal offset.
    
    Args:
        reference: Reference sequence of height maps
        target: Target sequence to be aligned
        max_offset: Maximum frame offset to consider
        similarity_metric: Metric for comparing frames
        
    Returns:
        Tuple of (aligned_sequence, alignment_info)
    """
    # Determine maximum offset to search
    ref_len = len(reference)
    target_len = len(target)
    
    if max_offset is None:
        max_offset = min(ref_len, target_len) // 2
    
    # Function to calculate similarity between frames
    def calc_similarity(frame1, frame2):
        if similarity_metric == "correlation":
            # Flatten arrays
            flat1 = frame1.flatten()
            flat2 = frame2.flatten()
            # Calculate correlation
            return np.corrcoef(flat1, flat2)[0, 1]
        elif similarity_metric == "mse":
            # Mean squared error (negated for similarity)
            return -np.mean((frame1 - frame2) ** 2)
        elif similarity_metric == "mae":
            # Mean absolute error (negated for similarity)
            return -np.mean(np.abs(frame1 - frame2))
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
    
    # Try different offsets
    offsets = list(range(-max_offset, max_offset + 1))
    similarities = []
    
    for offset in offsets:
        # Calculate overlapping region
        if offset >= 0:
            ref_start = offset
            ref_end = min(ref_len, target_len + offset)
            target_start = 0
            target_end = ref_end - offset
        else:
            ref_start = 0
            ref_end = min(ref_len, target_len + offset)
            target_start = -offset
            target_end = target_start + (ref_end - ref_start)
            
        # Check if we have valid region
        if ref_end <= ref_start or target_end <= target_start:
            # No overlap
            similarities.append(-float('inf'))
            continue
            
        # Calculate similarity on overlapping frames
        sim_values = []
        for i in range(ref_start, ref_end):
            j = i - offset - ref_start + target_start
            if 0 <= j < target_len:
                sim = calc_similarity(reference[i], target[j])
                sim_values.append(sim)
        
        # Average similarity over all frames
        if sim_values:
            similarities.append(np.mean(sim_values))
        else:
            similarities.append(-float('inf'))
    
    # Find best offset
    best_idx = np.argmax(similarities)
    best_offset = offsets[best_idx]
    
    # Create aligned sequence with the best offset
    aligned = []
    alignment_indices = []
    
    # Fill with frames from target sequence
    for i in range(ref_len):
        j = i - best_offset
        if 0 <= j < target_len:
            # Copy frame from target
            aligned.append(target[j])
            alignment_indices.append(j)
        else:
            # Fill with None for missing frames
            aligned.append(None)
            alignment_indices.append(None)
    
    # Replace None frames with nearest valid frames
    for i, frame in enumerate(aligned):
        if frame is None:
            # Find nearest valid frame
            valid_indices = [j for j, f in enumerate(aligned) if f is not None]
            if valid_indices:
                nearest_idx = min(valid_indices, key=lambda j: abs(j - i))
                aligned[i] = aligned[nearest_idx]
    
    # Return aligned sequence and info
    info = {
        "method": "cross_correlation",
        "similarity_metric": similarity_metric,
        "best_offset": best_offset,
        "max_offset_searched": max_offset,
        "original_length": target_len,
        "aligned_length": ref_len,
        "similarity_score": similarities[best_idx],
        "alignment_indices": alignment_indices
    }
    
    return aligned, info


def _align_by_dtw(
    reference: List[np.ndarray],
    target: List[np.ndarray],
    window_size: Optional[int] = None,
    similarity_metric: str = "correlation"
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Align sequences using Dynamic Time Warping.
    
    Args:
        reference: Reference sequence of height maps
        target: Target sequence to be aligned
        window_size: Window size for DTW algorithm
        similarity_metric: Metric for comparing frames
        
    Returns:
        Tuple of (aligned_sequence, alignment_info)
    """
    # Function to calculate distance between frames
    def calc_distance(frame1, frame2):
        if similarity_metric == "correlation":
            # Flatten arrays
            flat1 = frame1.flatten()
            flat2 = frame2.flatten()
            # Convert correlation to distance
            correlation = np.corrcoef(flat1, flat2)[0, 1]
            return 1.0 - correlation  # Distance (0 = identical)
        elif similarity_metric == "mse":
            # Mean squared error
            return np.mean((frame1 - frame2) ** 2)
        elif similarity_metric == "mae":
            # Mean absolute error
            return np.mean(np.abs(frame1 - frame2))
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
    
    # Check for FastDTW library
    try:
        from fastdtw import fastdtw
        use_fastdtw = True
    except ImportError:
        use_fastdtw = False
        logger.warning("FastDTW library not found. Using standard DTW implementation.")
    
    # Compute the DTW distance matrix
    ref_len = len(reference)
    target_len = len(target)
    
    if use_fastdtw:
        # Use FastDTW library
        # Precompute frame features
        ref_features = [frame.flatten() for frame in reference]
        target_features = [frame.flatten() for frame in target]
        
        # Compute DTW
        distance, path = fastdtw(
            ref_features, 
            target_features,
            radius=window_size or max(ref_len, target_len) // 10,
            dist=lambda x, y: calc_distance(x.reshape(reference[0].shape), y.reshape(target[0].shape))
        )
    else:
        # Standard DTW implementation
        # Create distance matrix
        distances = np.zeros((ref_len, target_len))
        for i in range(ref_len):
            for j in range(target_len):
                distances[i, j] = calc_distance(reference[i], target[j])
        
        # Compute DTW matrix
        dtw_matrix = np.zeros((ref_len + 1, target_len + 1)) + float('inf')
        dtw_matrix[0, 0] = 0
        
        # Fill DTW matrix
        for i in range(1, ref_len + 1):
            # Apply window constraint if specified
            if window_size:
                j_start = max(1, i - window_size)
                j_end = min(target_len + 1, i + window_size + 1)
            else:
                j_start = 1
                j_end = target_len + 1
                
            for j in range(j_start, j_end):
                cost = distances[i-1, j-1]
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],      # Insertion
                    dtw_matrix[i-1, j-1],    # Match
                    dtw_matrix[i, j-1]       # Deletion
                )
        
        # Backtrack to find path
        path = []
        i, j = ref_len, target_len
        while i > 0 and j > 0:
            path.append((i-1, j-1))
            
            # Find minimal cost direction
            min_cost = min(
                dtw_matrix[i-1, j],
                dtw_matrix[i-1, j-1],
                dtw_matrix[i, j-1]
            )
            
            if min_cost == dtw_matrix[i-1, j-1]:
                # Diagonal move
                i -= 1
                j -= 1
            elif min_cost == dtw_matrix[i-1, j]:
                # Move up
                i -= 1
            else:
                # Move left
                j -= 1
                
        # Reverse path to get correct order
        path.reverse()
    
    # Create aligned sequence using the DTW path
    aligned = []
    alignment_indices = []
    
    # Extract frames according to path
    path_dict = {}
    for ref_idx, target_idx in path:
        path_dict[ref_idx] = target_idx
    
    # Create aligned sequence frame by frame
    for i in range(ref_len):
        if i in path_dict:
            # Get corresponding target frame
            j = path_dict[i]
            aligned.append(target[j])
            alignment_indices.append(j)
        else:
            # No mapping, use nearest available
            nearest = min(path_dict.items(), key=lambda x: abs(x[0] - i))
            aligned.append(target[path_dict[nearest[0]]])
            alignment_indices.append(path_dict[nearest[0]])
    
    # Return aligned sequence and info
    info = {
        "method": "dtw",
        "similarity_metric": similarity_metric,
        "dtw_path": path,
        "original_length": target_len,
        "aligned_length": ref_len,
        "alignment_indices": alignment_indices
    }
    
    return aligned, info


def resample_sequence(
    sequence: List[np.ndarray],
    target_length: int,
    method: str = "linear"
) -> List[np.ndarray]:
    """
    Resample a sequence to a target length.
    
    Args:
        sequence: Sequence of height maps
        target_length: Desired length after resampling
        method: Interpolation method ("linear", "nearest", "cubic")
        
    Returns:
        Resampled sequence
    """
    if not sequence:
        return []
        
    source_length = len(sequence)
    
    if source_length == target_length:
        return sequence.copy()
        
    # Create indices for resampling
    source_indices = np.arange(source_length)
    target_indices = np.linspace(0, source_length - 1, target_length)
    
    # For single frame, repeat it
    if source_length == 1:
        return [sequence[0]] * target_length
    
    # Choose interpolation method
    if method == "nearest":
        # Nearest neighbor interpolation
        indices = np.round(target_indices).astype(int)
        return [sequence[idx] for idx in indices]
    elif method == "cubic" and source_length >= 4:
        # Cubic interpolation requires at least 4 points
        from scipy.interpolate import interp1d
        
        # Get frame dimensions
        frame_shape = sequence[0].shape
        
        # Reshape sequence to [frames, pixels]
        flat_sequence = np.array([frame.flatten() for frame in sequence])
        
        # Create interpolator
        interp_func = interp1d(
            source_indices, 
            flat_sequence, 
            axis=0, 
            kind="cubic", 
            bounds_error=False, 
            fill_value="extrapolate"
        )
        
        # Interpolate
        flat_resampled = interp_func(target_indices)
        
        # Reshape back to original frame dimensions
        return [frame.reshape(frame_shape) for frame in flat_resampled]
    else:
        # Linear interpolation (default)
        resampled = []
        for idx in target_indices:
            # Get integer indices for interpolation
            idx_low = int(np.floor(idx))
            idx_high = int(np.ceil(idx))
            
            if idx_low == idx_high:
                # Exact match
                resampled.append(sequence[idx_low])
            else:
                # Linear interpolation
                weight_high = idx - idx_low
                weight_low = 1.0 - weight_high
                
                # Interpolate frames
                frame = sequence[idx_low] * weight_low + sequence[idx_high] * weight_high
                resampled.append(frame)
                
        return resampled
