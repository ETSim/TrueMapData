"""
Alignment tools for TMD sequences.
This module provides functionality to align multiple TMD sequences,
which is useful for comparing sequences that might have spatial offsets.
"""

import os
import logging
import numpy as np
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import concurrent.futures  # Add threading support
import math  # For dividing work among threads

# Import OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenCV (cv2) is not installed. Alignment functions will have limited functionality.")

logger = logging.getLogger(__name__)


def align_heightmaps(
    source_map: np.ndarray,
    target_map: np.ndarray,
    method: str = 'ecc',
    max_iterations: int = 1000,
    epsilon: float = 1e-6,
    scale_factor: float = 1.0,
    enable_rotation: bool = False,
    max_angle: float = 10.0,  # Maximum rotation angle in degrees
    angle_step: float = 1.0    # Step size for rotation search in degrees
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a source heightmap to a target heightmap.
    
    Args:
        source_map: Source heightmap to align
        target_map: Target heightmap to align to
        method: Alignment method ('ecc', 'orb', 'sift', 'template')
        max_iterations: Maximum number of iterations for ECC method
        epsilon: Convergence threshold for ECC method
        scale_factor: Scale factor for resizing images before alignment (for speed)
        enable_rotation: Whether to find and apply rotation for better alignment
        max_angle: Maximum rotation angle to search in degrees
        angle_step: Step size for rotation angle search in degrees
        
    Returns:
        Tuple of (aligned_heightmap, transformation_matrix)
    """
    if not HAS_OPENCV:
        logger.error("OpenCV is required for heightmap alignment")
        return source_map, np.eye(2, 3)

    # Resize source if shape mismatch
    if source_map.shape != target_map.shape:
        logger.warning(f"Resizing source from {source_map.shape} to {target_map.shape}")
        source_map = cv2.resize(source_map, (target_map.shape[1], target_map.shape[0]), interpolation=cv2.INTER_CUBIC)

    best_angle = 0.0
    best_error = float('inf')
    best_warp_matrix = np.eye(2, 3, dtype=np.float32)
    best_aligned = None
    
    # If rotation is enabled, try different rotation angles
    if enable_rotation:
        logger.info(f"Searching for best rotation angle within ±{max_angle} degrees")
        angles = np.arange(-max_angle, max_angle + angle_step, angle_step)
        
        # Speed optimization: First try with downsampled images
        small_source = cv2.resize(source_map, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        small_target = cv2.resize(target_map, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        
        for angle in angles:
            # Create rotation matrix
            h, w = small_source.shape
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation to source
            rotated_source = cv2.warpAffine(small_source, rotation_matrix, (w, h), 
                                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            # Align the rotated source to target
            aligned_small, trans_matrix = _align_without_rotation(rotated_source, small_target, method, 
                                                         max_iterations, epsilon, scale_factor)
            
            # Calculate error
            diff = aligned_small - small_target
            error = np.mean(np.square(diff))
            
            if error < best_error:
                best_error = error
                best_angle = angle
        
        logger.info(f"Best rotation angle: {best_angle:.2f} degrees")
        
        # Apply the best rotation to the original source map
        h, w = source_map.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated_source = cv2.warpAffine(source_map, rotation_matrix, (w, h), 
                                      flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Align the rotated source to target
        best_aligned, translation_matrix = _align_without_rotation(rotated_source, target_map, method, 
                                                      max_iterations, epsilon, scale_factor)
        
        # Combine rotation and translation matrices
        # Convert 2x3 matrices to 3x3 homogeneous matrices for multiplication
        rotation_matrix_hom = np.vstack([rotation_matrix, [0, 0, 1]])
        translation_matrix_hom = np.vstack([translation_matrix, [0, 0, 1]])
        
        # Multiply the homogeneous matrices
        combined_matrix_hom = np.matmul(translation_matrix_hom, rotation_matrix_hom)
        
        # Convert back to 2x3 affine matrix
        best_warp_matrix = combined_matrix_hom[:2]
    else:
        # Align without rotation
        best_aligned, best_warp_matrix = _align_without_rotation(source_map, target_map, method, 
                                                   max_iterations, epsilon, scale_factor)
    
    return best_aligned, best_warp_matrix

def _align_without_rotation(
    source_map: np.ndarray,
    target_map: np.ndarray,
    method: str = 'ecc',
    max_iterations: int = 1000,
    epsilon: float = 1e-6,
    scale_factor: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Internal function to align without rotation.
    Extracted from the original align_heightmaps for code reuse.
    """
    def normalize(hmap):
        h_min, h_max = np.min(hmap), np.max(hmap)
        return ((hmap - h_min) / (h_max - h_min) * 255).astype(np.uint8) if h_max > h_min else np.zeros_like(hmap, dtype=np.uint8)

    source_norm = normalize(source_map)
    target_norm = normalize(target_map)

    # Scale down
    if scale_factor != 1.0:
        src_h, src_w = source_norm.shape
        new_h, new_w = int(src_h * scale_factor), int(src_w * scale_factor)
        source_small = cv2.resize(source_norm, (new_w, new_h), interpolation=cv2.INTER_AREA)
        target_small = cv2.resize(target_norm, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        source_small, target_small = source_norm, target_norm

    warp_matrix = np.eye(2, 3, dtype=np.float32)

    try:
        if method == 'ecc':
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iterations, epsilon)
            try:
                _, warp_matrix = cv2.findTransformECC(target_small, source_small, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
            except cv2.error as e:
                logger.warning(f"ECC alignment failed: {e}. Using identity transform.")
                warp_matrix = np.eye(2, 3, dtype=np.float32)

        elif method in ['orb', 'sift']:
            # Check if SIFT is available, otherwise fall back to ORB
            if method == 'sift' and hasattr(cv2, 'SIFT_create'):
                detector = cv2.SIFT_create()
            else:
                detector = cv2.ORB_create(1000)
                
            kp1, des1 = detector.detectAndCompute(target_small, None)
            kp2, des2 = detector.detectAndCompute(source_small, None)

            if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                norm_type = cv2.NORM_L2 if method == 'sift' else cv2.NORM_HAMMING
                matcher = cv2.BFMatcher(norm_type)
                try:
                    matches = matcher.knnMatch(des1, des2, k=2)
                    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

                    if len(good) >= 4:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                        
                        if len(good) > 10:
                            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            if H is not None:
                                warp_matrix = H[:2]
                        else:
                            result = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                            if result[0] is not None:
                                warp_matrix = result[0]
                    else:
                        logger.warning("Not enough good matches found for alignment using identity transform")
                except Exception as e:
                    logger.warning(f"Feature matching failed: {e}")
            else:
                logger.warning("Not enough keypoints found for alignment, using identity transform")

        elif method == 'template':
            try:
                res = cv2.matchTemplate(source_small, target_small, cv2.TM_CCOEFF_NORMED)
                _, _, _, max_loc = cv2.minMaxLoc(res)
                warp_matrix[0, 2], warp_matrix[1, 2] = max_loc
            except Exception as e:
                logger.warning(f"Template matching failed: {e}")

        else:
            logger.warning(f"Unknown alignment method: {method}")

    except Exception as e:
        logger.warning(f"Alignment failed with {method}: {e}")

    # Scale the transformation back if needed
    if scale_factor != 1.0:
        warp_matrix[:, 2] /= scale_factor

    # Apply the transformation to get the aligned map
    aligned_map = cv2.warpAffine(
        source_map, 
        warp_matrix, 
        (target_map.shape[1], target_map.shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, 
        borderMode=cv2.BORDER_REPLICATE
    )

    return aligned_map, warp_matrix


def align_sequence_to_reference(
    source_sequence: List[np.ndarray],
    reference_sequence: List[np.ndarray],
    method: str = 'ecc',
    frame_indices: Optional[List[int]] = None,
    **kwargs
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Align a sequence of heightmaps to a reference sequence.
    
    Args:
        source_sequence: Source sequence to align
        reference_sequence: Reference sequence to align to
        method: Alignment method ('ecc', 'orb', 'sift', 'template')
        frame_indices: Indices of frames to use for alignment calculation
        **kwargs: Additional keyword arguments passed to align_heightmaps
        
    Returns:
        Tuple of (aligned_sequence, transformation_matrices)
    """
    if not HAS_OPENCV:
        logger.error("OpenCV is required for sequence alignment")
        return source_sequence, [np.eye(2, 3)] * len(source_sequence)
        
    if not source_sequence or not reference_sequence:
        logger.error("Empty sequences")
        return source_sequence, [np.eye(2, 3)] * len(source_sequence)

    # If no frame indices provided, use all frames that are in both sequences
    if frame_indices is None:
        frame_indices = list(range(min(len(source_sequence), len(reference_sequence))))
    
    # Filter out invalid indices
    frame_indices = [i for i in frame_indices if i < len(source_sequence) and i < len(reference_sequence)]

    if not frame_indices:
        logger.error("No valid frame indices")
        return source_sequence, [np.eye(2, 3)] * len(source_sequence)

    warp_matrices = []
    aligned_frames = []

    # Align each specified frame
    for i in frame_indices:
        aligned, warp = align_heightmaps(source_sequence[i], reference_sequence[i], method, **kwargs)
        aligned_frames.append(aligned)
        warp_matrices.append(warp)

    # Calculate average transformation matrix
    avg_matrix = np.zeros_like(warp_matrices[0])
    for matrix in warp_matrices:
        avg_matrix += matrix
    avg_matrix /= len(warp_matrices)

    # Apply average transformation to all frames
    aligned_sequence = []
    transforms = []
    for frame in source_sequence:
        aligned = cv2.warpAffine(
            frame, 
            avg_matrix, 
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, 
            borderMode=cv2.BORDER_REPLICATE
        )
        aligned_sequence.append(aligned)
        transforms.append(avg_matrix.copy())  # Make a copy to be safe

    return aligned_sequence, transforms


def visualize_alignment(
    source: np.ndarray,
    target: np.ndarray,
    aligned: np.ndarray,
    output_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize the alignment of heightmaps.
    
    Args:
        source: Source heightmap before alignment
        target: Target heightmap to align to
        aligned: Source heightmap after alignment
        output_path: Path to save the visualization
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    # Create a figure with 3 subplots (not 6)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Define titles for each subplot
    titles = ["Source Heightmap", "Target Heightmap", "Aligned Heightmap"]
    
    # Plot each heightmap
    for ax, img, title in zip(axes, [source, target, aligned], titles):
        im = ax.imshow(img, cmap='terrain')
        ax.set_title(title)
        ax.set_xticks([])  # Remove x ticks
        ax.set_yticks([])  # Remove y ticks
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Save figure if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        
    # Show figure if requested
    if show:
        plt.show()
    
    return fig


def simple_overlap_alignment(
    source_map: np.ndarray,
    target_map: np.ndarray,
    search_range: int = 100,
    step_size: int = 1,
    num_threads: int = 4,
    use_gpu: bool = False,
    multi_scale: bool = False,
    enable_rotation: bool = False,
    angle_range: float = 10.0,  # Maximum rotation angle in degrees
    angle_step: float = 2.0,    # Rotation step size in degrees
    use_icp: bool = False       # Whether to use ICP for refinement
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a source heightmap to a target heightmap using simple overlap search.
    
    This function performs a brute-force search for the best overlap position
    by trying different translations and finding the one that minimizes the
    squared difference between the overlapping regions.
    
    Args:
        source_map: Source heightmap to align
        target_map: Target heightmap to align to
        search_range: Maximum pixel offset to search in each direction
        step_size: Step size for the search grid (1 = check every pixel)
        num_threads: Number of threads to use for parallel processing
        use_gpu: Whether to use GPU acceleration if available
        multi_scale: Whether to use multi-scale search for faster estimation
        enable_rotation: Whether to include rotation in the alignment search
        angle_range: Maximum rotation angle to search in degrees (±angle_range)
        angle_step: Step size for rotation search in degrees
        use_icp: Whether to refine alignment using ICP algorithm
        
    Returns:
        Tuple of (aligned_heightmap, transformation_matrix)
    """
    if not HAS_OPENCV:
        logger.error("OpenCV is required for heightmap alignment")
        return source_map, np.eye(2, 3)

    # Resize source if shape mismatch
    if source_map.shape != target_map.shape:
        logger.warning(f"Resizing source from {source_map.shape} to {target_map.shape}")
        source_map = cv2.resize(source_map, (target_map.shape[1], target_map.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Short-circuit for rotation if enabled
    if enable_rotation:
        return _rotation_aware_overlap_alignment(
            source_map, target_map, search_range, step_size, num_threads,
            use_gpu, multi_scale, angle_range, angle_step
        )

    # Continue with non-rotational alignment
    source_norm = source_map.copy()
    target_norm = target_map.copy()
    
    # Normalize both maps to [0,1] if their ranges differ significantly
    source_range = np.max(source_norm) - np.min(source_norm)
    target_range = np.max(target_norm) - np.min(target_norm)
    
    if abs(source_range - target_range) > 0.001:
        source_norm = (source_norm - np.min(source_norm)) / (source_range or 1.0)
        target_norm = (target_norm - np.min(target_norm)) / (target_range or 1.0)
    
    height, width = source_norm.shape
    
    # Try GPU-accelerated version if requested and CUDA is available
    if use_gpu and HAS_OPENCV:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.info("Using GPU acceleration for overlap search")
                return _gpu_overlap_search(source_map, target_map, source_norm, target_norm, search_range, step_size)
        except (ImportError, AttributeError, cv2.error) as e:
            logger.warning(f"GPU acceleration failed: {e}. Falling back to CPU.")
    
    # Speed-optimized multi-scale approach
    if multi_scale:
        return _fast_multiscale_overlap_search(
            source_map, source_norm, target_norm, 
            search_range, step_size, num_threads
        )
    
    # Use multi-threaded version for full resolution search if requested
    if num_threads > 1:
        logger.info(f"Using {num_threads} threads for overlap search")
        return _threaded_overlap_search(
            source_map, source_norm, target_norm, 
            search_range, step_size, num_threads
        )
    
    # Fall back to single-threaded processing
    min_error = float('inf')
    best_dx, best_dy = 0, 0
    
    # Limit search around the initial estimate
    min_dx = max(-search_range, -width//2)
    max_dx = min(search_range, width//2)
    min_dy = max(-search_range, -height//2)
    max_dy = min(search_range, height//2)
    
    # Try various translations and find the one with minimal error
    for dy in range(min_dy, max_dy + 1, step_size):
        for dx in range(min_dx, max_dx + 1, step_size):
            # Calculate overlapping regions
            y1_src = max(0, dy)
            y2_src = min(height, height+dy)
            x1_src = max(0, dx)
            x2_src = min(width, width+dx)
            
            y1_target = max(0, -dy)
            y2_target = min(height, height-dy)
            x1_target = max(0, -dx)
            x2_target = min(width, width-dx)
            
            # Skip if overlap region is too small
            overlap_height = min(y2_src - y1_src, y2_target - y1_target)
            overlap_width = min(x2_src - x1_src, x2_target - x1_target)
            
            if overlap_height <= height/4 or overlap_width <= width/4:
                continue
            
            # Calculate squared error for the overlapping region
            src_region = source_norm[y1_src:y2_src, x1_src:x2_src]
            target_region = target_norm[y1_target:y2_target, x1_target:x2_target]
            
            # Ensure regions are the same size
            min_h = min(src_region.shape[0], target_region.shape[0])
            min_w = min(src_region.shape[1], target_region.shape[1])
            
            if min_h == 0 or min_w == 0:
                continue
                
            src_region = src_region[:min_h, :min_w]
            target_region = target_region[:min_h, :min_w]
            
            # Calculate mean squared error
            diff = src_region - target_region
            mse = np.mean(diff * diff)  # Faster than np.square
            
            if mse < min_error:
                min_error = mse
                best_dx, best_dy = dx, dy
                    
    # Create transformation matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = best_dx
    warp_matrix[1, 2] = best_dy
    
    # Apply the transformation to get the aligned map
    aligned_map = cv2.warpAffine(
        source_map, 
        warp_matrix, 
        (target_map.shape[1], target_map.shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, 
        borderMode=cv2.BORDER_REPLICATE
    )
                        
    logger.info(f"Best overlap found at offset: ({best_dx}, {best_dy}) with MSE: {min_error:.6f}")
    
    # Add ICP refinement if requested
    if use_icp:
        logger.info("Refining alignment using ICP")
        # Use the result from overlap alignment as initial guess for ICP
        initial_aligned, initial_transform = aligned_map, warp_matrix
        
        try:
            # Apply ICP with the initial alignment as starting point
            refined_aligned, refined_transform = icp_alignment(
                source_map, 
                target_map, 
                sample_rate=15,  # Increase for speed, decrease for accuracy
                max_iterations=30
            )
            
            # Combine the transformations
            combined_transform = np.eye(2, 3, dtype=np.float32)
            combined_transform[0, 2] = initial_transform[0, 2] + refined_transform[0, 2]
            combined_transform[1, 2] = initial_transform[1, 2] + refined_transform[1, 2]
            
            logger.info(f"ICP refinement completed. Final offset: ({combined_transform[0, 2]:.1f}, {combined_transform[1, 2]:.1f})")
            return refined_aligned, combined_transform
        
        except Exception as e:
            logger.warning(f"ICP refinement failed: {e}. Using overlap alignment result.")
            return initial_aligned, initial_transform
    
    # If ICP not used, return the result from overlap alignment
    return aligned_map, warp_matrix


def _rotation_aware_overlap_alignment(
    source_map: np.ndarray,
    target_map: np.ndarray,
    search_range: int = 100,
    step_size: int = 1,
    num_threads: int = 4,
    use_gpu: bool = False,
    multi_scale: bool = False,
    angle_range: float = 10.0,
    angle_step: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alignment that considers both translation and rotation.
    
    This is a coarse-to-fine approach that first finds an approximate
    rotation angle using downsampled images, then refines the translation
    at the best rotation angle.
    """
    # Normalize maps
    source_norm = source_map.copy()
    target_norm = target_map.copy()
    
    source_min, source_max = np.min(source_norm), np.max(source_norm)
    target_min, target_max = np.min(target_norm), np.max(target_norm)
    
    if source_max > source_min:
        source_norm = (source_norm - source_min) / (source_max - source_min)
    if target_max > target_min:
        target_norm = (target_norm - target_min) / (target_max - target_min)
    
    # Step 1: Find approximate rotation using downsampled images
    # Downsample for speed (1/4 resolution)
    height, width = source_norm.shape
    ds_height, ds_width = height // 4, width // 4
    ds_source = cv2.resize(source_norm, (ds_width, ds_height), interpolation=cv2.INTER_AREA)
    ds_target = cv2.resize(target_norm, (ds_width, ds_height), interpolation=cv2.INTER_AREA)
    
    # Center coordinates for rotation
    center = (ds_width // 2, ds_height // 2)
    
    # Test different rotation angles
    best_angle = 0.0
    best_error = float('inf')
    best_shift = (0, 0)
    
    logger.info(f"Searching for best rotation in range ±{angle_range} degrees with step {angle_step}")
    
    angles = np.arange(-angle_range, angle_range + angle_step, angle_step)
    for angle in angles:
        # Rotate source image
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(ds_source, rotation_matrix, (ds_width, ds_height), 
                                borderMode=cv2.BORDER_REPLICATE)
        
        # Use cross-correlation to find best shift quickly
        correlation = cv2.matchTemplate(rotated, ds_target, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(correlation)
        
        # Adjust to center-based coordinates
        shift_x = max_loc[0] - ds_width // 2
        shift_y = max_loc[1] - ds_height // 2
        
        # Apply shift
        shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        aligned = cv2.warpAffine(rotated, shift_matrix, (ds_width, ds_height), 
                                borderMode=cv2.BORDER_REPLICATE)
        
        # Calculate error
        diff = aligned - ds_target
        mse = np.mean(np.square(diff))
        
        if mse < best_error:
            best_error = mse
            best_angle = angle
            best_shift = (shift_x * 4, shift_y * 4)  # Scale back to original size
    
    logger.info(f"Best rotation angle: {best_angle} degrees with initial offset: {best_shift}")
    
    # Step 2: Apply best rotation to original image
    center_full = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center_full, best_angle, 1.0)
    rotated_source = cv2.warpAffine(source_map, rotation_matrix, (width, height), 
                                  borderMode=cv2.BORDER_REPLICATE)
    
    # Step 3: Fine-tune translation using the rotated image
    # Reuse existing alignment functions but with a smaller search range around best_shift
    rotated_norm = rotated_source.copy()
    if source_max > source_min:
        rotated_norm = (rotated_norm - source_min) / (source_max - source_min)
    
    # Use the faster multi-scale alignment method
    if multi_scale:
        # Center search around best_shift from rotation search
        dx, dy = best_shift
        aligned_map, trans_matrix = _fast_multiscale_overlap_search(
            rotated_source, rotated_norm, target_norm, 
            search_range // 2, step_size, num_threads, 
            initial_dx=dx, initial_dy=dy
        )
    else:
        # Use threaded search centered around best_shift
        dx, dy = best_shift
        aligned_map, trans_matrix = _threaded_overlap_search(
            rotated_source, rotated_norm, target_norm, 
            search_range // 2, step_size, num_threads,
            initial_dx=dx, initial_dy=dy
        )
    
    # Combine rotation and translation
    # We need to convert both matrices to 3x3 homogeneous format for proper combination
    rotation_matrix_hom = np.eye(3, dtype=np.float32)
    rotation_matrix_hom[:2, :] = rotation_matrix
    
    trans_matrix_hom = np.eye(3, dtype=np.float32)
    trans_matrix_hom[:2, :] = trans_matrix
    
    # Compute the combined transformation matrix
    final_matrix_hom = np.dot(trans_matrix_hom, rotation_matrix_hom)
    
    # Convert back to 2x3 for OpenCV
    final_matrix = final_matrix_hom[:2, :]
    
    return aligned_map, final_matrix


def _fast_multiscale_overlap_search(
    source_map: np.ndarray,
    source_norm: np.ndarray,
    target_norm: np.ndarray,
    search_range: int = 100,
    step_size: int = 1,
    num_threads: int = 4,
    initial_dx: int = 0,
    initial_dy: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized multi-scale search that uses progressively refined grids.
    
    This implementation reduces redundant computations and uses a more
    efficient multi-scale strategy for faster convergence.
    """
    height, width = source_norm.shape
    logger.info("Using optimized multi-scale approach for faster alignment")
    
    # Initial estimates
    best_dx, best_dy = initial_dx, initial_dy
    
    # Define scales for coarse-to-fine approach (powers of 2 work well)
    scales = [8, 4, 2]
    
    # Process each scale
    for scale in scales:
        # Skip if the image would be too small at this scale
        if min(height, width) <= scale * 10:
            continue
            
        # Calculate dimensions at this scale
        small_height, small_width = height // scale, width // scale
        
        # Adjust search range for this scale
        # At coarser scales, we search more broadly but with larger steps
        # At finer scales, we search more narrowly but with smaller steps
        scaled_range = max(10, search_range // scale)
        scaled_step = max(1, step_size // (9 // scale))  # Adaptive step size
        
        # Quick resize of source and target
        small_source = cv2.resize(source_norm, (small_width, small_height), interpolation=cv2.INTER_AREA)
        small_target = cv2.resize(target_norm, (small_width, small_height), interpolation=cv2.INTER_AREA)
        
        # Adjust search center based on previous best
        initial_dx_scaled = best_dx // scale
        initial_dy_scaled = best_dy // scale
        
        # Define search bounds
        min_dx = max(-scaled_range + initial_dx_scaled, -small_width//2)
        max_dx = min(scaled_range + initial_dx_scaled, small_width//2)
        min_dy = max(-scaled_range + initial_dy_scaled, -small_height//2)
        max_dy = min(scaled_range + initial_dy_scaled, small_height//2)
        
        # Use threaded search if requested
        if num_threads > 1 and (max_dx - min_dx) * (max_dy - min_dy) > 1000:  # Only if search space is large enough
            # Create limited offset list for this scale
            offsets = [(dy, dx) 
                      for dy in range(min_dy, max_dy + 1, scaled_step) 
                      for dx in range(min_dx, max_dx + 1, scaled_step)]
            
            # Use threaded search at this scale
            min_error = float('inf')
            
            # Split search space among threads
            batch_size = len(offsets) // num_threads + 1
            offset_batches = [offsets[i:i+batch_size] for i in range(0, len(offsets), batch_size)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                results = []
                for batch in offset_batches:
                    results.append(executor.submit(_search_batch, small_source, small_target, 
                                                 batch, small_height, small_width))
                
                # Get results
                for future in concurrent.futures.as_completed(results):
                    error, dx, dy = future.result()
                    if error < min_error:
                        min_error = error
                        best_dx_scaled = dx
                        best_dy_scaled = dy
        else:
            # Single-threaded search
            min_error = float('inf')
            best_dx_scaled, best_dy_scaled = initial_dx_scaled, initial_dy_scaled
            
            for dy in range(min_dy, max_dy + 1, scaled_step):
                for dx in range(min_dx, max_dx + 1, scaled_step):
                    # Calculate overlapping regions
                    y1_src = max(0, dy)
                    y2_src = min(small_height, small_height + dy)
                    x1_src = max(0, dx)
                    x2_src = min(small_width, small_width + dx)
                    
                    y1_target = max(0, -dy)
                    y2_target = min(small_height, small_height - dy)
                    x1_target = max(0, -dx)
                    x2_target = min(small_width, small_width - dx)
                    
                    # Skip if overlap region is too small
                    overlap_height = min(y2_src - y1_src, y2_target - y1_target)
                    overlap_width = min(x2_src - x1_src, x2_target - x1_target)
                    
                    if overlap_height <= small_height/4 or overlap_width <= small_width/4:
                        continue
                    
                    # Extract regions
                    src_region = small_source[y1_src:y2_src, x1_src:x2_src]
                    target_region = small_target[y1_target:y2_target, x1_target:x2_target]
                    
                    # Ensure regions are the same size
                    min_h = min(src_region.shape[0], target_region.shape[0])
                    min_w = min(src_region.shape[1], target_region.shape[1])
                    
                    if min_h == 0 or min_w == 0:
                        continue
                        
                    src_region = src_region[:min_h, :min_w]
                    target_region = target_region[:min_h, :min_w]
                    
                    # Calculate MSE
                    diff = src_region - target_region
                    mse = np.mean(diff * diff)
                    
                    if mse < min_error:
                        min_error = mse
                        best_dx_scaled = dx
                        best_dy_scaled = dy
        
        # Update full-resolution estimates
        best_dx = best_dx_scaled * scale
        best_dy = best_dy_scaled * scale
        
        logger.info(f"Multi-scale estimation at 1/{scale}: offset ({best_dx_scaled}, {best_dy_scaled}) -> full size ({best_dx}, {best_dy})")
    
    # Create transformation matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = best_dx
    warp_matrix[1, 2] = best_dy
    
    # Apply the transformation
    aligned_map = cv2.warpAffine(
        source_map, 
        warp_matrix, 
        (source_norm.shape[1], source_norm.shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    logger.info(f"Multi-scale search found best overlap at ({best_dx}, {best_dy})")
    return aligned_map, warp_matrix

def _search_batch(source, target, offsets, height, width):
    """Helper function for multi-threaded multi-scale search."""
    min_error = float('inf')
    best_dx, best_dy = 0, 0
    
    for dy, dx in offsets:
        # Calculate overlapping regions
        y1_src = max(0, dy)
        y2_src = min(height, height + dy)
        x1_src = max(0, dx)
        x2_src = min(width, width + dx)
        
        y1_target = max(0, -dy)
        y2_target = min(height, height - dy)
        x1_target = max(0, -dx)
        x2_target = min(width, width - dx)
        
        # Skip if overlap region is too small
        overlap_height = min(y2_src - y1_src, y2_target - y1_target)
        overlap_width = min(x2_src - x1_src, x2_target - x1_target)
        
        if overlap_height <= height/4 or overlap_width <= width/4:
            continue
        
        # Extract regions
        src_region = source[y1_src:y2_src, x1_src:x2_src]
        target_region = target[y1_target:y2_target, x1_target:x2_target]
        
        # Ensure regions are the same size
        min_h = min(src_region.shape[0], target_region.shape[0])
        min_w = min(src_region.shape[1], target_region.shape[1])
        
        if min_h == 0 or min_w == 0:
            continue
            
        src_region = src_region[:min_h, :min_w]
        target_region = target_region[:min_h, :min_w]
        
        # Calculate MSE
        diff = src_region - target_region
        mse = np.mean(diff * diff)
        
        if mse < min_error:
            min_error = mse
            best_dx = dx
            best_dy = dy
            
    return min_error, best_dx, best_dy


def _threaded_overlap_search(
    source_map: np.ndarray,
    source_norm: np.ndarray,
    target_norm: np.ndarray,
    search_range: int,
    step_size: int,
    num_threads: int,
    initial_dx: int = 0,
    initial_dy: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-threaded implementation of overlap search.
    
    Args:
        source_map: Original source heightmap
        source_norm: Normalized source heightmap
        target_norm: Normalized target heightmap
        search_range: Maximum pixel offset to search
        step_size: Step size for the search grid
        num_threads: Number of threads to use
        initial_dx: Initial x offset estimate
        initial_dy: Initial y offset estimate
        
    Returns:
        Tuple of (aligned_heightmap, transformation_matrix)
    """
    height, width = source_norm.shape
    
    # Optimize by using early termination and work batching
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    # Use shared variables with lock to track best result across all threads
    lock = threading.Lock()
    best_result = {
        'error': float('inf'),
        'dx': initial_dx,
        'dy': initial_dy
    }
    
    # Determine search bounds around initial estimate
    min_dx = max(-search_range + initial_dx, -width//2)
    max_dx = min(search_range + initial_dx, width//2)
    min_dy = max(-search_range + initial_dy, -height//2)
    max_dy = min(search_range + initial_dy, height//2)
    
    # Pre-compute all possible dy, dx combinations to allow early termination
    offsets = [(dy, dx) 
               for dy in range(min_dy, max_dy + 1, step_size) 
               for dx in range(min_dx, max_dx + 1, step_size)]
    
    # Shuffle offsets to distribute work more evenly and increase chances of early termination
    import random
    random.shuffle(offsets)
    
    # Split offsets into batches for each thread
    batch_size = len(offsets) // num_threads + (1 if len(offsets) % num_threads else 0)
    offset_batches = [offsets[i:i+batch_size] for i in range(0, len(offsets), batch_size)]
    
    # Use numpy's faster operations 
    def calculate_mse(src_region, target_region):
        # Faster MSE calculation using numpy vectorized operations
        diff = src_region - target_region
        return np.mean(diff * diff)  # Faster than using np.square
    
    # Worker function to process a batch of offsets with early termination
    def process_batch(batch):
        local_best = {'error': float('inf'), 'dx': initial_dx, 'dy': initial_dy}
        early_stop = False
        
        # Process offsets in smaller chunks to check for early termination
        chunk_size = 50
        
        for i in range(0, len(batch), chunk_size):
            chunk = batch[i:i+chunk_size]
            
            # Check if we should terminate early
            with lock:
                if local_best['error'] > 0 and best_result['error'] < local_best['error'] * 0.7:  # Another thread found a solution 30% better
                    early_stop = True
                    break
            
            for dy, dx in chunk:
                # Calculate overlapping regions
                y1_src = max(0, dy)
                y2_src = min(height, height+dy)
                x1_src = max(0, dx)
                x2_src = min(width, width+dx)
                
                y1_target = max(0, -dy)
                y2_target = min(height, height-dy)
                x1_target = max(0, -dx)
                x2_target = min(width, width-dx)
                
                # Skip if overlap region is too small
                overlap_height = min(y2_src - y1_src, y2_target - y1_target)
                overlap_width = min(x2_src - x1_src, x2_target - x1_target)
                
                if overlap_height <= height/4 or overlap_width <= width/4:
                    continue
                
                # Calculate squared error for the overlapping region
                src_region = source_norm[y1_src:y2_src, x1_src:x2_src]
                target_region = target_norm[y1_target:y2_target, x1_target:x2_target]
                
                # Ensure regions are the same size
                min_h = min(src_region.shape[0], target_region.shape[0])
                min_w = min(src_region.shape[1], target_region.shape[1])
                
                if min_h == 0 or min_w == 0:
                    continue
                    
                src_region = src_region[:min_h, :min_w]
                target_region = target_region[:min_h, :min_w]
                
                # Calculate mean squared error - using faster calculation
                mse = calculate_mse(src_region, target_region)
                
                if mse < local_best['error']:
                    local_best = {'error': mse, 'dx': dx, 'dy': dy}
                    
                    # Update global best if this is better
                    with lock:
                        if mse < best_result['error']:
                            best_result['error'] = mse
                            best_result['dx'] = dx
                            best_result['dy'] = dy
        
        return local_best
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_batch, offset_batches))
    
    # Find the best result across all threads
    min_error, best_dx, best_dy = best_result['error'], best_result['dx'], best_result['dy']
    
    logger.info(f"Threaded search found best overlap at ({best_dx}, {best_dy}) with MSE: {min_error:.6f}")
    
    # Create transformation matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = best_dx
    warp_matrix[1, 2] = best_dy
    
    # Apply the transformation to get the aligned map
    aligned_map = cv2.warpAffine(
        source_map, 
        warp_matrix, 
        (source_norm.shape[1], source_norm.shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return aligned_map, warp_matrix


def _gpu_overlap_search(
    source_map: np.ndarray,
    target_map: np.ndarray,
    source_norm: np.ndarray,
    target_norm: np.ndarray,
    search_range: int,
    step_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated overlap search using OpenCV CUDA.
    
    Args:
        source_map: Original source heightmap
        target_map: Original target heightmap
        source_norm: Normalized source heightmap 
        target_norm: Normalized target heightmap
        search_range: Maximum pixel offset to search
        step_size: Step size for the search grid
        
    Returns:
        Tuple of (aligned_heightmap, transformation_matrix)
    """
    # Copy to GPU
    gpu_source = cv2.cuda_GpuMat()
    gpu_target = cv2.cuda_GpuMat()
    gpu_source.upload(source_norm.astype(np.float32))
    gpu_target.upload(target_norm.astype(np.float32))
    
    min_error = float('inf')
    best_dx, best_dy = 0, 0
    height, width = source_norm.shape
    
    # Create a temporary GPU mat for shifted source
    gpu_shifted = cv2.cuda_GpuMat(height, width, cv2.CV_32F)
    gpu_diff = cv2.cuda_GpuMat(height, width, cv2.CV_32F)
    
    for dy in range(-search_range, search_range+1, step_size):
        for dx in range(-search_range, search_range+1, step_size):
            # Create transformation matrix for the current shift
            shift_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
            
            # Apply transformation on GPU
            cv2.cuda.warpAffine(gpu_source, shift_matrix, (width, height), gpu_shifted, 
                             borderMode=cv2.BORDER_REPLICATE)
            
            # Calculate error
            cv2.cuda.subtract(gpu_shifted, gpu_target, gpu_diff)
            cv2.cuda.multiply(gpu_diff, gpu_diff, gpu_diff)
            
            # Calculate mean error
            error = cv2.cuda.absSum(gpu_diff)[0] / (width * height)
            
            if error < min_error:
                min_error = error
                best_dx, best_dy = dx, dy
    
    logger.info(f"GPU search found best overlap at ({best_dx}, {best_dy}) with error: {min_error:.6f}")
    
    # Create transformation matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = best_dx
    warp_matrix[1, 2] = best_dy
    
    # Apply the transformation to get the aligned map
    aligned_map = cv2.warpAffine(
        source_map, 
        warp_matrix, 
        (target_map.shape[1], target_map.shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return aligned_map, warp_matrix


def cross_correlation_alignment(
    source_map: np.ndarray,
    target_map: np.ndarray,
    max_offset: int = 100,
    enable_rotation: bool = False,  # Added rotation parameter
    max_angle: float = 10.0,        # Added max rotation angle
    angle_step: float = 1.0         # Added rotation step size
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a source heightmap to a target heightmap using cross-correlation.
    
    This function uses the maximum of the cross-correlation between the two
    heightmaps to find the best alignment.
    
    Args:
        source_map: Source heightmap to align
        target_map: Target heightmap to align to
        max_offset: Maximum pixel offset to allow
        enable_rotation: Whether to enable rotation in alignment
        max_angle: Maximum rotation angle to search in degrees
        angle_step: Step size for rotation angle search
        
    Returns:
        Tuple of (aligned_heightmap, transformation_matrix)
    """
    if not HAS_OPENCV:
        logger.error("OpenCV is required for heightmap alignment")
        return source_map, np.eye(2, 3)

    # Resize source if shape mismatch
    if source_map.shape != target_map.shape:
        logger.warning(f"Resizing source from {source_map.shape} to {target_map.shape}")
        source_map = cv2.resize(source_map, (target_map.shape[1], target_map.shape[0]), interpolation=cv2.INTER_CUBIC)

    # If rotation is enabled, use rotation-aware alignment
    if enable_rotation:
        return optimized_cross_correlation_alignment(
            source_map, target_map, max_offset, 
            use_gpu=False, quick_mode=False, 
            enable_rotation=True, 
            max_angle=max_angle, 
            angle_step=angle_step
        )

    # Continue with standard cross-correlation alignment
    # Normalize maps to 0-1 range for correlation
    source_norm = source_map.copy()
    target_norm = target_map.copy()
    
    # Fix: properly get min and max values
    source_min, source_max = np.min(source_norm), np.max(source_norm)
    target_min, target_max = np.min(target_norm), np.max(target_norm)
    
    # Normalize to [0,1] range
    if source_max > source_min:
        source_norm = (source_norm - source_min) / (source_max - source_min)
    if target_max > target_min:
        target_norm = (target_norm - target_min) / (target_max - target_min)

    # Compute cross-correlation
    correlation = cv2.matchTemplate(
        source_norm.astype(np.float32),
        target_norm.astype(np.float32),
        cv2.TM_CCORR_NORMED
    )
    
    # Find the location of maximum correlation
    _, _, _, max_loc = cv2.minMaxLoc(correlation)
    dx, dy = max_loc
    
    # Adjust offset relative to center
    height, width = source_norm.shape
    dx -= width // 2
    dy -= height // 2
    
    # Limit maximum offset
    dx = max(-max_offset, min(max_offset, dx))
    dy = max(-max_offset, min(max_offset, dy))
    
    # Create transformation matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = dx
    warp_matrix[1, 2] = dy
    
    # Apply the transformation to get the aligned map
    aligned_map = cv2.warpAffine(
        source_map, 
        warp_matrix, 
        (target_map.shape[1], target_map.shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    logger.info(f"Cross-correlation alignment found offset: ({dx}, {dy})")
    return aligned_map, warp_matrix


def optimized_cross_correlation_alignment(
    source_map: np.ndarray,
    target_map: np.ndarray,
    max_offset: int = 100,
    use_gpu: bool = False,
    quick_mode: bool = False,
    enable_rotation: bool = False,
    max_angle: float = 10.0,
    angle_step: float = 2.0,
    use_icp: bool = False       # Added ICP refinement option
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a source heightmap to a target heightmap using optimized cross-correlation.
    
    This function uses OpenCV's optimized phase correlation and GPU acceleration
    when available for faster processing.
    
    Args:
        source_map: Source heightmap to align
        target_map: Target heightmap to align to
        max_offset: Maximum pixel offset to allow
        use_gpu: Whether to use GPU acceleration if available
        quick_mode: Whether to use downscaled images for faster processing
        enable_rotation: Whether to search for the best rotation angle
        max_angle: Maximum rotation angle to search in degrees (±max_angle)
        angle_step: Step size for rotation angle search in degrees
        use_icp: Whether to refine the alignment using ICP
        
    Returns:
        Tuple of (aligned_heightmap, transformation_matrix)
    """
    if not HAS_OPENCV:
        logger.error("OpenCV is required for heightmap alignment")
        return source_map, np.eye(2, 3)

    # Resize source if shape mismatch
    if source_map.shape != target_map.shape:
        logger.warning(f"Resizing source from {source_map.shape} to {target_map.shape}")
        source_map = cv2.resize(source_map, (target_map.shape[1], target_map.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Quick mode - use downscaled images for faster processing
    scale_factor = 0.25 if (quick_mode and min(source_map.shape) > 500) else 1.0
    
    if scale_factor != 1.0:
        small_source = cv2.resize(source_map, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        small_target = cv2.resize(target_map, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        
        # Normalize these downsampled maps
        source_norm = cv2.normalize(small_source, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        target_norm = cv2.normalize(small_target, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    else:
        # Normalize maps for better correlation
        source_norm = cv2.normalize(source_map, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        target_norm = cv2.normalize(target_map, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    # For rotation search
    best_angle = 0.0
    best_correlation = -1.0  # Correlation ranges from -1 to 1, higher is better
    
    if enable_rotation:
        logger.info(f"Searching for best rotation angle within ±{max_angle} degrees")
        height, width = source_norm.shape
        center = (width // 2, height // 2)
        angles = np.arange(-max_angle, max_angle + angle_step, angle_step)
        
        for angle in angles:
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Apply rotation to source
            rotated_source = cv2.warpAffine(source_norm, rotation_matrix, (width, height), 
                                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            # Calculate correlation
            result = cv2.matchTemplate(rotated_source, target_norm, cv2.TM_CCORR_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            
            if max_val > best_correlation:
                best_correlation = max_val
                best_angle = angle
        
        logger.info(f"Best rotation angle: {best_angle:.2f} degrees with correlation: {best_correlation:.4f}")
        
        # Apply the best rotation to create a new source image
        height, width = source_norm.shape
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        source_norm = cv2.warpAffine(source_norm, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Save original rotation matrix for later
        original_rotation = rotation_matrix
    
    # Try GPU acceleration for the final correlation if requested
    if use_gpu and HAS_OPENCV:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                logger.info("Using GPU acceleration for cross-correlation")
                
                # Upload to GPU
                gpu_source = cv2.cuda_GpuMat()
                gpu_target = cv2.cuda_GpuMat()
                gpu_source.upload(source_norm)
                gpu_target.upload(target_norm)
                
                # Use GPU-accelerated correlation
                result_gpu = cv2.cuda.createTemplateMatching(gpu_source.type(), cv2.TM_CCORR_NORMED)
                result = result_gpu.match(gpu_source, gpu_target).download()
                
                _, _, _, max_loc = cv2.minMaxLoc(result)
                dx, dy = max_loc
        except (ImportError, AttributeError, cv2.error) as e:
            logger.warning(f"GPU acceleration failed: {e}. Falling back to CPU.")
            # Fall back to CPU version if GPU fails
            result = cv2.matchTemplate(source_norm, target_norm, cv2.TM_CCORR_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            dx, dy = max_loc
    else:
        # CPU version - use OpenCV's optimized implementation
        result = cv2.matchTemplate(source_norm, target_norm, cv2.TM_CCORR_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)
        dx, dy = max_loc

    # Adjust offset relative to center and limit maximum offset
    height, width = source_norm.shape
    dx -= width // 2
    dy -= height // 2
    
    dx = max(-max_offset, min(max_offset, dx))
    dy = max(-max_offset, min(max_offset, dy))
    
    # Scale back up if using quick mode
    if scale_factor != 1.0:
        dx = int(dx / scale_factor)
        dy = int(dy / scale_factor)

    # Create transformation matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = dx
    warp_matrix[1, 2] = dy
    
    # Combine with rotation if used
    if enable_rotation:
        # Scale rotation matrix if needed
        if scale_factor != 1.0:
            full_height, full_width = source_map.shape
            full_center = (full_width // 2, full_height // 2)
            full_rotation = cv2.getRotationMatrix2D(full_center, best_angle, 1.0)
            
            # Convert to homogeneous coordinates for multiplication
            warp_matrix_hom = np.vstack([warp_matrix, [0, 0, 1]])
            full_rotation_hom = np.vstack([full_rotation, [0, 0, 1]])
            
            # Multiply and convert back to 2x3
            combined_matrix_hom = np.matmul(warp_matrix_hom, full_rotation_hom)
            warp_matrix = combined_matrix_hom[:2]
        else:
            # Convert to homogeneous coordinates for multiplication
            warp_matrix_hom = np.vstack([warp_matrix, [0, 0, 1]])
            original_rotation_hom = np.vstack([original_rotation, [0, 0, 1]])
            
            # Multiply and convert back to 2x3
            combined_matrix_hom = np.matmul(warp_matrix_hom, original_rotation_hom)
            warp_matrix = combined_matrix_hom[:2]
    
    # Apply the final transformation
    aligned_map = cv2.warpAffine(
        source_map, 
        warp_matrix, 
        (target_map.shape[1], target_map.shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    if enable_rotation:
        logger.info(f"Optimized alignment with rotation {best_angle:.2f}° and offset: ({dx}, {dy})")
    else:
        logger.info(f"Optimized cross-correlation found offset: ({dx}, {dy})")
    
    # Add ICP refinement if requested
    if use_icp:
        logger.info("Refining alignment using ICP")
        # Use the result from correlation alignment as initial guess for ICP
        initial_aligned, initial_transform = aligned_map, warp_matrix
        
        try:
            # Apply ICP with the initial alignment as starting point
            refined_aligned, refined_transform = icp_alignment(
                source_map, 
                target_map, 
                sample_rate=15,
                max_iterations=30
            )
            
            # Combine the transformations (simplified for 2D)
            combined_transform = np.eye(2, 3, dtype=np.float32)
            combined_transform[0, 2] = initial_transform[0, 2] + refined_transform[0, 2]
            combined_transform[1, 2] = initial_transform[1, 2] + refined_transform[1, 2]
            
            logger.info(f"ICP refinement completed. Final offset: ({combined_transform[0, 2]:.1f}, {combined_transform[1, 2]:.1f})")
            return refined_aligned, combined_transform
        
        except Exception as e:
            logger.warning(f"ICP refinement failed: {e}. Using correlation alignment result.")
            return initial_aligned, initial_transform
    
    # If ICP not used, return the result from correlation alignment
    return aligned_map, warp_matrix


def phase_correlation_alignment(
    source_map: np.ndarray,
    target_map: np.ndarray,
    enable_rotation: bool = False,
    max_angle: float = 10.0,
    angle_step: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a source heightmap to a target heightmap using phase correlation.
    
    This function uses phase correlation to find the translation between images.
    
    Args:
        source_map: Source heightmap to align
        target_map: Target heightmap to align to
        enable_rotation: Whether to search for the best rotation angle
        max_angle: Maximum rotation angle to search in degrees (±max_angle)
        angle_step: Step size for rotation angle search in degrees
        
    Returns:
        Tuple of (aligned_heightmap, transformation_matrix)
    """
    if not HAS_OPENCV:
        logger.error("OpenCV is required for heightmap alignment")
        return source_map, np.eye(2, 3)

    # Resize source if shape mismatch
    if source_map.shape != target_map.shape:
        logger.warning(f"Resizing source from {source_map.shape} to {target_map.shape}")
        source_map = cv2.resize(source_map, (target_map.shape[1], target_map.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Handle rotation if enabled
    if enable_rotation:
        # Use rotation-aware search approach
        return _rotation_aware_phase_correlation(source_map, target_map, max_angle, angle_step)

    # Normalize to 8-bit for phase correlation
    source_norm = cv2.normalize(source_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    target_norm = cv2.normalize(target_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Find shift using phase correlation
    shift, response = cv2.phaseCorrelate(
        source_norm.astype(np.float32),
        target_norm.astype(np.float32)
    )
    
    # Create transformation matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = shift[0]
    warp_matrix[1, 2] = shift[1]
    
    # Apply the transformation to get the aligned map
    aligned_map = cv2.warpAffine(
        source_map, 
        warp_matrix, 
        (target_map.shape[1], target_map.shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    logger.info(f"Phase correlation found shift: ({shift[0]:.2f}, {shift[1]:.2f}) with response: {response:.4f}")
    return aligned_map, warp_matrix

def _rotation_aware_phase_correlation(
    source_map: np.ndarray,
    target_map: np.ndarray,
    max_angle: float = 10.0,
    angle_step: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotation-aware phase correlation alignment.
    
    Args:
        source_map: Source heightmap to align
        target_map: Target heightmap to align to
        max_angle: Maximum rotation angle to search in degrees
        angle_step: Step size for rotation search in degrees
        
    Returns:
        Tuple of (aligned_heightmap, transformation_matrix)
    """
    # Normalize maps
    source_norm = cv2.normalize(source_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    target_norm = cv2.normalize(target_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Downsize for faster rotation search
    height, width = source_norm.shape
    ds_height, ds_width = height // 4, width // 4
    ds_source = cv2.resize(source_norm, (ds_width, ds_height), interpolation=cv2.INTER_AREA)
    ds_target = cv2.resize(target_norm, (ds_width, ds_height), interpolation=cv2.INTER_AREA)
    
    # Search for best rotation
    best_angle = 0.0
    best_response = -float('inf')
    best_shift = (0.0, 0.0)
    center = (ds_width // 2, ds_height // 2)
    
    for angle in np.arange(-max_angle, max_angle + angle_step, angle_step):
        # Create rotation matrix
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation to source
        rotated = cv2.warpAffine(ds_source, rot_matrix, (ds_width, ds_height), 
                               borderMode=cv2.BORDER_REPLICATE)
        
        # Find shift using phase correlation
        try:
            shift, response = cv2.phaseCorrelate(
                rotated.astype(np.float32),
                ds_target.astype(np.float32)
            )
            
            if response > best_response:
                best_response = response
                best_angle = angle
                best_shift = shift
        except Exception as e:
            logger.warning(f"Phase correlation failed for angle {angle}: {e}")
    
    logger.info(f"Best rotation angle: {best_angle:.2f}° with response: {best_response:.4f}")
    
    # Scale shift back to original size
    scaled_shift = (best_shift[0] * 4, best_shift[1] * 4)
    
    # Apply best rotation to full resolution image
    full_center = (width // 2, height // 2)
    rot_matrix = cv2.getRotationMatrix2D(full_center, best_angle, 1.0)
    rotated_source = cv2.warpAffine(source_map, rot_matrix, (width, height), 
                                  borderMode=cv2.BORDER_REPLICATE)
    
    # Apply shift to create transformation matrix
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_matrix[0, 2] = scaled_shift[0]
    warp_matrix[1, 2] = scaled_shift[1]
    
    # Combine rotation and translation
    combined_matrix = np.vstack([rot_matrix, [0, 0, 1]])  # Make homogeneous
    translation = np.array([[1, 0, scaled_shift[0]], [0, 1, scaled_shift[1]], [0, 0, 1]])
    
    # Compute combined transformation
    final_transform = np.matmul(translation, combined_matrix)
    final_transform = final_transform[:2, :]  # Convert back to 2x3
    
    # Apply transformation
    aligned_map = cv2.warpAffine(
        source_map,
        final_transform,
        (target_map.shape[1], target_map.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    
    return aligned_map, final_transform

def _heightmap_to_point_cloud(height_map: np.ndarray, sample_rate: int = 10) -> np.ndarray:
    """
    Convert a heightmap to a 3D point cloud.
    
    Args:
        height_map: Input heightmap.
        sample_rate: Sample every nth point to reduce computation (higher = faster but less accurate).
    
    Returns:
        Nx3 array of 3D points.
    """
    rows, cols = height_map.shape
    
    # Create a mesh grid for x and y coordinates
    y, x = np.mgrid[0:rows:sample_rate, 0:cols:sample_rate]
    
    # Sample the height map at the grid locations
    z = height_map[::sample_rate, ::sample_rate]
    
    # Create 3D points
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    
    return points

def _point_cloud_to_heightmap(points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert a 3D point cloud back to a heightmap.
    
    Args:
        points: Nx3 array of 3D points (x, y, z).
        shape: Shape of the output heightmap.
    
    Returns:
        Reconstructed heightmap.
    """
    from scipy.interpolate import griddata
    
    rows, cols = shape
    grid_y, grid_x = np.mgrid[0:rows, 0:cols]
    
    # Extract x, y, z coordinates from points
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # Interpolate z-values onto a regular grid
    heightmap = griddata(
        (x, y), z, (grid_x, grid_y), 
        method='linear', 
        fill_value=np.mean(z)
    )
    
    return heightmap

def _find_closest_points(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Find the closest points in the target for each point in the source.
    
    Args:
        source: Nx3 array of source points.
        target: Mx3 array of target points.
    
    Returns:
        Nx3 array of closest target points.
    """
    indices = np.zeros(len(source), dtype=np.int32)
    
    # For each source point, find the index of the closest target point
    for i, point in enumerate(source):
        # Calculate distances to all target points
        distances = np.sum((target - point) ** 2, axis=1)
        # Find the index of the point with the minimum distance
        indices[i] = np.argmin(distances)
    
    # Return the target points corresponding to the indices
    return target[indices]

def _compute_transformation(source: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the rigid transformation (rotation and translation) that best aligns source to target.
    
    Args:
        source: Nx3 array of source points.
        target: Nx3 array of target points (corresponding points).
    
    Returns:
        Tuple of (3x3 rotation matrix, 3x1 translation vector).
    """
    # Calculate centroids
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    
    # Center the point sets
    centered_source = source - source_centroid
    centered_target = target - target_centroid
    
    # Compute the covariance matrix
    H = centered_source.T @ centered_target
    
    # Use SVD to find the rotation
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure a proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute the translation
    t = target_centroid - R @ source_centroid
    
    return R, t

def _apply_transformation(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Apply a rigid transformation to a set of points.
    
    Args:
        points: Nx3 array of points.
        R: 3x3 rotation matrix.
        t: 3x1 translation vector.
    
    Returns:
        Transformed points.
    """
    return (R @ points.T).T + t

def icp_alignment(
    source_map: np.ndarray,
    target_map: np.ndarray,
    sample_rate: int = 10,
    max_iterations: int = 50,
    convergence_threshold: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align a source heightmap to a target heightmap using Iterative Closest Point (ICP).
    
    Args:
        source_map: Source heightmap to align
        target_map: Target heightmap to align to
        sample_rate: Sample every nth point to reduce computation
        max_iterations: Maximum ICP iterations
        convergence_threshold: Convergence threshold for ICP
        
    Returns:
        Tuple of (aligned_heightmap, transformation_matrix)
    """
    # Convert heightmaps to point clouds
    source_points = _heightmap_to_point_cloud(source_map, sample_rate)
    target_points = _heightmap_to_point_cloud(target_map, sample_rate)
    
    # Initialize transformation
    R = np.eye(3)  # Initial rotation: identity
    t = np.zeros(3)  # Initial translation: zero
    prev_error = float('inf')
    
    # ICP main loop
    for iteration in range(max_iterations):
        # Apply current transformation to source points
        transformed_source = _apply_transformation(source_points, R, t)
        
        # Find closest points in target for each transformed source point
        corresponding_target = _find_closest_points(transformed_source, target_points)
        
        # Compute new transformation
        delta_R, delta_t = _compute_transformation(transformed_source, corresponding_target)
        
        # Update cumulative transformation
        t = delta_t + delta_R @ t
        R = delta_R @ R
        
        # Calculate error
        error = np.mean(np.sum((corresponding_target - transformed_source) ** 2, axis=1))
        
        # Check for convergence
        if abs(prev_error - error) < convergence_threshold:
            logger.info(f"ICP converged after {iteration+1} iterations with error {error:.6f}")
            break
            
        prev_error = error
    
    # Create a 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t
    
    # Convert to OpenCV-compatible 2x3 transformation matrix
    # This simplifies the transformation to a 2D affine transformation
    cv_transform = np.eye(2, 3, dtype=np.float32)
    
    # Extract translation (x, y components)
    cv_transform[0, 2] = t[0]
    cv_transform[1, 2] = t[1]
    
    # Apply the transformation to get the aligned map
    # We use OpenCV's warpAffine for efficiency
    aligned_map = cv2.warpAffine(
        source_map, 
        cv_transform, 
        (target_map.shape[1], target_map.shape[0]),
        flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP, 
        borderMode=cv2.BORDER_REPLICATE
    )
    
    logger.info(f"ICP alignment completed with final error: {error:.6f}")
    return aligned_map, cv_transform