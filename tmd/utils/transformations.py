"""

Transformation utilities for TMD sequences.

This module provides functions to apply various transformations to heightmaps
in a TMD sequence, including translation, rotation, scaling, and registration.
"""

import numpy as np
from typing import Dict, Any, Tuple

def apply_translation(height_map: np.ndarray, translation: Tuple[float, float, float]) -> np.ndarray:
    """.

    Apply translation to a heightmap.

    Args:
        height_map: Input heightmap.
        translation: (tx, ty, tz) translation vector, where tx and ty are relative shifts (normalized to width/height),
                     and tz is an absolute height offset.
        
    Returns:
        Translated heightmap.
    """
    tx, ty, tz = translation
    result = height_map.copy()

    # Apply vertical (Z) translation
    if tz != 0:
        result += tz

    # Horizontal (X/Y) translation by shifting the array.
    if tx != 0 or ty != 0:
        rows, cols = height_map.shape
        # Convert normalized translation to pixel shifts.
        shift_x = int(round(tx * cols))
        shift_y = int(round(ty * rows))
        if shift_x != 0:
            result = np.roll(result, shift_x, axis=1)
        if shift_y != 0:
            result = np.roll(result, shift_y, axis=0)

    return result

def apply_rotation(height_map: np.ndarray, rotation: Tuple[float, float, float]) -> np.ndarray:
    """.

    Apply rotation to a heightmap.

    Args:
        height_map: Input heightmap.
        rotation: (rx, ry, rz) rotation angles in degrees. rz is the in-plane (Z-axis) rotation,
                  while rx and ry require a simplified 3D transformation.
    
    Returns:
        Rotated heightmap.
    """
    rx, ry, rz = rotation
    result = height_map.copy()

    # No significant rotation: return original.
    if abs(rx) < 1e-5 and abs(ry) < 1e-5 and abs(rz) < 1e-5:
        return result

    # Z-axis rotation using scipy.ndimage.rotate (angle in degrees).
    if abs(rz) >= 1e-5:
        from scipy.ndimage import rotate
        result = rotate(result, rz, reshape=False, mode='nearest')

    # X and Y rotations: perform a simplified 3D rotation and interpolate back to 2D.
    if abs(rx) >= 1e-5 or abs(ry) >= 1e-5:
        # Convert angles to radians.
        rx_rad = np.radians(rx)
        ry_rad = np.radians(ry)
        # Rotation matrix around X-axis.
        rotation_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx_rad), -np.sin(rx_rad)],
            [0, np.sin(rx_rad), np.cos(rx_rad)]
        ])
        # Rotation matrix around Y-axis.
        rotation_y = np.array([
            [np.cos(ry_rad), 0, np.sin(ry_rad)],
            [0, 1, 0],
            [-np.sin(ry_rad), 0, np.cos(ry_rad)]
        ])
        # Combined rotation: first X then Y.
        rotation_matrix = rotation_y @ rotation_x

        rows, cols = result.shape
        # Create a grid of coordinates.
        y_coords, x_coords = np.mgrid[0:rows, 0:cols]
        z_coords = result
        # Stack coordinates to form (N, 3) array.
        points = np.stack([x_coords.flatten(), y_coords.flatten(), z_coords.flatten()], axis=-1)
        # Apply the combined rotation.
        rotated_points = points @ rotation_matrix.T

        # Interpolate rotated height values back onto a regular grid.
        from scipy.interpolate import griddata
        grid_x, grid_y = np.mgrid[0:rows, 0:cols]
        rotated_z = griddata(
            points=(rotated_points[:, 1], rotated_points[:, 0]),
            values=rotated_points[:, 2],
            xi=(grid_x, grid_y),
            method='linear',
            fill_value=np.min(result)
        )
        result = rotated_z

    return result

def apply_scaling(height_map: np.ndarray, scaling: Tuple[float, float, float]) -> np.ndarray:
    """.

    Apply scaling to a heightmap.

    Args:
        height_map: Input heightmap.
        scaling: (sx, sy, sz) scaling factors. sx and sy scale the horizontal dimensions,
                 and sz scales the height values.
    
    Returns:
        Scaled heightmap.
    """
    sx, sy, sz = scaling
    result = height_map.copy()

    # Scale height values (Z-axis).
    if sz != 1.0:
        result *= sz

    # Scale horizontal dimensions using image resizing.
    if sx != 1.0 or sy != 1.0:
        rows, cols = result.shape
        new_rows = max(int(round(rows * sy)), 1) if sy > 0 else rows
        new_cols = max(int(round(cols * sx)), 1) if sx > 0 else cols

        try:
            import cv2
            result = cv2.resize(result, (new_cols, new_rows), interpolation=cv2.INTER_CUBIC)
        except ImportError:
            from scipy.ndimage import zoom
            zoom_factors = (sy, sx)
            result = zoom(result, zoom_factors, order=3)

    return result

def register_heightmaps(
    reference: np.ndarray, 
    target: np.ndarray, 
    method: str = 'phase_correlation'
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """.

    Register a target heightmap to a reference heightmap.

    Args:
        reference: The reference heightmap.
        target: The target heightmap to be aligned.
        method: Registration method ('phase_correlation', 'feature_matching', or 'icp').

    Returns:
        A tuple of:
            - The registered heightmap.
            - A dictionary containing the transformation parameters.
    """
    # Default transformation parameters.
    transformation: Dict[str, Any] = {
        'translation': [0.0, 0.0, 0.0],
        'rotation': [0.0, 0.0, 0.0],
        'scaling': [1.0, 1.0, 1.0]
    }

    # Ensure target matches the reference dimensions.
    if reference.shape != target.shape:
        try:
            import cv2
            target = cv2.resize(target, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)
        except ImportError:
            from scipy.ndimage import zoom
            zoom_x = reference.shape[1] / target.shape[1]
            zoom_y = reference.shape[0] / target.shape[0]
            target = zoom(target, (zoom_y, zoom_x), order=3)

    if method == 'phase_correlation':
        try:
            import cv2
            # Convert images to float32.
            ref_float = reference.astype(np.float32)
            target_float = target.astype(np.float32)
            # Use phase correlation to determine the shift.
            shifts, response = cv2.phaseCorrelate(ref_float, target_float)
            rows, cols = target.shape
            # Construct the affine transformation matrix.
            M = np.float32([[1, 0, shifts[0]], [0, 1, shifts[1]]])
            registered = cv2.warpAffine(target_float, M, (cols, rows))
            # Update transformation parameters (normalized).
            transformation['translation'][0] = shifts[0] / cols
            transformation['translation'][1] = shifts[1] / rows
            transformation['translation'][2] = 0.0
            return registered, transformation
        except ImportError:
            # Fallback: use FFT-based correlation with scipy.
            from scipy import signal
            correlation = signal.correlate2d(reference, target, mode='same')
            y, x = np.unravel_index(np.argmax(correlation), correlation.shape)
            y_shift = y - reference.shape[0] // 2
            x_shift = x - reference.shape[1] // 2
            registered = np.roll(np.roll(target, y_shift, axis=0), x_shift, axis=1)
            transformation['translation'][0] = x_shift / reference.shape[1]
            transformation['translation'][1] = y_shift / reference.shape[0]
            transformation['translation'][2] = 0.0
            return registered, transformation

    elif method == 'feature_matching':
        try:
            import cv2
            # Normalize and convert to uint8 for feature detection.
            ref_uint8 = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            target_uint8 = cv2.normalize(target, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            # Initialize ORB detector.
            orb = cv2.ORB_create()
            kp1, des1 = orb.detectAndCompute(ref_uint8, None)
            kp2, des2 = orb.detectAndCompute(target_uint8, None)
            if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
                return target, transformation
            # Match features.
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda m: m.distance)
            good_matches = matches[:min(50, len(matches))]
            # Extract keypoints.
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            # Estimate homography.
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if H is None:
                return target, transformation
            rows, cols = reference.shape
            registered = cv2.warpPerspective(target.astype(np.float32), H, (cols, rows))
            # Simplified extraction of translation from the homography using the center.
            center = np.array([[cols/2, rows/2, 1]]).T
            transformed = H @ center
            transformed /= transformed[2]
            tx = (transformed[0] - cols/2)[0] / cols
            ty = (transformed[1] - rows/2)[0] / rows
            transformation['translation'][0] = tx
            transformation['translation'][1] = ty
            transformation['translation'][2] = 0.0
            return registered, transformation
        except ImportError:
            # Fallback to phase correlation.
            return register_heightmaps(reference, target, method='phase_correlation')

    elif method == 'icp':
        # Placeholder for ICP (Iterative Closest Point).
        # A proper ICP implementation would typically rely on a 3D point cloud library (e.g., Open3D).
        # For now, we fallback to phase correlation.
        return register_heightmaps(reference, target, method='phase_correlation')

    else:
        # Unknown registration method: return the original target.
        return target, transformation