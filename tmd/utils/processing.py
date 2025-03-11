"""
Functions for manipulating height maps - cropping, flipping, rotating, thresholding, etc.
"""

import numpy as np
from scipy import ndimage
from typing import Tuple, Optional


def crop_height_map(
    height_map: np.ndarray, region: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Crop a height map to the specified region.

    Args:
        height_map: 2D numpy array of height values
        region: Tuple (row_start, row_end, col_start, col_end) defining the crop region

    Returns:
        Cropped height map as a 2D numpy array
    """
    row_start, row_end, col_start, col_end = region

    # Validate crop region
    rows, cols = height_map.shape
    if not (0 <= row_start < row_end <= rows and 0 <= col_start < col_end <= cols):
        raise ValueError(
            f"Invalid crop region {region} for height map of shape {height_map.shape}"
        )

    return height_map[row_start:row_end, col_start:col_end].copy()


def flip_height_map(height_map: np.ndarray, axis: int) -> np.ndarray:
    """
    Flip a height map along the specified axis.

    Args:
        height_map: 2D numpy array of height values
        axis: 0 for horizontal flip (left-right), 1 for vertical flip (up-down)

    Returns:
        Flipped height map as a 2D numpy array
    """
    if axis not in (0, 1):
        raise ValueError("Axis must be 0 (horizontal) or 1 (vertical)")

    return np.flip(height_map, axis=axis).copy()


def rotate_height_map(
    height_map: np.ndarray,
    angle: float,
    reshape: bool = True,
    interpolation_order: int = 1,
) -> np.ndarray:
    """
    Rotate a height map by the specified angle.

    Args:
        height_map: 2D numpy array of height values
        angle: Rotation angle in degrees (counterclockwise)
        reshape: Whether to reshape the output to contain the entire rotated image
        interpolation_order: The order of the spline interpolation (0-5)
            0: nearest neighbor
            1: bilinear
            2-5: higher-order splines

    Returns:
        Rotated height map as a 2D numpy array
    """
    return ndimage.rotate(
        height_map,
        angle,
        reshape=reshape,
        order=interpolation_order,
        mode="constant",
        cval=0.0,
    )


def threshold_height_map(
    height_map: np.ndarray,
    min_height: Optional[float] = None,
    max_height: Optional[float] = None,
    replacement: Optional[float] = None,
) -> np.ndarray:
    """
    Apply threshold to height values, either clipping or replacing values outside the range.

    Args:
        height_map: 2D numpy array of height values
        min_height: Minimum height threshold (None = no lower threshold)
        max_height: Maximum height threshold (None = no upper threshold)
        replacement: Value to use for points outside the threshold range
                    (None = clip to threshold values)

    Returns:
        Thresholded height map as a 2D numpy array
    """
    result = height_map.copy()

    if min_height is not None:
        if replacement is not None:
            result[result < min_height] = replacement
        else:
            result[result < min_height] = min_height

    if max_height is not None:
        if replacement is not None:
            result[result > max_height] = replacement
        else:
            result[result > max_height] = max_height

    return result


def extract_cross_section(
    height_map: np.ndarray,
    data_dict: dict,
    axis: str = "x",
    position: Optional[int] = None,
    start_point: Optional[Tuple[int, int]] = None,
    end_point: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a cross-section from the height map.

    Args:
        height_map: 2D numpy array of height values
        data_dict: Dictionary containing metadata (width, height, x_length, etc.)
        axis: 'x' for horizontal cross-section, 'y' for vertical, 'custom' for arbitrary line
        position: Position along the perpendicular axis (row/column index)
        start_point: (row, col) start point for custom cross-section
        end_point: (row, col) end point for custom cross-section

    Returns:
        Tuple of (positions, heights) along the cross-section
    """
    rows, cols = height_map.shape

    if axis.lower() == "x":
        # Horizontal cross-section (constant y)
        if position is None:
            position = rows // 2  # Default to middle row
        if position < 0 or position >= rows:
            raise ValueError(
                f"Position {position} out of range for height map with {rows} rows"
            )

        # Extract the cross-section
        heights = height_map[position, :].copy()

        # Create position coordinates
        if "x_offset" in data_dict and "x_length" in data_dict:
            positions = np.linspace(
                data_dict["x_offset"],
                data_dict["x_offset"] + data_dict["x_length"],
                cols,
            )
        else:
            positions = np.arange(cols)

        return positions, heights

    elif axis.lower() == "y":
        # Vertical cross-section (constant x)
        if position is None:
            position = cols // 2  # Default to middle column
        if position < 0 or position >= cols:
            raise ValueError(
                f"Position {position} out of range for height map with {cols} columns"
            )

        # Extract the cross-section
        heights = height_map[:, position].copy()

        # Create position coordinates
        if "y_offset" in data_dict and "y_length" in data_dict:
            positions = np.linspace(
                data_dict["y_offset"],
                data_dict["y_offset"] + data_dict["y_length"],
                rows,
            )
        else:
            positions = np.arange(rows)

        return positions, heights

    elif axis.lower() == "custom":
        # Custom cross-section along arbitrary line
        if start_point is None or end_point is None:
            raise ValueError(
                "Both start_point and end_point must be provided for custom cross-section"
            )

        r0, c0 = start_point
        r1, c1 = end_point

        # Check bounds
        if not (
            0 <= r0 < rows and 0 <= c0 < cols and 0 <= r1 < rows and 0 <= c1 < cols
        ):
            raise ValueError("Start or end point out of bounds")

        # Generate points along the line
        num_points = (
            max(abs(r1 - r0) + 1, abs(c1 - c0) + 1) * 2
        )  # Oversample to avoid aliasing
        rs = np.linspace(r0, r1, num_points)
        cs = np.linspace(c0, c1, num_points)

        # Extract heights using bilinear interpolation
        heights = ndimage.map_coordinates(height_map, [rs, cs], order=1, mode="nearest")

        # Calculate positions as distance along the line
        line_length = np.sqrt((r1 - r0) ** 2 + (c1 - c0) ** 2)
        if "x_length" in data_dict and "y_length" in data_dict:
            # Scale to physical dimensions if available
            pixel_size_x = data_dict["x_length"] / (cols - 1) if cols > 1 else 1.0
            pixel_size_y = data_dict["y_length"] / (rows - 1) if rows > 1 else 1.0
            physical_length = np.sqrt(
                ((c1 - c0) * pixel_size_x) ** 2 + ((r1 - r0) * pixel_size_y) ** 2
            )
            positions = np.linspace(0, physical_length, num_points)
        else:
            positions = np.linspace(0, line_length, num_points)

        return positions, heights

    else:
        raise ValueError(f"Unknown axis: {axis}. Must be 'x', 'y', or 'custom'.")


def extract_profile_at_percentage(
    height_map: np.ndarray,
    data_dict: dict,
    axis: str = "x",
    percentage: float = 50.0,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Extracts a profile cross-section at a given percentage along the X or Y axis.

    Args:
        height_map: 2D numpy array of height values.
        data_dict: Dictionary containing metadata (width, height, x_length, y_length).
        axis: 'x' for horizontal profile, 'y' for vertical profile.
        percentage: The percentage location along the axis (0% = start, 100% = end).
        save_path: Optional file path to save the extracted profile as a .npy file.

    Returns:
        Extracted height profile as a 1D numpy array.
    """
    rows, cols = height_map.shape
    percentage = np.clip(percentage, 0, 100)  # Ensure valid percentage range

    if axis.lower() == "x":
        # Extract row at the given percentage along the Y-axis
        row_idx = int((percentage / 100) * (rows - 1))
        profile = height_map[row_idx, :].copy()

        # Generate X positions based on metadata
        if "x_offset" in data_dict and "x_length" in data_dict:
            positions = np.linspace(
                data_dict["x_offset"],
                data_dict["x_offset"] + data_dict["x_length"],
                cols,
            )
        else:
            positions = np.arange(cols)

    elif axis.lower() == "y":
        # Extract column at the given percentage along the X-axis
        col_idx = int((percentage / 100) * (cols - 1))
        profile = height_map[:, col_idx].copy()

        # Generate Y positions based on metadata
        if "y_offset" in data_dict and "y_length" in data_dict:
            positions = np.linspace(
                data_dict["y_offset"],
                data_dict["y_offset"] + data_dict["y_length"],
                rows,
            )
        else:
            positions = np.arange(rows)

    else:
        raise ValueError(f"Unknown axis: {axis}. Must be 'x' or 'y'.")

    # Save the extracted profile if a path is provided
    if save_path:
        np.save(save_path, profile)
        print(f"Saved extracted profile to {save_path}")

    return profile
