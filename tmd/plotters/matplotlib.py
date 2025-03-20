"""
Matplotlib-based visualization functions for TMD data.
"""

import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Handle potential import issues with mplot3d
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
except ImportError:
    warnings.warn(
        "Could not import Axes3D from mpl_toolkits.mplot3d. "
        "3D plotting functionality may be limited."
    )

# Default settings
COLORBAR_LABEL = "Height (µm)"


def plot_height_map_matplotlib(
    height_map, colorbar_label=None, filename="height_map.png", partial_range=None
):
    """
    Creates a 3D surface plot of the height map using Matplotlib.

    Args:
        height_map: 2D numpy array of height values
        colorbar_label: Label for the color bar (default: "Height (µm)")
        filename: Name of the image file to save
        partial_range: Optional tuple (row_start, row_end, col_start, col_end) for partial rendering

    Returns:
        Matplotlib figure object
    """
    if colorbar_label is None:
        colorbar_label = COLORBAR_LABEL

    if partial_range is not None:
        height_map = height_map[
            partial_range[0] : partial_range[1], partial_range[2] : partial_range[3]
        ]
        print(
            f"Partial render applied: rows {partial_range[0]}:{partial_range[1]}, cols {partial_range[2]}:{partial_range[3]}"
        )

    # Check if 3D plotting is available
    has_3d = False
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        has_3d = True
    except ImportError:
        warnings.warn("3D plotting not available - falling back to 2D contour plot")

    if has_3d:
        # Create 3D surface plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Create the mesh grid
        rows, cols = height_map.shape
        x = np.arange(0, cols, 1)
        y = np.arange(0, rows, 1)
        x, y = np.meshgrid(x, y)

        # Plot the surface
        surf = ax.plot_surface(
            x, y, height_map, cmap="viridis", linewidth=0, antialiased=True, alpha=0.8
        )

        # Add colorbar
        colorbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        colorbar.set_label(colorbar_label)

        # Set labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel(colorbar_label)
        ax.set_title("3D Surface Plot (Matplotlib)")
    else:
        # Create 2D contour plot as fallback
        fig, ax = plt.subplots(figsize=(10, 8))
        contour = ax.contourf(height_map, cmap="viridis", levels=20)
        colorbar = fig.colorbar(contour, ax=ax)
        colorbar.set_label(colorbar_label)
        ax.set_title("Height Map Contour Plot (Fallback)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {filename}")

    return fig


def plot_2d_heatmap_matplotlib(height_map, colorbar_label=None, filename="2d_heatmap.png"):
    """
    Creates a 2D heatmap of the height map using Matplotlib.

    Args:
        height_map: 2D numpy array of height values
        colorbar_label: Label for the color bar (default: "Height (µm)")
        filename: Name of the image file to save

    Returns:
        Matplotlib figure object
    """
    if colorbar_label is None:
        colorbar_label = COLORBAR_LABEL

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(height_map, cmap="viridis", origin="lower")

    # Add colorbar
    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label(colorbar_label)

    # Set labels
    ax.set_title("2D Heatmap (Matplotlib)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"2D Heatmap saved to {filename}")

    return fig


def plot_x_profile_matplotlib(data, profile_row=None, filename="x_profile.png"):
    """
    Extracts an X profile from the height map and plots a 2D line chart using Matplotlib.

    Args:
        data: Dictionary containing height_map, width, x_offset, x_length
        profile_row: Row index to extract (default: middle row)
        filename: Name of the image file to save

    Returns:
        Tuple of (x_coordinates, profile_heights, figure)
    """
    height_map = data["height_map"]
    width = data["width"]

    if profile_row is None:
        profile_row = height_map.shape[0] // 2

    x_coords = np.linspace(data["x_offset"], data["x_offset"] + data["x_length"], num=width)
    x_profile = height_map[profile_row, :]

    print(f"\nX Profile at row {profile_row}:")
    print("X coordinates (first 10):", x_coords[:10])
    print("Heights (first 10):", x_profile[:10])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_coords, x_profile, "b-", linewidth=1)
    ax.scatter(x_coords[::10], x_profile[::10], color="red", s=20)  # Add points every 10th element

    ax.set_title(f"X Profile at Row {profile_row}")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel(COLORBAR_LABEL)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"X Profile plot saved to {filename}")

    return x_coords, x_profile, fig
