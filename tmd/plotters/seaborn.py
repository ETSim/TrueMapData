""".

Seaborn-based visualization functions for TMD data.

This module provides advanced statistical visualizations for height maps and profiles.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    raise ImportError(
        "Seaborn is required for this module. Install with: pip install seaborn"
    )

# Default settings
COLORBAR_LABEL = "Height (µm)"


def plot_height_map_seaborn(
    height_map,
    colorbar_label=None,
    filename="seaborn_height_map.png",
    partial_range=None,
):
    """.

    Creates a heatmap visualization of the height map using Seaborn.

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

    # Set the Seaborn style
    sns.set(style="whitegrid")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create the heatmap
    sns.heatmap(height_map, cmap="viridis", cbar_kws={"label": colorbar_label}, ax=ax)

    # Customize the plot
    ax.set_title("Height Map (Seaborn)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Seaborn height map saved to {filename}")

    return fig


def plot_2d_heatmap_seaborn(
    height_map, colorbar_label=None, filename="seaborn_2d_heatmap.png"
):
    """.

    Creates a detailed 2D heatmap of the height map using Seaborn with additional annotations.

    Args:
        height_map: 2D numpy array of height values
        colorbar_label: Label for the color bar (default: "Height (µm)")
        filename: Name of the image file to save

    Returns:
        Matplotlib figure object
    """
    if colorbar_label is None:
        colorbar_label = COLORBAR_LABEL

    # Set the Seaborn style
    sns.set(style="ticks")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create the heatmap
    sns.heatmap(height_map, cmap="viridis", cbar_kws={"label": colorbar_label}, ax=ax)

    # Add contour lines to show levels
    rows, cols = height_map.shape
    if (
        rows <= 1000 and cols <= 1000
    ):  # Only for smaller maps to avoid excessive computation
        x = np.arange(0, cols, 1)
        y = np.arange(0, rows, 1)
        X, Y = np.meshgrid(x, y)
        levels = np.linspace(height_map.min(), height_map.max(), 10)
        ax.contour(
            X, Y, height_map, levels=levels, colors="white", alpha=0.5, linewidths=0.5
        )

    # Customize the plot
    ax.set_title("Enhanced Height Map (Seaborn)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Enhanced Seaborn heatmap saved to {filename}")

    return fig


def plot_height_distribution(
    height_map,
    title="Height Distribution",
    filename=None,
    bins=50,
    kde=True,
    color="blue",
    fill=True,
):
    """.

    Create a distribution plot of height values.

    Args:
        height_map: 2D numpy array of height values
        title: Plot title
        filename: Output filename (optional)
        bins: Number of histogram bins
        kde: Whether to include KDE curve
        color: Color for the distribution
        fill: Whether to fill under the KDE curve

    Returns:
        Matplotlib figure and axes objects
    """
    if not HAS_SEABORN:
        raise ImportError("Seaborn is required for this function")

    # Flatten the height map
    heights = height_map.flatten()

    # Set up the plot style
    sns.set_style("whitegrid")

    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the distribution
    sns.histplot(heights, bins=bins, kde=kde, color=color, alpha=0.6, fill=fill, ax=ax)

    # Add labels and title
    ax.set_xlabel("Height Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    # Add distribution statistics as text
    stats_text = f"Mean: {heights.mean():.4f}\n"
    stats_text += f"Std Dev: {heights.std():.4f}\n"
    stats_text += f"Min: {heights.min():.4f}\n"
    stats_text += f"Max: {heights.max():.4f}\n"
    stats_text += f"Median: {np.median(heights):.4f}"

    # Add text box with statistics
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=bbox_props,
    )

    # Save the figure if a filename is provided
    if filename:
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved distribution plot to {filename}")

    return fig, ax


def plot_heightmap_heatmap(
    height_map,
    title="Height Map Heatmap",
    filename=None,
    cmap="viridis",
    annot=False,
    robust=True,
):
    """.

    Create a heatmap visualization of a height map using Seaborn.

    Args:
        height_map: 2D numpy array of height values
        title: Plot title
        filename: Output filename (optional)
        cmap: Colormap to use
        annot: Whether to annotate cells with values
        robust: Whether to use robust quantiles for color mapping

    Returns:
        Matplotlib figure and axes objects
    """
    if not HAS_SEABORN:
        raise ImportError("Seaborn is required for this function")

    # For large height maps, downsample to avoid memory issues with annotations
    if annot and (height_map.shape[0] > 50 or height_map.shape[1] > 50):
        # Downsample to at most 50x50
        sample_rate = max(height_map.shape[0] // 50, height_map.shape[1] // 50, 1)
        height_map_display = height_map[::sample_rate, ::sample_rate]
        print(
            f"Downsampling large height map from {height_map.shape} to {height_map_display.shape} for display"
        )
    else:
        height_map_display = height_map

    # Set up the plot
    plt.figure(figsize=(12, 10))

    # Create the heatmap
    ax = sns.heatmap(
        height_map_display,
        cmap=cmap,
        annot=annot,
        fmt=".2f" if annot else None,
        cbar_kws={"label": "Height"},
        robust=robust,
    )

    # Add title
    ax.set_title(title)

    # Remove tick labels if the map is large
    if height_map_display.shape[0] > 20 or height_map_display.shape[1] > 20:
        ax.set_xticks([])
        ax.set_yticks([])

    # Save the figure if a filename is provided
    if filename:
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved heatmap to {filename}")

    return plt.gcf(), ax


def plot_profile_comparison(
    profiles,
    labels=None,
    title="Profile Comparison",
    filename=None,
    palette="husl",
    alpha=0.7,
    fill=False,
):
    """.

    Create a line plot comparing multiple profiles.

    Args:
        profiles: List of 1D arrays representing profiles
        labels: List of strings for legend labels
        title: Plot title
        filename: Output filename (optional)
        palette: Seaborn color palette
        alpha: Transparency of lines
        fill: Whether to fill under the lines

    Returns:
        Matplotlib figure and axes objects
    """
    if not HAS_SEABORN:
        raise ImportError("Seaborn is required for this function")

    # Validate inputs
    if not profiles:
        raise ValueError("No profiles provided")

    if labels is None:
        labels = [f"Profile {i + 1}" for i in range(len(profiles))]

    if len(labels) != len(profiles):
        raise ValueError("Number of labels must match number of profiles")

    # Set the style
    sns.set_style("whitegrid")

    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Get color palette
    colors = sns.color_palette(palette, len(profiles))

    # Plot each profile
    for i, profile in enumerate(profiles):
        x = np.arange(len(profile))
        ax.plot(x, profile, label=labels[i], color=colors[i], alpha=alpha, linewidth=2)

        if fill:
            ax.fill_between(
                x, np.zeros_like(profile), profile, color=colors[i], alpha=0.2
            )

    # Add labels and title
    ax.set_xlabel("Position")
    ax.set_ylabel("Height")
    ax.set_title(title)
    ax.legend()

    # Add grid and set limits
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlim(0, max(len(p) for p in profiles) - 1)

    # Save the figure if a filename is provided
    if filename:
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved profile comparison plot to {filename}")

    return fig, ax


def plot_joint_distribution(
    height_map,
    title="Height and Gradient Joint Distribution",
    filename=None,
    cmap="viridis",
    kind="scatter",
    marginal_kws=None,
):
    """.

    Create a joint distribution plot of height values and their gradients.

    Args:
        height_map: 2D numpy array of height values
        title: Plot title
        filename: Output filename (optional)
        cmap: Colormap to use
        kind: Kind of plot ('scatter', 'kde', 'hex', etc.)
        marginal_kws: Additional keyword args for marginal plots

    Returns:
        Seaborn JointGrid object
    """
    if not HAS_SEABORN:
        raise ImportError("Seaborn is required for this function")

    # Calculate gradient magnitudes using simple central differences
    gradient_y, gradient_x = np.gradient(height_map)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Flatten arrays
    heights = height_map.flatten()
    gradients = gradient_magnitude.flatten()

    # Create a DataFrame for the data
    data = pd.DataFrame({"Height": heights, "Gradient": gradients})

    # Set default marginal_kws based on the kind of plot
    if marginal_kws is None:
        if kind in ["scatter", "hex"]:
            marginal_kws = {"bins": 30}
        elif kind == "kde":
            marginal_kws = {}  # KDE plots don't use 'bins'

    # In case marginal_kws was provided but kind is 'kde', remove 'bins' if it exists
    if kind == "kde" and "bins" in marginal_kws:
        marginal_kws.pop("bins")

    joint_grid = sns.jointplot(
        data=data,
        x="Height",
        y="Gradient",
        kind=kind,
        cmap=cmap,
        marginal_kws=marginal_kws,
        height=8,
    )

    # Add title
    joint_grid.fig.suptitle(title, y=1.02)

    # Save the figure if a filename is provided
    if filename:
        joint_grid.savefig(filename, dpi=300)
        print(f"Saved joint distribution plot to {filename}")

    return joint_grid
