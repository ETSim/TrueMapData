"""
Plotly-based visualization functions for TMD data.
"""

import os
from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go

# Default settings
COLORBAR_LABEL = "Height (µm)"
SCALE_FACTORS = [0.5, 1, 2, 3]  # Z-axis scaling factors for slider


def plot_height_map_with_slider(
    height_map,
    colorbar_label=None,
    html_filename="slider_plot.html",
    partial_range=None,
    scale_factors=None,
):
    """
    Creates a 3D surface plot with a slider to adjust vertical scaling.

    Args:
        height_map: 2D numpy array of height values
        colorbar_label: Label for the color bar (default: "Height (µm)")
        html_filename: Name of the HTML file to save
        partial_range: Optional tuple (row_start, row_end, col_start, col_end) for partial rendering
        scale_factors: List of vertical scale factors for the slider

    Returns:
        Plotly figure object
    """
    if colorbar_label is None:
        colorbar_label = COLORBAR_LABEL
    if scale_factors is None:
        scale_factors = SCALE_FACTORS

    if partial_range is not None:
        height_map = height_map[
            partial_range[0] : partial_range[1], partial_range[2] : partial_range[3]
        ]
        print(
            f"Partial render applied: rows {partial_range[0]}:{partial_range[1]}, cols {partial_range[2]}:{partial_range[3]}"
        )

    zmin = float(height_map.min())
    zmax = float(height_map.max())
    surface = go.Surface(
        z=height_map,
        cmin=zmin,
        cmax=zmax,
        colorscale="Viridis",
        colorbar=dict(title=colorbar_label),
    )

    fig = go.Figure(data=[surface])
    fig.update_layout(
        title="3D Surface Plot",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title=colorbar_label,
            aspectmode="cube",
        ),
        margin=dict(l=65, r=50, b=65, t=90),
    )

    steps = []
    for sf in scale_factors:
        steps.append(
            dict(
                method="relayout",
                args=[{"scene.aspectratio": dict(x=1, y=1, z=sf)}],
                label=f"{sf}x",
            )
        )

    sliders = [dict(active=1, currentvalue={"prefix": "Z-scale: "}, steps=steps, pad={"t": 50})]

    fig.update_layout(sliders=sliders)
    fig.write_html(html_filename, include_plotlyjs="cdn")
    print(f"3D Plot saved to {html_filename}")
    return fig


def plot_2d_heatmap(height_map, colorbar_label=None, html_filename="2d_heatmap.html"):
    """
    Creates a 2D heatmap of the height map.

    Args:
        height_map: 2D numpy array of height values
        colorbar_label: Label for the color bar (default: "Height (µm)")
        html_filename: Name of the HTML file to save

    Returns:
        Plotly figure object
    """
    if colorbar_label is None:
        colorbar_label = COLORBAR_LABEL

    fig = go.Figure(
        data=go.Heatmap(z=height_map, colorscale="Viridis", colorbar=dict(title=colorbar_label))
    )

    fig.update_layout(title="2D Heatmap of Height Map", xaxis_title="X", yaxis_title="Y")

    fig.write_html(html_filename, include_plotlyjs="cdn")
    print(f"2D Heatmap saved to {html_filename}")
    return fig


def plot_x_profile(data, profile_row=None, html_filename="x_profile.html"):
    """
    Extracts an X profile from the height map and plots a 2D line chart.

    Args:
        data: Dictionary containing height_map, width, x_offset, x_length
        profile_row: Row index to extract (default: middle row)
        html_filename: Name of the HTML file to save

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

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_coords, y=x_profile, mode="lines+markers", name="X Profile"))

    fig.update_layout(
        title=f"X Profile (row {profile_row})",
        xaxis_title="X Coordinate",
        yaxis_title=COLORBAR_LABEL,
    )

    fig.write_html(html_filename, include_plotlyjs="cdn")
    print(f"X Profile plot saved to {html_filename}")
    return x_coords, x_profile, fig


def plot_height_map_3d(
    height_map, title="Height Map", filename="height_map.html", colorscale="Viridis"
):
    """
    Creates a 3D surface plot of the height map using Plotly.

    Args:
        height_map: 2D numpy array of height values
        title: Plot title
        filename: Output file name for HTML
        colorscale: Plotly colorscale name

    Returns:
        Path to the saved HTML file
    """
    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(z=height_map, colorscale=colorscale)])

    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Height",
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
        margin=dict(l=65, r=50, b=65, t=90),
    )

    # Save as HTML
    if filename:
        fig.write_html(filename)
        print(f"Saved height map plot as {filename}")

    return filename


def plot_height_map_2d(
    height_map, title="Height Map", filename="height_map_2d.html", colorscale="Viridis"
):
    """
    Creates a 2D heatmap visualization of the height map using Plotly.

    Args:
        height_map: 2D numpy array of height values
        title: Plot title
        filename: Output file name for HTML
        colorscale: Plotly colorscale name

    Returns:
        Path to the saved HTML file
    """
    fig = go.Figure(data=go.Heatmap(z=height_map, colorscale=colorscale))

    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")

    if filename:
        fig.write_html(filename)
        print(f"Saved 2D height map plot as {filename}")

    return filename


def plot_cross_section_plotly(
    x_positions, 
    heights, 
    title="Surface Cross-Section", 
    filename="cross_section.html"
):
    """
    Create an interactive cross-section plot using Plotly.
    
    Args:
        x_positions: Array of x-axis positions for the cross-section
        heights: Array of height values at each position
        title: Title for the plot
        filename: Output filename for the interactive HTML
        
    Returns:
        Plotly figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly is required for this function. Install with: pip install plotly")
        
    fig = go.Figure()
    
    # Add the profile line
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=heights,
            mode="lines",
            name="Surface Profile",
            line=dict(color="blue", width=2)
        )
    )
    
    # Add filled area beneath the profile
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=[0] * len(x_positions),
            mode="lines",
            name="Base",
            line=dict(width=0),
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=heights,
            mode="lines",
            fill="tonexty",
            name="Profile Area",
            line=dict(width=0),
            fillcolor="rgba(0, 0, 255, 0.2)",
            showlegend=False
        )
    )
    
    # Configure layout
    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title="Height",
        hovermode="closest",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    
    # Save to file if filename is provided
    if filename:
        fig.write_html(filename)
        print(f"Interactive cross-section plot saved to {filename}")
    
    return fig
