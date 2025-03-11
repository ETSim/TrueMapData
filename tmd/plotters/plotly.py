"""
Plotly-based visualization functions for TMD data.
"""

import numpy as np
import plotly.graph_objects as go
import os
from typing import Optional, Tuplefrom tmd.utils import extract_cross_section

# Import needed function from processing module
from tmd.processing import extract_cross_section
SCALE_FACTORS = [0.5, 1, 2, 3]  # Z-axis scaling factors for slider
# Default settings
COLORBAR_LABEL = "Height (µm)"
SCALE_FACTORS = [0.5, 1, 2, 3]  # Z-axis scaling factors for slider_map_with_slider(


def plot_height_map_with_slider(er_plot.html",
    height_map,
    colorbar_label=None,  scale_factors=None,
    html_filename="slider_plot.html",
    partial_range=None,
    scale_factors=None,    Creates a 3D surface plot with a slider to adjust vertical scaling.
):
    """
    Creates a 3D surface plot with a slider to adjust vertical scaling.
ault: "Height (µm)")
    Args:
        height_map: 2D numpy array of height values, col_end) for partial rendering
        colorbar_label: Label for the color bar (default: "Height (µm)")        scale_factors: List of vertical scale factors for the slider
        html_filename: Name of the HTML file to save
        partial_range: Optional tuple (row_start, row_end, col_start, col_end) for partial rendering
        scale_factors: List of vertical scale factors for the slider Plotly figure object

    Returns:
        Plotly figure objectRBAR_LABEL
    """
    if colorbar_label is None:        scale_factors = SCALE_FACTORS
        colorbar_label = COLORBAR_LABEL
    if scale_factors is None::
        scale_factors = SCALE_FACTORS
   partial_range[0] : partial_range[1], partial_range[2] : partial_range[3]
    if partial_range is not None:
        height_map = height_map[
            partial_range[0] : partial_range[1], partial_range[2] : partial_range[3]   f"Partial render applied: rows {partial_range[0]}:{partial_range[1]}, cols {partial_range[2]}:{partial_range[3]}"
        ]        )
        print(
            f"Partial render applied: rows {partial_range[0]}:{partial_range[1]}, cols {partial_range[2]}:{partial_range[3]}"
        )ap.max())
ace(
    zmin = float(height_map.min())ap,
    zmax = float(height_map.max())
    surface = go.Surface(
        z=height_map,
        cmin=zmin,   colorbar=dict(title=colorbar_label),
        cmax=zmax,    )
        colorscale="Viridis",
        colorbar=dict(title=colorbar_label),ta=[surface])
    )
urface Plot",
    fig = go.Figure(data=[surface])
    fig.update_layout(
        title="3D Surface Plot",
        scene=dict(ar_label,
            xaxis_title="X",  aspectmode="cube",
            yaxis_title="Y",
            zaxis_title=colorbar_label,   margin=dict(l=65, r=50, b=65, t=90),
            aspectmode="cube",    )
        ),
        margin=dict(l=65, r=50, b=65, t=90),
    )actors:
end(
    steps = []
    for sf in scale_factors:
        steps.append(spectratio": dict(x=1, y=1, z=sf)}],
            dict(   label=f"{sf}x",
                method="relayout",   )
                args=[{"scene.aspectratio": dict(x=1, y=1, z=sf)}],        )
                label=f"{sf}x",
            )
        )   dict(active=1, currentvalue={"prefix": "Z-scale: "}, steps=steps, pad={"t": 50})
    ]
    sliders = [
        dict(active=1, currentvalue={"prefix": "Z-scale: "}, steps=steps, pad={"t": 50})
    ]lyjs="cdn")
 Plot saved to {html_filename}")
    fig.update_layout(sliders=sliders)    return fig
    fig.write_html(html_filename, include_plotlyjs="cdn")
    print(f"3D Plot saved to {html_filename}")
    return figt_2d_heatmap(height_map, colorbar_label=None, html_filename="2d_heatmap.html"):

    Creates a 2D heatmap of the height map.
def plot_2d_heatmap(height_map, colorbar_label=None, html_filename="2d_heatmap.html"):
    """
    Creates a 2D heatmap of the height map.
ault: "Height (µm)")
    Args:        html_filename: Name of the HTML file to save
        height_map: 2D numpy array of height values
        colorbar_label: Label for the color bar (default: "Height (µm)")
        html_filename: Name of the HTML file to save Plotly figure object

    Returns:
        Plotly figure object        colorbar_label = COLORBAR_LABEL
    """
    if colorbar_label is None:
        colorbar_label = COLORBAR_LABEL
   z=height_map, colorscale="Viridis", colorbar=dict(title=colorbar_label)
    fig = go.Figure(   )
        data=go.Heatmap(    )
            z=height_map, colorscale="Viridis", colorbar=dict(title=colorbar_label)
        )
    )   title="2D Heatmap of Height Map", xaxis_title="X", yaxis_title="Y"
    )
    fig.update_layout(
        title="2D Heatmap of Height Map", xaxis_title="X", yaxis_title="Y"s="cdn")
    ) Heatmap saved to {html_filename}")
    return fig
    fig.write_html(html_filename, include_plotlyjs="cdn")
    print(f"2D Heatmap saved to {html_filename}")
    return figt_x_profile(data, profile_row=None, html_filename="x_profile.html"):

    Extracts an X profile from the height map and plots a 2D line chart.
def plot_x_profile(data, profile_row=None, html_filename="x_profile.html"):
    """
    Extracts an X profile from the height map and plots a 2D line chart., x_length
middle row)
    Args:        html_filename: Name of the HTML file to save
        data: Dictionary containing height_map, width, x_offset, x_length
        profile_row: Row index to extract (default: middle row)
        html_filename: Name of the HTML file to save Tuple of (x_coordinates, profile_heights, figure)

    Returns:ight_map"]
        Tuple of (x_coordinates, profile_heights, figure)    width = data["width"]
    """
    height_map = data["height_map"]
    width = data["width"]        profile_row = height_map.shape[0] // 2

    if profile_row is None:
        profile_row = height_map.shape[0] // 2   data["x_offset"], data["x_offset"] + data["x_length"], num=width

    x_coords = np.linspace(    x_profile = height_map[profile_row, :]
        data["x_offset"], data["x_offset"] + data["x_length"], num=width
    )
    x_profile = height_map[profile_row, :]:10])
    print("Heights (first 10):", x_profile[:10])
    print(f"\nX Profile at row {profile_row}:")
    print("X coordinates (first 10):", x_coords[:10])e()
    print("Heights (first 10):", x_profile[:10])
   go.Scatter(x=x_coords, y=x_profile, mode="lines+markers", name="X Profile")
    fig = go.Figure()    )
    fig.add_trace(
        go.Scatter(x=x_coords, y=x_profile, mode="lines+markers", name="X Profile")
    )file_row})",

    fig.update_layout(   yaxis_title=COLORBAR_LABEL,
        title=f"X Profile (row {profile_row})",    )
        xaxis_title="X Coordinate",
        yaxis_title=COLORBAR_LABEL,dn")
    ) {html_filename}")
    return x_coords, x_profile, fig
    fig.write_html(html_filename, include_plotlyjs="cdn")
    print(f"X Profile plot saved to {html_filename}")
    return x_coords, x_profile, fig
  height_map, title="Height Map", filename="height_map.html", colorscale="Viridis"

def plot_height_map_3d(
    height_map, title="Height Map", filename="height_map.html", colorscale="Viridis"    Creates a 3D surface plot of the height map using Plotly.
):
    """
    Creates a 3D surface plot of the height map using Plotly.mpy array of height values

    Args:L
        height_map: 2D numpy array of height values        colorscale: Plotly colorscale name
        title: Plot title
        filename: Output file name for HTML
        colorscale: Plotly colorscale name Path to the saved HTML file

    Returns:
        Path to the saved HTML file    fig = go.Figure(data=[go.Surface(z=height_map, colorscale=colorscale)])
    """
    # Create 3D surface plot
    fig = go.Figure(data=[go.Surface(z=height_map, colorscale=colorscale)])t(
,
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",  aspectratio=dict(x=1, y=1, z=0.5),
            yaxis_title="Y",
            zaxis_title="Height",   margin=dict(l=65, r=50, b=65, t=90),
            aspectratio=dict(x=1, y=1, z=0.5),    )
        ),
        margin=dict(l=65, r=50, b=65, t=90),ML
    )

    # Save as HTML        print(f"Saved height map plot as {filename}")
    if filename:
        fig.write_html(filename)    return filename
        print(f"Saved height map plot as {filename}")

    return filename
  height_map, title="Height Map", filename="height_map_2d.html", colorscale="Viridis"

def plot_height_map_2d(
    height_map, title="Height Map", filename="height_map_2d.html", colorscale="Viridis"    Creates a 2D heatmap visualization of the height map using Plotly.
):
    """
    Creates a 2D heatmap visualization of the height map using Plotly.mpy array of height values

    Args:L
        height_map: 2D numpy array of height values        colorscale: Plotly colorscale name
        title: Plot title
        filename: Output file name for HTML
        colorscale: Plotly colorscale name Path to the saved HTML file

    Returns:    fig = go.Figure(data=go.Heatmap(z=height_map, colorscale=colorscale))
        Path to the saved HTML file
    """    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")
    fig = go.Figure(data=go.Heatmap(z=height_map, colorscale=colorscale))

    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")
        print(f"Saved 2D height map plot as {filename}")
    if filename:
        fig.write_html(filename)    return filename
        print(f"Saved 2D height map plot as {filename}")

    return filename
  positions, heights, title="Cross Section", filename="cross_section.html"

def plot_cross_section(
    positions, heights, title="Cross Section", filename="cross_section.html"    Creates a line plot of a height map cross-section using Plotly.
):
    """
    Creates a line plot of a height map cross-section using Plotly.lues
 height values
    Args:
        positions: Array of position values        filename: Output file name for HTML
        heights: Array of height values
        title: Plot title
        filename: Output file name for HTML Path to the saved HTML file

    Returns:    fig = go.Figure()
        Path to the saved HTML file
    """
    fig = go.Figure()   go.Scatter(x=positions, y=heights, mode="lines", name="Height Profile")
    )
    fig.add_trace(
        go.Scatter(x=positions, y=heights, mode="lines", name="Height Profile")
    )   title=title, xaxis_title="Position", yaxis_title="Height", showlegend=True
    )
    fig.update_layout(
        title=title, xaxis_title="Position", yaxis_title="Height", showlegend=True
    )
        print(f"Saved cross section plot as {filename}")
    if filename:
        fig.write_html(filename)    return filename
        print(f"Saved cross section plot as {filename}")

    return filenamee_normal_map_image(normal_map, title="Normal Map", filename="normal_map.png"):

    Saves a normal map as an image using Plotly.
def save_normal_map_image(normal_map, title="Normal Map", filename="normal_map.png"):
    """
    Saves a normal map as an image using Plotly.mpy array of normal vectors or RGB image

    Args:        filename: Output file name for PNG
        normal_map: 3D numpy array of normal vectors or RGB image
        title: Plot title
        filename: Output file name for PNG Path to the saved image file

    Returns:age
        Path to the saved image file
    """l_map.dtype == np.float64:
    # Check if input is a normal map or already an RGB image
    if normal_map.ndim == 3 and normal_map.shape[2] == 3:mg = ((normal_map + 1) / 2 * 255).astype(np.uint8)
        if normal_map.dtype == np.float32 or normal_map.dtype == np.float64:
            # Convert from normal map to RGB image   img = normal_map
            img = ((normal_map + 1) / 2 * 255).astype(np.uint8)
        else:        raise ValueError("Input must be a normal map with shape (height, width, 3)")
            img = normal_map
    else:    fig = go.Figure(go.Image(z=img))
        raise ValueError("Input must be a normal map with shape (height, width, 3)")
    fig.update_layout(title=title, width=img.shape[1], height=img.shape[0])
    fig = go.Figure(go.Image(z=img))

    fig.update_layout(title=title, width=img.shape[1], height=img.shape[0])
        print(f"Saved normal map image as {filename}")
    if filename:
        fig.write_image(filename)    return filename
        print(f"Saved normal map image as {filename}")

    return filenamely(
darray,

def plot_cross_section_plotly(
    height_map: np.ndarray,
    data_dict: dict,e,
    axis: str = "x",t, int]] = None,
    position: Optional[int] = None,
    start_point: Optional[Tuple[int, int]] = None, None,
    end_point: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,tional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Height", = 500,
    html_filename: Optional[str] = None,.Figure:
    width: int = 800,
    height: int = 500,    Plot a cross-section of a height map using Plotly.
) -> go.Figure:
    """
    Plot a cross-section of a height map using Plotly.es

    Args:
        height_map: 2D numpy array of height valuesindex)
        data_dict: Dictionary containing metadatation
        axis: 'x', 'y', or 'custom' for the cross-section directioncol) end point for custom cross-section
        position: Position along the perpendicular axis (row/column index)
        start_point: (row, col) start point for custom cross-section
        end_point: (row, col) end point for custom cross-section
        title: Plot title save the plot to this HTML file
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis        height: Plot height in pixels
        html_filename: If provided, save the plot to this HTML file
        width: Plot width in pixels
        height: Plot height in pixels Plotly Figure object

    Returns:
        Plotly Figure object
    """   height_map, data_dict, axis, position, start_point, end_point
    # Extract the cross-section    )
    positions, heights = extract_cross_section(
        height_map, data_dict, axis, position, start_point, end_point
    )    fig = go.Figure()

    # Create plots-section line
    fig = go.Figure()

    # Add the cross-section lines,
    fig.add_trace(
        go.Scatter(
            x=positions,, width=2),
            y=heights,   name="Height Profile",
            mode="lines",   )
            line=dict(color="blue", width=2),    )
            name="Height Profile",
        )every 10th point
    )

    # Add markers every 10th point],
    fig.add_trace(,
        go.Scatter(
            x=positions[::10],d", size=8),
            y=heights[::10],   name="Sample Points",
            mode="markers",   )
            marker=dict(color="red", size=8),    )
            name="Sample Points",
        )
    )

    # Set axis labels
    if xlabel is None:
        if axis.lower() == "x":label = "Y Position"
            xlabel = "X Position"
        elif axis.lower() == "y":            xlabel = "Distance"
            xlabel = "Y Position"
        else:
            xlabel = "Distance"

    # Set titleposition is not None else ""
    if title is None:tion{pos_str}"
        if axis.lower() == "x":
            pos_str = f" at row {position}" if position is not None else ""if position is not None else ""
            title = f"X Cross-Section{pos_str}"itle = f"Y Cross-Section{pos_str}"
        elif axis.lower() == "y":
            pos_str = f" at column {position}" if position is not None else ""            title = "Custom Cross-Section"
            title = f"Y Cross-Section{pos_str}"
        else:
            title = "Custom Cross-Section"t(

    # Update layout
    fig.update_layout(ylabel,
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,anchor="right", x=0.99),
        height=height,   margin=dict(l=65, r=50, b=65, t=90),
        hovermode="closest",    )
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        margin=dict(l=65, r=50, b=65, t=90),ntal reference line at y=0
    )

    # Add a horizontal reference line at y=0n(positions),
    fig.add_shape(
        type="line",x(positions),
        x0=min(positions),
        y0=0,
        x1=max(positions),lack",
        y1=0,
        line=dict(  dash="solid",
            color="black",
            width=1,
            dash="solid",   layer="below",
        ),    )
        opacity=0.3,
        layer="below",
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")lename is provided
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="LightGrey")

    # Save if html_filename is provided        print(f"Cross-section plot saved to {html_filename}")
    if html_filename:
        fig.write_html(html_filename, include_plotlyjs="cdn")    return fig




    return fig        print(f"Cross-section plot saved to {html_filename}")