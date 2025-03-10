"""
Functions for plotting cross-sections of TMD height maps.
"""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Tuple, Optional, List, Union

from tmd.processing import extract_cross_section


def plot_cross_section_matplotlib(height_map: np.ndarray,
                                data_dict: dict,
                                axis: str = 'x',
                                position: Optional[int] = None,
                                start_point: Optional[Tuple[int, int]] = None,
                                end_point: Optional[Tuple[int, int]] = None,
                                title: Optional[str] = None,
                                xlabel: Optional[str] = None,
                                ylabel: str = "Height",
                                filename: Optional[str] = None,
                                show_grid: bool = True,
                                figsize: Tuple[int, int] = (10, 6)) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a cross-section of a height map using Matplotlib.
    
    Args:
        height_map: 2D numpy array of height values
        data_dict: Dictionary containing metadata
        axis: 'x', 'y', or 'custom' for the cross-section direction
        position: Position along the perpendicular axis (row/column index)
        start_point: (row, col) start point for custom cross-section
        end_point: (row, col) end point for custom cross-section
        title: Plot title
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        filename: If provided, save the plot to this file
        show_grid: Whether to show grid lines
        figsize: Figure size (width, height) in inches
        
    Returns:
        Tuple of (figure, axes)
    """
    # Extract the cross-section
    positions, heights = extract_cross_section(
        height_map, data_dict, axis, position, start_point, end_point
    )
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the cross-section
    ax.plot(positions, heights, 'b-', linewidth=1.5)
    ax.scatter(positions[::10], heights[::10], color='r', s=20)  # Add points every 10th element
    
    # Set axis labels
    if xlabel is None:
        if axis.lower() == 'x':
            xlabel = "X Position"
        elif axis.lower() == 'y':
            xlabel = "Y Position"
        else:
            xlabel = "Distance"
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # Set title
    if title is None:
        if axis.lower() == 'x':
            pos_str = f" at row {position}" if position is not None else ""
            title = f"X Cross-Section{pos_str}"
        elif axis.lower() == 'y':
            pos_str = f" at column {position}" if position is not None else ""
            title = f"Y Cross-Section{pos_str}"
        else:
            title = "Custom Cross-Section"
    
    ax.set_title(title)
    
    # Show grid
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add horizontal reference line at zero
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if filename is provided
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Cross-section plot saved to {filename}")
    
    return fig, ax


def plot_cross_section_plotly(height_map: np.ndarray,
                             data_dict: dict,
                             axis: str = 'x',
                             position: Optional[int] = None,
                             start_point: Optional[Tuple[int, int]] = None,
                             end_point: Optional[Tuple[int, int]] = None,
                             title: Optional[str] = None,
                             xlabel: Optional[str] = None,
                             ylabel: str = "Height",
                             html_filename: Optional[str] = None,
                             width: int = 800,
                             height: int = 500) -> go.Figure:
    """
    Plot a cross-section of a height map using Plotly.
    
    Args:
        height_map: 2D numpy array of height values
        data_dict: Dictionary containing metadata
        axis: 'x', 'y', or 'custom' for the cross-section direction
        position: Position along the perpendicular axis (row/column index)
        start_point: (row, col) start point for custom cross-section
        end_point: (row, col) end point for custom cross-section
        title: Plot title
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        html_filename: If provided, save the plot to this HTML file
        width: Plot width in pixels
        height: Plot height in pixels
        
    Returns:
        Plotly Figure object
    """
    # Extract the cross-section
    positions, heights = extract_cross_section(
        height_map, data_dict, axis, position, start_point, end_point
    )
    
    # Create plot
    fig = go.Figure()
    
    # Add the cross-section line
    fig.add_trace(
        go.Scatter(
            x=positions,
            y=heights,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Height Profile'
        )
    )
    
    # Add markers every 10th point
    fig.add_trace(
        go.Scatter(
            x=positions[::10],
            y=heights[::10],
            mode='markers',
            marker=dict(color='red', size=8),
            name='Sample Points'
        )
    )
    
    # Set axis labels
    if xlabel is None:
        if axis.lower() == 'x':
            xlabel = "X Position"
        elif axis.lower() == 'y':
            xlabel = "Y Position"
        else:
            xlabel = "Distance"
    
    # Set title
    if title is None:
        if axis.lower() == 'x':
            pos_str = f" at row {position}" if position is not None else ""
            title = f"X Cross-Section{pos_str}"
        elif axis.lower() == 'y':
            pos_str = f" at column {position}" if position is not None else ""
            title = f"Y Cross-Section{pos_str}"
        else:
            title = "Custom Cross-Section"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        width=width,
        height=height,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    # Add a horizontal reference line at y=0
    fig.add_shape(
        type="line",
        x0=min(positions),
        y0=0,
        x1=max(positions),
        y1=0,
        line=dict(
            color="black",
            width=1,
            dash="solid",
        ),
        opacity=0.3,
        layer="below"
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    # Save if html_filename is provided
    if html_filename:
        fig.write_html(html_filename, include_plotlyjs="cdn")
        print(f"Cross-section plot saved to {html_filename}")
    
    return fig


def plot_multiple_cross_sections(height_map: np.ndarray,
                               data_dict: dict,
                               positions: List[int],
                               axis: str = 'x',
                               title: Optional[str] = None,
                               xlabel: Optional[str] = None,
                               ylabel: str = "Height",
                               html_filename: Optional[str] = None) -> go.Figure:
    """
    Plot multiple cross-sections on the same graph using Plotly.
    
    Args:
        height_map: 2D numpy array of height values
        data_dict: Dictionary containing metadata
        positions: List of positions (rows/columns) for cross-sections
        axis: 'x' or 'y' for the cross-section direction
        title: Plot title
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        html_filename: If provided, save the plot to this HTML file
        
    Returns:
        Plotly Figure object
    """
    # Create plot
    fig = go.Figure()
    
    # Create colormap for different cross-sections
    import matplotlib.cm as cm
    colors = cm.rainbow(np.linspace(0, 1, len(positions)))
    
    for i, pos in enumerate(positions):
        # Extract the cross-section
        pos_values, height_values = extract_cross_section(
            height_map, data_dict, axis, position=pos
        )
        
        # Get the color for this line
        color = f"rgb({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)})"
        
        # Add the cross-section line
        pos_label = f"Row {pos}" if axis.lower() == 'x' else f"Column {pos}"
        fig.add_trace(
            go.Scatter(
                x=pos_values,
                y=height_values,
                mode='lines',
                line=dict(color=color, width=2),
                name=pos_label
            )
        )
    
    # Set axis labels
    if xlabel is None:
        if axis.lower() == 'x':
            xlabel = "X Position"
        else:
            xlabel = "Y Position"
    
    # Set title
    if title is None:
        title = f"Multiple {'X' if axis.lower() == 'x' else 'Y'} Cross-Sections"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(l=65, r=50, b=65, t=90)
    )
    
    # Add a horizontal reference line at y=0
    x_range = fig.data[0].x
    fig.add_shape(
        type="line",
        x0=min(x_range),
        y0=0,
        x1=max(x_range),
        y1=0,
        line=dict(
            color="black",
            width=1,
            dash="solid",
        ),
        opacity=0.3,
        layer="below"
    )
    
    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    # Save if html_filename is provided
    if html_filename:
        fig.write_html(html_filename, include_plotlyjs="cdn")
        print(f"Multiple cross-sections plot saved to {html_filename}")
    
    return fig
