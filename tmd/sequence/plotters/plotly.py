""".

Plotly-based visualizations for TMD sequences.

This module provides 2D and 3D visualization capabilities for TMD sequences
using the Plotly library.
"""

import logging
import numpy as np
from typing import Optional, List, Dict, Any, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Add PlotlyPlotter class that the sequence module tries to import
class PlotlyPlotter:
    """Plotter implementation using Plotly for TMD sequences.."""
    
    def __init__(self):
        """Initialize the Plotly plotter.."""
        self._check_dependencies()
    
    def _check_dependencies(self) -> None:
        """Check if required dependencies are available.."""
        try:
            import plotly
            logger.debug(f"Using Plotly version {plotly.__version__}")
        except ImportError:
            logger.warning("Plotly not installed. Install with: pip install plotly")
    
    def visualize_sequence(
        self,
        height_maps: List[np.ndarray],
        timestamps: List[str],
        view_type: str = '3d',
        **kwargs
    ) -> go.Figure:
        """.

        Visualize a sequence of height maps.
        
        Args:
            height_maps: List of height map arrays
            timestamps: List of timestamp strings for each frame
            view_type: Type of visualization ('3d' or '2d')
            **kwargs: Additional visualization options
            
        Returns:
            Plotly Figure object
        """
        if view_type == '3d':
            return visualize_sequence_3d(height_maps, timestamps, **kwargs)
        else:
            return visualize_sequence_2d(height_maps, timestamps, **kwargs)
    
    # Rename this method from plot_statistics to visualize_statistics to match what's called in sequence.py
    def visualize_statistics(
        self, 
        stats: List[Dict[str, float]],
        timestamps: List[str],
        **kwargs
    ) -> go.Figure:
        """.

        Visualize statistics for a sequence of height maps.
        
        Args:
            stats: List of statistics dictionaries for each frame
            timestamps: List of timestamp strings for each frame
            **kwargs: Additional plotting options
            
        Returns:
            Plotly Figure object
        """
        # Extract height maps data from stats if provided directly
        if 'height_maps' in kwargs:
            height_maps = kwargs.pop('height_maps')
            return visualize_sequence_stats(height_maps, timestamps, **kwargs)
        
        # Otherwise, create a figure directly from the statistics
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=["Height Statistics", "Height Variability"],
                           vertical_spacing=0.2)
        
        # Extract statistics
        mean_values = [s.get('mean', 0) for s in stats]
        min_values = [s.get('min', 0) for s in stats]
        max_values = [s.get('max', 0) for s in stats]
        std_values = [s.get('std', 0) for s in stats]
        
        # Add traces for height statistics
        fig.add_trace(
            go.Scatter(x=timestamps, y=mean_values, mode='lines+markers', name='Mean'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=min_values, mode='lines+markers', name='Min'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=max_values, mode='lines+markers', name='Max'),
            row=1, col=1
        )
        
        # Add trace for standard deviation
        fig.add_trace(
            go.Scatter(x=timestamps, y=std_values, mode='lines+markers', 
                      name='Std Dev', line=dict(color='red')),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=kwargs.get('title', "Sequence Statistics"),
            width=kwargs.get('width', 1000),
            height=kwargs.get('height', 600),
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Frame", row=2, col=1)
        fig.update_yaxes(title_text="Height Value", row=1, col=1)
        fig.update_yaxes(title_text="Standard Deviation", row=2, col=1)
        
        # Show or save
        if 'filename' in kwargs:
            fig.write_html(kwargs['filename'])
            logger.info(f"Statistics visualization saved to {kwargs['filename']}")
        
        if kwargs.get('show', False):
            fig.show()
        
        return fig
    
    def create_animation(
        self,
        frames_data: List[np.ndarray],
        timestamps: List[str],
        surface_type: str = '3d',
        **kwargs
    ) -> go.Figure:
        """.

        Create an animation of height maps.
        
        Args:
            frames_data: List of height map arrays
            timestamps: List of timestamp strings for each frame
            surface_type: Type of surface to show ('3d' or '2d')
            **kwargs: Additional animation options
            
        Returns:
            Plotly Figure object
        """
        return create_sequence_animation(frames_data, timestamps, surface_type=surface_type, **kwargs)

def visualize_sequence_3d(
    height_maps: List[np.ndarray],
    timestamps: Optional[List[str]] = None,
    title: str = "Sequence 3D Visualization",
    colorscale: str = "Viridis",
    width: int = 1000,
    height: int = 800,
    show: bool = True,
    filename: Optional[str] = None
) -> go.Figure:
    """.

    Create a 3D visualization of sequence frames with a slider.
    
    Args:
        height_maps: List of height map arrays to visualize
        timestamps: Optional list of timestamp strings for each frame
        title: Plot title
        colorscale: Plotly colorscale name
        width: Figure width in pixels
        height: Figure height in pixels
        show: Whether to show the figure
        filename: Optional filename to save the figure
        
    Returns:
        Plotly Figure object
    """
    if not height_maps:
        logger.warning("No height maps provided to visualize")
        return go.Figure()
    
    # Use provided timestamps or generate default ones
    if timestamps is None or len(timestamps) != len(height_maps):
        timestamps = [f"Frame {i+1}" for i in range(len(height_maps))]
    
    # Create figure
    fig = go.Figure()
    
    # Sample and downsample heightmaps for performance
    for i, height_map in enumerate(height_maps):
        rows, cols = height_map.shape
        max_points = 10000
        if rows * cols > max_points:
            downsample_factor = int(np.sqrt((rows * cols) / max_points))
            height_map = height_map[::downsample_factor, ::downsample_factor]
            rows, cols = height_map.shape
        
        x = np.linspace(0, 1, cols)
        y = np.linspace(0, 1, rows)
        
        # Create surface plot
        surface = go.Surface(
            z=height_map,
            x=x,
            y=y,
            colorscale=colorscale,
            showscale=True,
            visible=(i == 0)  # Only first frame visible by default
        )
        
        fig.add_trace(surface)
    
    # Create slider steps
    steps = []
    for i in range(len(height_maps)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(height_maps)},
                {"title": f"{title} - {timestamps[i]}"}
            ],
            label=f"Frame {i+1}"
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)
    
    # Create slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Displaying: "},
        pad={"t": 50},
        steps=steps
    )]
    
    # Update layout
    fig.update_layout(
        title=f"{title} - {timestamps[0]}",
        width=width,
        height=height,
        scene=dict(
            aspectratio=dict(x=1, y=1, z=0.5),
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Height'),
        ),
        sliders=sliders
    )
    
    # Show or save
    if filename:
        fig.write_html(filename)
        logger.info(f"3D visualization saved to {filename}")
    
    if show:
        fig.show()
    
    return fig

def visualize_sequence_2d(
    height_maps: List[np.ndarray],
    timestamps: Optional[List[str]] = None,
    title: str = "Sequence 2D Visualization",
    colorscale: str = "Viridis",
    width: int = 1000,
    height: int = 800,
    show: bool = True,
    filename: Optional[str] = None
) -> go.Figure:
    """.

    Create a 2D visualization of sequence frames with a slider.
    
    Args:
        height_maps: List of height map arrays to visualize
        timestamps: Optional list of timestamp strings for each frame
        title: Plot title
        colorscale: Plotly colorscale name
        width: Figure width in pixels
        height: Figure height in pixels
        show: Whether to show the figure
        filename: Optional filename to save the figure
        
    Returns:
        Plotly Figure object
    """
    if not height_maps:
        logger.warning("No height maps provided to visualize")
        return go.Figure()
    
    # Use provided timestamps or generate default ones
    if timestamps is None or len(timestamps) != len(height_maps):
        timestamps = [f"Frame {i+1}" for i in range(len(height_maps))]
    
    # Create figure
    fig = go.Figure()
    
    # Add heatmap for each frame
    for i, height_map in enumerate(height_maps):
        heatmap = go.Heatmap(
            z=height_map,
            colorscale=colorscale,
            showscale=True,
            visible=(i == 0)  # Only first frame visible by default
        )
        
        fig.add_trace(heatmap)
    
    # Create slider steps
    steps = []
    for i in range(len(height_maps)):
        step = dict(
            method="update",
            args=[
                {"visible": [False] * len(height_maps)},
                {"title": f"{title} - {timestamps[i]}"}
            ],
            label=f"Frame {i+1}"
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)
    
    # Create slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Displaying: "},
        pad={"t": 50},
        steps=steps
    )]
    
    # Update layout
    fig.update_layout(
        title=f"{title} - {timestamps[0]}",
        width=width,
        height=height,
        sliders=sliders
    )
    
    # Show or save
    if filename:
        fig.write_html(filename)
        logger.info(f"2D visualization saved to {filename}")
    
    if show:
        fig.show()
    
    return fig

def visualize_sequence_stats(
    height_maps: List[np.ndarray],
    timestamps: Optional[List[str]] = None,
    title: str = "Sequence Statistics",
    width: int = 1000,
    height: int = 600,
    show: bool = True,
    filename: Optional[str] = None
) -> go.Figure:
    """.

    Create statistical visualization of sequence frames.
    
    Args:
        height_maps: List of height map arrays to visualize
        timestamps: Optional list of timestamp strings for each frame
        title: Plot title
        width: Figure width in pixels
        height: Figure height in pixels
        show: Whether to show the figure
        filename: Optional filename to save the figure
        
    Returns:
        Plotly Figure object
    """
    if not height_maps:
        logger.warning("No height maps provided for statistics")
        return go.Figure()
    
    # Use provided timestamps or generate default ones
    if timestamps is None or len(timestamps) != len(height_maps):
        timestamps = [f"Frame {i+1}" for i in range(len(height_maps))]
    
    # Calculate statistics for each frame
    mean_values = [np.mean(hm) for hm in height_maps]
    median_values = [np.median(hm) for hm in height_maps]
    min_values = [np.min(hm) for hm in height_maps]
    max_values = [np.max(hm) for hm in height_maps]
    std_values = [np.std(hm) for hm in height_maps]
    
    # Create figure with subplots
    fig = make_subplots(rows=2, cols=1, 
                       subplot_titles=["Height Statistics", "Height Variability"],
                       vertical_spacing=0.2)
    
    # Add traces for height statistics
    fig.add_trace(
        go.Scatter(x=timestamps, y=mean_values, mode='lines+markers', name='Mean'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=timestamps, y=median_values, mode='lines+markers', name='Median'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=timestamps, y=min_values, mode='lines+markers', name='Min'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=timestamps, y=max_values, mode='lines+markers', name='Max'),
        row=1, col=1
    )
    
    # Add trace for standard deviation
    fig.add_trace(
        go.Scatter(x=timestamps, y=std_values, mode='lines+markers', 
                  name='Std Dev', line=dict(color='red')),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Frame", row=2, col=1)
    fig.update_yaxes(title_text="Height Value", row=1, col=1)
    fig.update_yaxes(title_text="Standard Deviation", row=2, col=1)
    
    # Show or save
    if filename:
        fig.write_html(filename)
        logger.info(f"Statistics visualization saved to {filename}")
    
    if show:
        fig.show()
    
    return fig

def create_sequence_animation(
    height_maps: List[np.ndarray],
    timestamps: Optional[List[str]] = None,
    title: str = "Sequence Animation",
    colorscale: str = "Viridis",
    surface_type: str = "3d",
    width: int = 1000,
    height: int = 800,
    fps: int = 2,
    show: bool = True,
    filename: Optional[str] = None
) -> go.Figure:
    """.

    Create an animation of sequence frames.
    
    Args:
        height_maps: List of height map arrays to animate
        timestamps: Optional list of timestamp strings for each frame
        title: Animation title
        colorscale: Plotly colorscale name
        surface_type: Type of surface to show ('3d' or '2d')
        width: Figure width in pixels
        height: Figure height in pixels
        fps: Frames per second for animation
        show: Whether to show the figure
        filename: Optional filename to save the figure
        
    Returns:
        Plotly Figure object
    """
    if not height_maps:
        logger.warning("No height maps provided for animation")
        return go.Figure()
    
    # Use provided timestamps or generate default ones
    if timestamps is None or len(timestamps) != len(height_maps):
        timestamps = [f"Frame {i+1}" for i in range(len(height_maps))]
    
    # Create frames
    frames = []
    
    if surface_type == '3d':
        # Prepare base figure with an empty 3D surface
        fig = go.Figure(data=[go.Surface(
            z=np.zeros_like(height_maps[0]),
            colorscale=colorscale,
            showscale=True
        )])
        
        # Create frames for animation
        for i, height_map in enumerate(height_maps):
            frame = go.Frame(
                data=[go.Surface(
                    z=height_map,
                    colorscale=colorscale,
                    showscale=True
                )],
                name=f"frame{i}",
                layout=go.Layout(title_text=f"{title} - {timestamps[i]}")
            )
            frames.append(frame)
        
        # Update layout for 3D
        fig.update_layout(
            scene=dict(
                aspectratio=dict(x=1, y=1, z=0.5),
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Height'),
            )
        )
    else:
        # Prepare base figure with an empty heatmap
        fig = go.Figure(data=[go.Heatmap(
            z=np.zeros_like(height_maps[0]),
            colorscale=colorscale,
            showscale=True
        )])
        
        # Create frames for animation
        for i, height_map in enumerate(height_maps):
            frame = go.Frame(
                data=[go.Heatmap(
                    z=height_map,
                    colorscale=colorscale,
                    showscale=True
                )],
                name=f"frame{i}",
                layout=go.Layout(title_text=f"{title} - {timestamps[i]}")
            )
            frames.append(frame)
    
    # Add frames to figure
    fig.frames = frames
    
    # Add animation controls
    fig.update_layout(
        title=f"{title} - {timestamps[0]}",
        width=width,
        height=height,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None, 
                            {
                                "frame": {"duration": 1000/fps, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0}
                            }
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None], 
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ],
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 0},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f"frame{i}"],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ],
                        "label": str(i+1),
                        "method": "animate"
                    }
                    for i in range(len(frames))
                ]
            }
        ]
    )
    
    # Show or save
    if filename:
        fig.write_html(filename)
        logger.info(f"Animation saved to {filename}")
    
    if show:
        fig.show()
    
    return fig
