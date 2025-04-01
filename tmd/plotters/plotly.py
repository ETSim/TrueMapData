#!/usr/bin/env python3
"""
Plotly-based visualization classes for TMD data and sequences.

This module provides two classes:
  - PlotlyHeightMapVisualizer: For creating 3D surface plots, 2D heatmaps,
    cross-section plots, and other height map visualizations.
  - PlotlySequenceVisualizer: For creating sequence visualizations including
    animations, slider-based frame displays, and statistical plots.

Both classes implement the BasePlotter and BaseSequencePlotter interfaces.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import base classes
from tmd.plotters.base import BasePlotter, BaseSequencePlotter

# Set up logging
logger = logging.getLogger(__name__)

# Default constants
DEFAULT_COLORBAR_LABEL = "Height (µm)"
DEFAULT_SCALE_FACTORS = [0.5, 1, 2, 3]


class PlotlyHeightMapVisualizer(BasePlotter):
    """
    Provides Plotly-based visualizations for single TMD height maps.

    Implements the BasePlotter interface for compatibility with the factory pattern.
    """
    
    NAME = "plotly"
    DEFAULT_COLORMAP = "Viridis"
    SUPPORTED_MODES = ["2d", "3d", "contour", "profile", "slider"]
    REQUIRED_DEPENDENCIES = ["plotly", "plotly.graph_objects"]
    
    def __init__(self) -> None:
        """Initialize the Plotly plotter and check for dependencies."""
        super().__init__()
        
        # Check for Plotly dependencies
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            self.go = go
            self.pio = pio
            try:
                from plotly.subplots import make_subplots
                self.make_subplots = make_subplots
            except ImportError:
                self.make_subplots = None
                logger.warning("plotly.subplots not available - subplot functionality limited")
        except ImportError:
            raise ImportError("plotly.graph_objects is required for PlotlyHeightMapVisualizer")

    def plot(self, height_map: np.ndarray, **kwargs) -> Any:
        """
        Plot a TMD height map using Plotly visualizations.
        
        Args:
            height_map: 2D numpy array with height data
            **kwargs: Additional options including:
                - mode: Visualization mode ('2d', '3d', 'contour', 'profile', 'slider')
                - title: Plot title
                - colormap: Colormap name (in Plotly this is 'colorscale')
                - width, height: Figure dimensions in pixels
                - profile_row: Row index for profile plots
                - show: Whether to display the figure
                - partial_range: Tuple (row_start, row_end, col_start, col_end) for plotting subset
                
        Returns:
            Plotly figure object
        """
        # Extract parameters with defaults
        mode = kwargs.get("mode", "2d").lower()
        title = kwargs.get("title", "TMD Height Map")
        colorscale = kwargs.get("colormap", self.DEFAULT_COLORMAP)
        width = kwargs.get("width", 800)
        height = kwargs.get("height", 600)
        show = kwargs.get("show", False)
        colorbar_label = kwargs.get("colorbar_label", DEFAULT_COLORBAR_LABEL)
        
        # Apply partial range if specified
        partial_range = kwargs.get("partial_range", None)
        if partial_range is not None:
            height_map = height_map[partial_range[0]:partial_range[1], partial_range[2]:partial_range[3]]
            logger.info(f"Partial render applied: rows {partial_range[0]}:{partial_range[1]}, "
                       f"cols {partial_range[2]}:{partial_range[3]}")
                       
        # Create a filtered copy of kwargs to avoid duplicate parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in 
                         ["mode", "title", "colormap", "colorscale", "width", "height", 
                          "show", "colorbar_label"]}
        
        # Create figure based on mode
        if mode == "3d":
            fig = self._create_3d_surface(height_map, title, colorscale, colorbar_label, **filtered_kwargs)
        elif mode == "contour":
            fig = self._create_contour(height_map, title, colorscale, **filtered_kwargs)
        elif mode == "profile":
            profile_row = kwargs.get("profile_row", height_map.shape[0] // 2)
            # Remove profile_row to prevent duplicate arguments
            if "profile_row" in filtered_kwargs:
                del filtered_kwargs["profile_row"]
            fig = self._create_profile(height_map, profile_row, title, colorbar_label, **filtered_kwargs)
        elif mode == "slider":
            scale_factors = kwargs.get("scale_factors", DEFAULT_SCALE_FACTORS)
            # Remove scale_factors to prevent duplicate arguments
            if "scale_factors" in filtered_kwargs:
                del filtered_kwargs["scale_factors"]
            fig = self._create_slider_viz(height_map, title, colorscale, colorbar_label, scale_factors, **filtered_kwargs)
        else:  # Default to 2D
            fig = self._create_2d_heatmap(height_map, title, colorscale, **filtered_kwargs)
            
        # Set figure dimensions
        fig.update_layout(width=width, height=height)
        
        # Display if requested
        if show:
            fig.show()
            
        return fig

    def plot_3d(self, height_map: np.ndarray, **kwargs) -> Any:
        """
        Create a 3D surface plot of the height map.
        
        Args:
            height_map: 2D numpy array with height data
            **kwargs: Additional options including:
                - title: Plot title
                - colormap: Colormap name
                - z_scale: Z-axis scaling factor
                - width, height: Figure dimensions
                - show: Whether to display the figure
                
        Returns:
            Plotly figure object
        """
        # Extract parameters with defaults
        title = kwargs.get("title", "TMD 3D Visualization")
        colormap = kwargs.get("colormap", self.DEFAULT_COLORMAP)
        z_scale = kwargs.get("z_scale", 1.0)
        width = kwargs.get("width", 800)
        height = kwargs.get("height", 600)
        show = kwargs.get("show", False)
        colorbar_label = kwargs.get("colorbar_label", DEFAULT_COLORBAR_LABEL)
        
        # Create 3D surface visualization
        fig = self._create_3d_surface(
            height_map, 
            title=title, 
            colorscale=colormap, 
            colorbar_label=colorbar_label,
            z_scale=z_scale
        )
        
        # Set figure dimensions
        fig.update_layout(width=width, height=height)
        
        # Display if requested
        if show:
            fig.show()
            
        return fig

    def save(self, plot_obj: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save a Plotly figure to a file.
        
        Args:
            plot_obj: Plotly figure object
            filename: Output filename
            **kwargs: Additional options including:
                - image_export: Bool to export as image instead of HTML
                - format: Image format for export ('png', 'jpeg', 'svg', etc.)
                
        Returns:
            Filename if saved successfully, None otherwise
        """
        try:
            image_export = kwargs.get("image_export", False)
            
            # Create directory if needed
            directory = os.path.dirname(os.path.abspath(filename))
            os.makedirs(directory, exist_ok=True)
            
            if image_export:
                fmt = kwargs.get("format", "png")
                if self.pio is None:
                    logger.error("plotly.io module not available for image export")
                    return None
                self.pio.write_image(plot_obj, filename, format=fmt)
                logger.info(f"Saved plot as image: {filename}")
            else:
                include_plotlyjs = kwargs.get("include_plotlyjs", "cdn")
                plot_obj.write_html(filename, include_plotlyjs=include_plotlyjs)
                logger.info(f"Saved plot as HTML: {filename}")
                
            return filename
        except Exception as e:
            logger.error(f"Error saving figure to {filename}: {e}")
            return None

    def _create_3d_surface(self, height_map: np.ndarray, title: str, 
                          colorscale: str, colorbar_label: str, **kwargs) -> Any:
        """Create a 3D surface plot of the height map."""
        z_scale = kwargs.get("z_scale", 1.0)
        
        # Create surface plot
        fig = self.go.Figure(data=[self.go.Surface(z=height_map, colorscale=colorscale,
                                              colorbar=dict(title=colorbar_label))])
        
        # Update layout with title and axes
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title=colorbar_label,
                aspectratio=dict(x=1, y=1, z=z_scale),
            ),
            margin=dict(l=65, r=50, b=65, t=90),
        )
        
        return fig

    def _create_2d_heatmap(self, height_map: np.ndarray, title: str, 
                          colorscale: str, **kwargs) -> Any:
        """Create a 2D heatmap of the height map."""
        # Create heatmap
        fig = self.go.Figure(data=self.go.Heatmap(z=height_map, colorscale=colorscale))
        
        # Update layout with title and axes
        fig.update_layout(
            title=title,
            xaxis_title="X Position",
            yaxis_title="Y Position"
        )
        
        return fig

    def _create_contour(self, height_map: np.ndarray, title: str, 
                       colorscale: str, **kwargs) -> Any:
        """Create a contour plot of the height map."""
        # Calculate contour levels
        zmin = np.min(height_map)
        zmax = np.max(height_map)
        levels = kwargs.get("levels", 20)
        contour_size = (zmax - zmin) / levels
        
        # Create contours configuration
        contours = dict(
            start=zmin,
            end=zmax,
            size=contour_size,
            showlabels=True,
        )
        
        # Create contour plot
        fig = self.go.Figure(data=self.go.Contour(z=height_map, contours=contours, 
                                              colorscale=colorscale))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="X Position",
            yaxis_title="Y Position"
        )
        
        return fig

    def _create_profile(self, height_map: np.ndarray, profile_row: int, 
                       title: str, colorbar_label: str, **kwargs) -> Any:
        """Create a profile plot along a specific row of the height map."""
        # Check profile row is valid
        if profile_row < 0 or profile_row >= height_map.shape[0]:
            profile_row = height_map.shape[0] // 2
            logger.warning(f"Invalid profile row. Using middle row: {profile_row}")
            
        # Create x coordinates
        width = height_map.shape[1]
        x_length = kwargs.get("x_length", None)
        
        if x_length is not None:
            x_offset = kwargs.get("x_offset", 0)
            x_coords = np.linspace(x_offset, x_offset + x_length, num=width)
            x_label = "X Position (mm)"
        else:
            x_coords = np.arange(width)
            x_label = "X Position (pixels)"
            
        # Get profile data for the specified row
        y_profile = height_map[profile_row, :]
        
        # Generate title if not provided
        if title is None or title == "TMD Height Map":
            title = f"Height Profile at Row {profile_row}"
            
        # Create scatter plot
        fig = self.go.Figure()
        
        # Add profile line
        fig.add_trace(self.go.Scatter(
            x=x_coords, 
            y=y_profile,
            mode="lines+markers" if kwargs.get("show_markers", True) else "lines",
            marker=dict(size=8) if kwargs.get("show_markers", True) else None,
            name="Profile"
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=colorbar_label
        )
        
        # Add grid if requested
        if kwargs.get("show_grid", True):
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
            
        return fig

    def _create_slider_viz(self, height_map: np.ndarray, title: str, 
                          colorscale: str, colorbar_label: str,
                          scale_factors: List[float], **kwargs) -> Any:
        """Create a 3D surface plot with a slider for Z-axis scaling."""
        # Create surface plot
        zmin = float(height_map.min())
        zmax = float(height_map.max())
        
        surface = self.go.Surface(
            z=height_map, 
            cmin=zmin, 
            cmax=zmax, 
            colorscale=colorscale,
            colorbar=dict(title=colorbar_label)
        )
        
        fig = self.go.Figure(data=[surface])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X Position", 
                yaxis_title="Y Position", 
                zaxis_title=colorbar_label, 
                aspectmode="cube"
            ),
            margin=dict(l=65, r=50, b=65, t=90)
        )
        
        # Create slider steps
        steps = [
            dict(
                method="relayout", 
                args=[{"scene.aspectratio": dict(x=1, y=1, z=sf)}],
                label=f"{sf}x"
            ) for sf in scale_factors
        ]
        
        # Add slider
        sliders = [dict(
            active=1,  # Default to second position (usually 1.0)
            currentvalue={"prefix": "Z-scale: "},
            steps=steps,
            pad={"t": 50}
        )]
        
        fig.update_layout(sliders=sliders)
        
        return fig


class PlotlySequenceVisualizer(BaseSequencePlotter):
    """
    Provides Plotly-based visualizations for sequences of TMD height maps.
    
    Implements the BaseSequencePlotter interface for compatibility with the factory pattern.
    """
    
    NAME = "plotly"
    DEFAULT_COLORMAP = "Viridis"
    SUPPORTED_MODES = ["2d", "3d", "animation", "statistics"]
    REQUIRED_DEPENDENCIES = ["plotly", "plotly.graph_objects", "plotly.subplots"]
    
    def __init__(self) -> None:
        """Initialize the Plotly sequence plotter and check for dependencies."""
        super().__init__()
        
        # Check for Plotly dependencies
        try:
            import plotly.graph_objects as go
            import plotly.io as pio
            self.go = go
            self.pio = pio
            try:
                from plotly.subplots import make_subplots
                self.make_subplots = make_subplots
            except ImportError:
                self.make_subplots = None
                logger.warning("plotly.subplots not available - statistical visualization limited")
        except ImportError:
            raise ImportError("plotly.graph_objects is required for PlotlySequenceVisualizer")

    def visualize_sequence(self, frames: List[np.ndarray], **kwargs) -> Any:
        """
        Visualize a sequence of TMD height maps with a slider interface.
        
        Args:
            frames: List of 2D numpy arrays representing the sequence
            **kwargs: Additional options including:
                - n_frames: Number of frames to display (default: all)
                - mode: Visualization mode, either '2d' or '3d' (default: '2d')
                - title: Visualization title
                - colormap: Colormap name
                - width, height: Figure dimensions in pixels
                - timestamps: List of frame timestamps or labels
                - show: Whether to display the figure
                
        Returns:
            Plotly figure with the sequence visualization
        """
        if not frames:
            logger.error("No frames provided for sequence visualization")
            fig = self.go.Figure()
            fig.add_annotation(
                text="No frames to visualize",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
        
        # Extract parameters with defaults
        n_frames = kwargs.get('n_frames', len(frames))
        mode = kwargs.get('mode', '2d').lower()
        colormap = kwargs.get('colormap', self.DEFAULT_COLORMAP)
        width = kwargs.get('width', 1000)
        height = kwargs.get('height', 800)
        title = kwargs.get('title', 'TMD Sequence Visualization')
        show = kwargs.get('show', False)
        
        # Get frame timestamps/labels
        timestamps = kwargs.get('timestamps', [f"Frame {i+1}" for i in range(len(frames))])
        
        # Select frames to display (either all or subsampled)
        if len(frames) > n_frames:
            indices = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
            selected_frames = [frames[i] for i in indices]
            selected_timestamps = [timestamps[i] if i < len(timestamps) else f"Frame {i+1}" for i in indices]
        else:
            selected_frames = frames
            selected_timestamps = timestamps[:len(frames)]
        
        # Create visualization based on mode
        if mode == '3d':
            fig = self._visualize_3d_sequence(
                selected_frames, 
                timestamps=selected_timestamps,
                title=title, 
                colorscale=colormap,
                width=width,
                height=height
            )
        else:  # Default to 2D
            fig = self._visualize_2d_sequence(
                selected_frames, 
                timestamps=selected_timestamps,
                title=title, 
                colorscale=colormap,
                width=width,
                height=height
            )
        
        # Display if requested
        if show:
            fig.show()
            
        return fig

    def create_animation(self, frames: List[np.ndarray], **kwargs) -> Any:
        """
        Create an animation from a sequence of TMD height maps.
        
        Args:
            frames: List of 2D numpy arrays representing the sequence
            **kwargs: Additional options including:
                - fps: Frames per second (default: 2)
                - title: Animation title
                - colormap: Colormap name
                - width, height: Figure dimensions in pixels
                - timestamps: List of frame timestamps or labels
                - mode: Animation mode, either '2d' or '3d' (default: '2d')
                - show: Whether to display the animation
                
        Returns:
            Plotly figure with animation capabilities
        """
        if not frames:
            logger.error("No frames provided for animation")
            fig = self.go.Figure()
            fig.add_annotation(
                text="No frames to animate",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
            
        # Extract parameters with defaults
        fps = kwargs.get('fps', 2)
        colormap = kwargs.get('colormap', self.DEFAULT_COLORMAP)
        width = kwargs.get('width', 1000)
        height = kwargs.get('height', 800)
        title = kwargs.get('title', 'TMD Sequence Animation')
        mode = kwargs.get('mode', '2d').lower()
        show = kwargs.get('show', False)
        
        # Get frame timestamps/labels
        timestamps = kwargs.get('timestamps', [f"Frame {i+1}" for i in range(len(frames))])
        
        # Create animation frames
        plotly_frames = []
        
        if mode == '3d':
            # Initial 3D surface (placeholder)
            fig = self.go.Figure(data=[
                self.go.Surface(
                    z=np.zeros_like(frames[0]),
                    colorscale=colormap,
                    showscale=True
                )
            ])
            
            # Create frames
            for i, frame_data in enumerate(frames):
                frame = self.go.Frame(
                    data=[self.go.Surface(
                        z=frame_data, 
                        colorscale=colormap, 
                        showscale=True
                    )],
                    name=f"frame{i}",
                    layout=self.go.Layout(title_text=f"{title} - {timestamps[i] if i < len(timestamps) else f'Frame {i+1}'}")
                )
                plotly_frames.append(frame)
                
            # Update 3D layout
            fig.update_layout(
                scene=dict(
                    aspectratio=dict(x=1, y=1, z=0.5),
                    xaxis=dict(title='X Position'),
                    yaxis=dict(title='Y Position'),
                    zaxis=dict(title='Height'),
                )
            )
        else:
            # Initial 2D heatmap (placeholder)
            fig = self.go.Figure(data=[
                self.go.Heatmap(
                    z=np.zeros_like(frames[0]),
                    colorscale=colormap,
                    showscale=True
                )
            ])
            
            # Create frames
            for i, frame_data in enumerate(frames):
                frame = self.go.Frame(
                    data=[self.go.Heatmap(
                        z=frame_data, 
                        colorscale=colormap, 
                        showscale=True
                    )],
                    name=f"frame{i}",
                    layout=self.go.Layout(title_text=f"{title} - {timestamps[i] if i < len(timestamps) else f'Frame {i+1}'}")
                )
                plotly_frames.append(frame)
        
        # Add frames to figure
        fig.frames = plotly_frames
        
        # Update layout
        fig.update_layout(
            title=f"{title} - {timestamps[0] if timestamps else 'Frame 1'}",
            width=width,
            height=height,
            updatemenus=[{
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {
                            "frame": {"duration": 1000/fps, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 10},
                "showactive": False,
                "x": 0.1,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top"
            }],
            sliders=[{
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
                    for i in range(len(plotly_frames))
                ]
            }]
        )
        
        # Display if requested
        if show:
            fig.show()
            
        return fig

    def visualize_statistics(self, stats_data: Dict[str, List[float]], **kwargs) -> Any:
        """
        Visualize statistical data from the sequence.
        
        Args:
            stats_data: Dictionary with metric names as keys and lists of values
            **kwargs: Additional options including:
                - title: Plot title
                - width, height: Figure dimensions in pixels
                - x_label, y_label: Axis labels
                - metrics: List of specific metrics to include (default: all available)
                - show: Whether to display the figure
                
        Returns:
            Plotly figure with statistical visualization
        """
        if not stats_data:
            logger.error("No statistical data provided for visualization")
            fig = self.go.Figure()
            fig.add_annotation(
                text="No statistical data to visualize",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
            
        # Check for required plotly.subplots
        if self.make_subplots is None:
            logger.warning("plotly.subplots not available - creating simple plot instead")
            return self._create_simple_stats_plot(stats_data, **kwargs)
            
        # Extract parameters with defaults
        width = kwargs.get('width', 1000)
        height = kwargs.get('height', 600)
        title = kwargs.get('title', 'TMD Statistics')
        show = kwargs.get('show', False)
        
        # Identify metrics to plot (exclude timestamps)
        all_metrics = [m for m in stats_data.keys() if m != 'timestamps']
        requested_metrics = kwargs.get('metrics', all_metrics)
        
        # Filter to available metrics
        metrics = [m for m in requested_metrics if m in stats_data]
        
        if not metrics:
            logger.error("No valid metrics found in the data")
            fig = self.go.Figure()
            fig.add_annotation(
                text="No valid metrics to visualize",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            return fig
            
        # Get x-axis values (timestamps or indices)
        x_values = stats_data.get('timestamps', list(range(len(stats_data[metrics[0]]))))
        
        # Group metrics by type
        summary_metrics = ['mean', 'median', 'min', 'max']
        available_summary = [m for m in summary_metrics if m in metrics]
        variability_metrics = ['std', 'variance', 'range']
        available_variability = [m for m in variability_metrics if m in metrics]
        other_metrics = [m for m in metrics if m not in summary_metrics and m not in variability_metrics]
        
        # Create subplots based on available metrics
        num_plots = sum([
            1 if available_summary else 0,
            1 if available_variability else 0,
            len(other_metrics)
        ])
        
        fig = self.make_subplots(
            rows=num_plots,
            cols=1,
            subplot_titles=[
                "Height Statistics" if available_summary else None,
                "Height Variability" if available_variability else None
            ] + [m.capitalize() for m in other_metrics],
            vertical_spacing=0.1
        )
        
        # Add data traces
        current_row = 1
        
        # Add summary statistics
        if available_summary:
            for metric in available_summary:
                fig.add_trace(
                    self.go.Scatter(
                        x=x_values[:len(stats_data[metric])],
                        y=stats_data[metric],
                        mode='lines+markers',
                        name=metric.capitalize()
                    ),
                    row=current_row,
                    col=1
                )
            current_row += 1
            
        # Add variability statistics
        if available_variability:
            for metric in available_variability:
                fig.add_trace(
                    self.go.Scatter(
                        x=x_values[:len(stats_data[metric])],
                        y=stats_data[metric],
                        mode='lines+markers',
                        name=metric.capitalize(),
                        line=dict(color='red' if metric == 'std' else None)
                    ),
                    row=current_row,
                    col=1
                )
            current_row += 1
            
        # Add other metrics
        for metric in other_metrics:
            fig.add_trace(
                self.go.Scatter(
                    x=x_values[:len(stats_data[metric])],
                    y=stats_data[metric],
                    mode='lines+markers',
                    name=metric.capitalize()
                ),
                row=current_row,
                col=1
            )
            current_row += 1
            
        # Update layout
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            showlegend=True
        )
        
        # Update x-axis labels for the bottom subplot
        fig.update_xaxes(title_text=kwargs.get('x_label', 'Frame'), row=num_plots, col=1)
        
        # Update all y-axis labels
        for i in range(1, num_plots + 1):
            fig.update_yaxes(title_text=kwargs.get('y_label', 'Value'), row=i, col=1)
            
        # Display if requested
        if show:
            fig.show()
            
        return fig

    def save_figure(self, fig: Any, filename: str, **kwargs) -> Optional[str]:
        """
        Save a Plotly figure to a file.
        
        Args:
            fig: Plotly figure object
            filename: Output filename
            **kwargs: Additional options including:
                - image_export: Bool to export as image instead of HTML
                - format: Image format for export ('png', 'jpeg', 'svg', etc.)
                
        Returns:
            Filename if saved successfully, None otherwise
        """
        try:
            image_export = kwargs.get("image_export", False)
            
            # Create directory if needed
            directory = os.path.dirname(os.path.abspath(filename))
            os.makedirs(directory, exist_ok=True)
            
            if image_export:
                fmt = kwargs.get("format", "png")
                if self.pio is None:
                    logger.error("plotly.io module not available for image export")
                    return None
                self.pio.write_image(fig, filename, format=fmt)
                logger.info(f"Saved plot as image: {filename}")
            else:
                include_plotlyjs = kwargs.get("include_plotlyjs", "cdn")
                fig.write_html(filename, include_plotlyjs=include_plotlyjs)
                logger.info(f"Saved plot as HTML: {filename}")
                
            return filename
        except Exception as e:
            logger.error(f"Error saving figure to {filename}: {e}")
            return None

    def _visualize_3d_sequence(self, height_maps: List[np.ndarray],
                             timestamps: Optional[List[str]] = None,
                             title: str = "Sequence 3D Visualization",
                             colorscale: str = "Viridis", width: int = 1000,
                             height: int = 800) -> Any:
        """Create a 3D visualization of sequence frames with a slider."""
        if not height_maps:
            return self.go.Figure()
            
        if timestamps is None or len(timestamps) != len(height_maps):
            timestamps = [f"Frame {i+1}" for i in range(len(height_maps))]
        # Create initial figure
        fig = self.go.Figure()
        fig.add_trace(self.go.Surface(z=height_maps[0], colorscale=colorscale))
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X Position",
                yaxis_title="Y Position",
                zaxis_title="Height",
                aspectmode="cube"
            ),
            width=width,
            height=height
        )
        # Create frames for each height map
        for i, frame in enumerate(height_maps):
            fig.add_frame(
                data=[self.go.Surface(z=frame, colorscale=colorscale)],
                name=f"frame{i}",
                layout=self.go.Layout(title_text=f"{title} - {timestamps[i]}")
            )
        # Add slider
        fig.update_layout(
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "right"
                },
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
                    for i in range(len(height_maps))
                ]
            }]
        )
        return fig
    
    def _visualize_2d_sequence(self, height_maps: List[np.ndarray],
                                timestamps: Optional[List[str]] = None,
                                title: str = "Sequence 2D Visualization",
                                colorscale: str = "Viridis", width: int = 1000,
                                height: int = 800) -> Any:
            """Create a 2D visualization of sequence frames with a slider."""
            if not height_maps:
                return self.go.Figure()
                
            if timestamps is None or len(timestamps) != len(height_maps):
                timestamps = [f"Frame {i+1}" for i in range(len(height_maps))]
            # Create initial figure
            fig = self.go.Figure()
            fig.add_trace(self.go.Heatmap(z=height_maps[0], colorscale=colorscale))
            fig.update_layout(
                title=title,
                xaxis_title="X Position",
                yaxis_title="Y Position",
                width=width,
                height=height
            )
            # Create frames for each height map
            for i, frame in enumerate(height_maps):
                fig.add_frame(
                    data=[self.go.Heatmap(z=frame, colorscale=colorscale)],
                    name=f"frame{i}",
                    layout=self.go.Layout(title_text=f"{title} - {timestamps[i]}")
                )
            # Add slider
            fig.update_layout(
                sliders=[{
                    "active": 0,
                    "yanchor": "top",
                    "xanchor": "left",
                    "currentvalue": {
                        "prefix": "Frame: ",
                        "visible": True,
                        "xanchor": "right"
                    },
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
                        for i in range(len(height_maps))
                    ]
                }]
            )
            return fig