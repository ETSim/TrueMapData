""".

Plotly-based visualization functions for TMD data.
"""

import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import plotly.graph_objects as go

# Set up logging
logger = logging.getLogger(__name__)

# Default settings
COLORBAR_LABEL = "Height (µm)"
SCALE_FACTORS = [0.5, 1, 2, 3]  # Z-axis scaling factors for slider

# Check if plotly is available
try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not installed. Install with 'pip install plotly'.")


def plot_height_map_with_slider(
    height_map,
    colorbar_label=None,
    html_filename="slider_plot.html",
    partial_range=None,
    scale_factors=None,
):
    """.

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

    sliders = [
        dict(active=1, currentvalue={"prefix": "Z-scale: "}, steps=steps, pad={"t": 50})
    ]

    fig.update_layout(sliders=sliders)

    if html_filename:
        fig.write_html(html_filename, include_plotlyjs="cdn")
        print(f"3D Plot saved to {html_filename}")

    return fig


def plot_2d_heatmap(height_map, colorbar_label=None, html_filename="2d_heatmap.html"):
    """.

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
        data=go.Heatmap(
            z=height_map, colorscale="Viridis", colorbar=dict(title=colorbar_label)
        )
    )

    fig.update_layout(
        title="2D Heatmap of Height Map", xaxis_title="X", yaxis_title="Y"
    )

    if html_filename:
        fig.write_html(html_filename, include_plotlyjs="cdn")
        print(f"2D Heatmap saved to {html_filename}")

    return fig


def plot_x_profile(data, profile_row=None, html_filename="x_profile.html"):
    """.

    Extracts an X profile from the height map and plots a 2D line chart.

    Args:
        data: Dictionary containing height_map, width, x_offset, x_length
        profile_row: Row index to extract (default: middle row)
        html_filename: Name of the HTML file to save

    Returns:
        Tuple of (x_coordinates, profile_heights, figure)
    """
    height_map = data["height_map"]
    width = data.get("width", height_map.shape[1])
    x_offset = data.get("x_offset", 0.0)
    x_length = data.get("x_length", width)

    if profile_row is None:
        profile_row = height_map.shape[0] // 2

    x_coords = np.linspace(x_offset, x_offset + x_length, num=width)
    x_profile = height_map[profile_row, :]

    print(f"\nX Profile at row {profile_row}:")
    print("X coordinates (first 10):", x_coords[:10])
    print("Heights (first 10):", x_profile[:10])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x_coords, y=x_profile, mode="lines+markers", name="X Profile")
    )

    fig.update_layout(
        title=f"X Profile (row {profile_row})",
        xaxis_title="X Coordinate",
        yaxis_title=COLORBAR_LABEL,
    )

    if html_filename:
        fig.write_html(html_filename, include_plotlyjs="cdn")
        print(f"X Profile plot saved to {html_filename}")

    return x_coords, x_profile, fig


def plot_height_map_3d(
    height_map, title="Height Map", filename=None, colorscale="Viridis"
):
    """.

    Creates a 3D surface plot of the height map using Plotly.

    Args:
        height_map: 2D numpy array of height values
        title: Plot title
        filename: Output file name for HTML (None = don't save)
        colorscale: Plotly colorscale name

    Returns:
        Plotly figure object
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

    return fig


def plot_height_map_2d(
    height_map, title="Height Map", filename=None, colorscale="Viridis"
):
    """.

    Creates a 2D heatmap visualization of the height map using Plotly.

    Args:
        height_map: 2D numpy array of height values
        title: Plot title
        filename: Output file name for HTML (None = don't save)
        colorscale: Plotly colorscale name

    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(z=height_map, colorscale=colorscale))

    fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y")

    if filename:
        fig.write_html(filename)
        print(f"Saved 2D height map plot as {filename}")

    return fig


def plot_cross_section_plotly(
    x_positions, heights, title="Surface Cross-Section", filename=None
):
    """.

    Create an interactive cross-section plot using Plotly.

    Args:
        x_positions: Array of x-axis positions for the cross-section
        heights: Array of height values at each position
        title: Title for the plot
        filename: Output filename for the interactive HTML (None = don't save)

    Returns:
        Plotly figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "Plotly is required for this function. Install with: pip install plotly"
        )

    fig = go.Figure()

    # Add the profile line
    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=heights,
            mode="lines",
            name="Surface Profile",
            line=dict(color="blue", width=2),
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
            showlegend=False,
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
            showlegend=False,
        )
    )

    # Configure layout
    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title="Height",
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Save to file if filename is provided
    if filename:
        fig.write_html(filename)
        print(f"Interactive cross-section plot saved to {filename}")

    return fig


def plot_multiple_profiles(
    profiles_data, title="Multiple Surface Profiles", filename=None, colorscale=None
):
    """.

    Create an interactive plot with multiple surface profiles for comparison.

    Args:
        profiles_data: List of dictionaries, each containing:
                      - 'x': x-position array
                      - 'y': height values array
                      - 'name': profile name for legend
        title: Title for the plot
        filename: Output filename for HTML (None = don't save)
        colorscale: List of colors to use for the lines (None = auto)

    Returns:
        Plotly figure object
    """
    try:
        import plotly.colors as pc
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError(
            "Plotly is required for this function. Install with: pip install plotly"
        )

    # Create default colorscale if none provided
    if colorscale is None:
        colorscale = pc.qualitative.Plotly

    fig = go.Figure()

    # Add each profile as a separate trace
    for i, profile in enumerate(profiles_data):
        color = colorscale[i % len(colorscale)]

        # Add the profile line
        fig.add_trace(
            go.Scatter(
                x=profile["x"],
                y=profile["y"],
                mode="lines",
                name=profile.get("name", f"Profile {i + 1}"),
                line=dict(color=color, width=2),
            )
        )

    # Configure layout
    fig.update_layout(
        title=title,
        xaxis_title="Position",
        yaxis_title="Height",
        hovermode="closest",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    # Save to file if filename is provided
    if filename:
        fig.write_html(filename)
        print(f"Multiple profiles plot saved to {filename}")

    return fig


""".

Plotly-based visualization functions for height maps.

This module provides a collection of functions for visualizing height maps
using the Plotly library.
"""

# Set up logging
logger = logging.getLogger(__name__)

try:
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not installed. Install with 'pip install plotly'.")


def create_surface_plot(
    height_map: np.ndarray,
    title: str = "3D Surface Plot",
    colorscale: str = "Viridis",
    width: int = 800,
    height: int = 600,
    **kwargs,
) -> Any:
    """.

    Create a 3D surface plot of a height map.

    Args:
        height_map: 2D array of height values
        title: Plot title
        colorscale: Colorscale name
        width: Plot width in pixels
        height: Plot height in pixels
        **kwargs: Additional keyword arguments for go.Surface

    Returns:
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for this function. Install with 'pip install plotly'."
        )

    # Create x, y coordinates
    rows, cols = height_map.shape
    x = np.linspace(0, 1, cols)
    y = np.linspace(0, 1, rows)

    # Create the 3D surface
    fig = go.Figure(
        data=[go.Surface(z=height_map, x=x, y=y, colorscale=colorscale, **kwargs)]
    )

    # Update layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Height",
            aspectratio=dict(x=1, y=1, z=0.5),
        ),
    )

    return fig


def create_heatmap(
    height_map: np.ndarray,
    title: str = "Heightmap",
    colorscale: str = "Viridis",
    width: int = 800,
    height: int = 600,
    **kwargs,
) -> Any:
    """.

    Create a 2D heatmap visualization of a height map.

    Args:
        height_map: 2D array of height values
        title: Plot title
        colorscale: Colorscale name
        width: Plot width in pixels
        height: Plot height in pixels
        **kwargs: Additional keyword arguments for go.Heatmap

    Returns:
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for this function. Install with 'pip install plotly'."
        )

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(z=height_map, colorscale=colorscale, **kwargs))

    # Update layout
    fig.update_layout(
        title=title, width=width, height=height, xaxis_title="X", yaxis_title="Y"
    )

    return fig


def create_contour_plot(
    height_map: np.ndarray,
    title: str = "Contour Map",
    colorscale: str = "Viridis",
    contours: Dict[str, Any] = None,
    width: int = 800,
    height: int = 600,
    **kwargs,
) -> Any:
    """.

    Create a contour plot of a height map.

    Args:
        height_map: 2D array of height values
        title: Plot title
        colorscale: Colorscale name
        contours: Contour settings dictionary
        width: Plot width in pixels
        height: Plot height in pixels
        **kwargs: Additional keyword arguments for go.Contour

    Returns:
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for this function. Install with 'pip install plotly'."
        )

    # Default contour settings
    if contours is None:
        contours = dict(
            start=np.min(height_map),
            end=np.max(height_map),
            size=(np.max(height_map) - np.min(height_map)) / 20,
            showlabels=True,
        )

    # Create the contour plot
    fig = go.Figure(
        data=go.Contour(
            z=height_map, contours=contours, colorscale=colorscale, **kwargs
        )
    )

    # Update layout
    fig.update_layout(
        title=title, width=width, height=height, xaxis_title="X", yaxis_title="Y"
    )

    return fig


def visualize_height_map(
    height_map: np.ndarray,
    plot_type: str = "heatmap",
    title: str = "Height Map",
    colorscale: str = "Viridis",
    width: int = 800,
    height: int = 600,
    filename: Optional[str] = None,
    image_export: bool = False,
    show: bool = True,
    **kwargs,
) -> Any:
    """.

    Visualize a height map using Plotly.

    Args:
        height_map: 2D array of height values
        plot_type: Type of plot ('heatmap', 'contour')
        title: Plot title
        colorscale: Colorscale name
        width: Plot width in pixels
        height: Plot height in pixels
        filename: Optional filename to save the plot
        image_export: Whether to export as an image file
        show: Whether to display the plot
        **kwargs: Additional keyword arguments for the plot

    Returns:
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for this function. Install with 'pip install plotly'."
        )

    # Create the appropriate plot type
    if plot_type.lower() == "contour":
        fig = create_contour_plot(
            height_map=height_map,
            title=title,
            colorscale=colorscale,
            width=width,
            height=height,
            **kwargs,
        )
    else:  # Default to heatmap
        fig = create_heatmap(
            height_map=height_map,
            title=title,
            colorscale=colorscale,
            width=width,
            height=height,
            **kwargs,
        )

    # Save to file if requested
    if filename:
        if image_export:
            # Export as static image
            try:
                import plotly.io as pio

                pio.write_image(fig, filename)
                logger.info(f"Saved plot as image: {filename}")
            except Exception as e:
                logger.error(f"Error saving image: {e}")
        else:
            # Export as HTML
            try:
                # Ensure directory exists before writing file
                os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
                fig.write_html(filename)
                logger.info(f"Saved plot as HTML: {filename}")
            except Exception as e:
                logger.error(f"Error saving HTML: {e}")

    # Show the plot if requested
    if show:
        try:
            fig.show()
        except Exception as e:
            logger.error(f"Error displaying plot: {e}")

    return fig


def visualize_height_map_3d(
    height_map: np.ndarray,
    title: str = "3D Height Map",
    colorscale: str = "Viridis",
    width: int = 800,
    height: int = 600,
    filename: Optional[str] = None,
    image_export: bool = False,
    show: bool = True,
    **kwargs,
) -> Any:
    """.

    Create a 3D visualization of a height map.

    Args:
        height_map: 2D array of height values
        title: Plot title
        colorscale: Colorscale name
        width: Plot width in pixels
        height: Plot height in pixels
        filename: Optional filename to save the plot
        image_export: Whether to export as an image file
        show: Whether to display the plot
        **kwargs: Additional keyword arguments for the 3D plot

    Returns:
        Plotly figure object
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "Plotly is required for this function. Install with 'pip install plotly'."
        )

    # Create 3D surface plot
    fig = create_surface_plot(
        height_map=height_map,
        title=title,
        colorscale=colorscale,
        width=width,
        height=height,
        **kwargs,
    )

    # Save to file if requested
    if filename:
        if image_export:
            # Export as static image
            try:
                import plotly.io as pio

                pio.write_image(fig, filename)
                logger.info(f"Saved 3D plot as image: {filename}")
            except Exception as e:
                logger.error(f"Error saving image: {e}")
        else:
            # Export as HTML
            try:
                # Ensure directory exists before writing file
                os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
                fig.write_html(filename)
                logger.info(f"Saved 3D plot as HTML: {filename}")
            except Exception as e:
                logger.error(f"Error saving HTML: {e}")

    # Show the plot if requested
    if show:
        try:
            fig.show()
        except Exception as e:
            logger.error(f"Error displaying plot: {e}")

    return fig


# Add more Plotly-based visualization functions as needed


# Add unit tests for this module below
if __name__ == "__main__":
    import tempfile
    import unittest

    class TestPlotlyFunctions(unittest.TestCase):
        """Test case for Plotly visualization functions."""

        def setUp(self):
            """Create test data for all tests."""
            # Create a test height map: small to keep tests fast
            self.height_map = np.zeros((50, 50))
            # Add a simple pattern for visualization
            x = np.linspace(-3, 3, 50)
            y = np.linspace(-3, 3, 50)
            X, Y = np.meshgrid(x, y)
            self.height_map = np.sin(X) * np.cos(Y)

            # Create a temp directory for test outputs
            self.temp_dir = tempfile.TemporaryDirectory()

            # Create test profile data
            x = np.linspace(0, 10, 100)
            profile1 = np.sin(x)
            profile2 = np.cos(x)
            self.profiles_data = [
                {"x": x, "y": profile1, "name": "Sine Profile"},
                {"x": x, "y": profile2, "name": "Cosine Profile"},
            ]

            # Create test TMD-like data structure
            self.tmd_data = {
                "height_map": self.height_map,
                "width": 50,
                "height": 50,
                "x_offset": 0.0,
                "y_offset": 0.0,
                "x_length": 10.0,
                "y_length": 10.0,
            }

        def tearDown(self):
            """Clean up temp directory."""
            self.temp_dir.cleanup()

        def test_plot_height_map_3d(self):
            """Test creating a 3D surface plot."""
            # Test without saving
            fig = plot_height_map_3d(self.height_map, title="Test 3D")
            self.assertIsNotNone(fig)

            # Test with saving
            filename = os.path.join(self.temp_dir.name, "test_3d.html")
            fig = plot_height_map_3d(
                self.height_map, title="Test 3D", filename=filename
            )
            self.assertTrue(os.path.exists(filename))

        def test_plot_height_map_2d(self):
            """Test creating a 2D heatmap."""
            # Test without saving
            fig = plot_height_map_2d(self.height_map, title="Test 2D")
            self.assertIsNotNone(fig)

            # Test with saving
            filename = os.path.join(self.temp_dir.name, "test_2d.html")
            fig = plot_height_map_2d(
                self.height_map, title="Test 2D", filename=filename
            )
            self.assertTrue(os.path.exists(filename))

        def test_plot_cross_section(self):
            """Test creating a cross-section plot."""
            x = np.linspace(0, 10, 100)
            heights = np.sin(x)

            # Test without saving
            fig = plot_cross_section_plotly(x, heights, title="Test Cross-Section")
            self.assertIsNotNone(fig)

            # Test with saving
            filename = os.path.join(self.temp_dir.name, "test_cross_section.html")
            fig = plot_cross_section_plotly(
                x, heights, title="Test Cross-Section", filename=filename
            )
            self.assertTrue(os.path.exists(filename))

        def test_plot_multiple_profiles(self):
            """Test creating a plot with multiple profiles."""
            # Test without saving
            fig = plot_multiple_profiles(
                self.profiles_data, title="Test Multiple Profiles"
            )
            self.assertIsNotNone(fig)

            # Test with saving
            filename = os.path.join(self.temp_dir.name, "test_multiple_profiles.html")
            fig = plot_multiple_profiles(
                self.profiles_data, title="Test Multiple Profiles", filename=filename
            )
            self.assertTrue(os.path.exists(filename))

        def test_plot_height_map_with_slider(self):
            """Test creating a 3D plot with z-scale slider."""
            # Test without saving
            fig = plot_height_map_with_slider(self.height_map, html_filename=None)
            self.assertIsNotNone(fig)

            # Test with saving
            filename = os.path.join(self.temp_dir.name, "test_slider.html")
            fig = plot_height_map_with_slider(self.height_map, html_filename=filename)
            self.assertTrue(os.path.exists(filename))

            # Test with partial range
            partial_range = (10, 40, 10, 40)
            fig = plot_height_map_with_slider(
                self.height_map, html_filename=None, partial_range=partial_range
            )
            self.assertIsNotNone(fig)

        def test_plot_x_profile(self):
            """Test extracting and plotting x profile."""
            # Test with default profile row
            x_coords, x_profile, fig = plot_x_profile(self.tmd_data, html_filename=None)
            self.assertEqual(len(x_coords), self.tmd_data["width"])
            self.assertEqual(len(x_profile), self.tmd_data["width"])
            self.assertIsNotNone(fig)

            # Test with specific profile row
            profile_row = 25
            x_coords, x_profile, fig = plot_x_profile(
                self.tmd_data, profile_row=profile_row, html_filename=None
            )
            self.assertEqual(len(x_coords), self.tmd_data["width"])
            self.assertEqual(len(x_profile), self.tmd_data["width"])
            # Check that we got the right row
            np.testing.assert_array_equal(x_profile, self.height_map[profile_row, :])

    # Run the tests
    unittest.main()
