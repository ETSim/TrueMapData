""".

Polyscope-based visualization for TMD data.

This module provides 3D visualization capabilities using Polyscope.
"""

from typing import Optional, Tuple

import numpy as np

try:
    import polyscope as ps
    import polyscope.imgui as psim

    HAS_POLYSCOPE = True
except ImportError:
    HAS_POLYSCOPE = False
    raise ImportError(
        "Polyscope is required for this module. Install with: pip install polyscope"
    )


class PolyscopePlotter:
    """Class for creating interactive 3D visualizations of height maps and 3D data using Polyscope.."""

    def __init__(self, backend="", width=1024, height=768, background_color=None):
        """.

        Initialize the Polyscope plotter.

        Args:
            backend: Rendering backend (empty string for default)
            width: Window width
            height: Window height
            background_color: Background color as (r,g,b) tuple (default: None for Polyscope default)
        """
        if not HAS_POLYSCOPE:
            raise ImportError(
                "Polyscope is required. Install with: pip install polyscope"
            )

        self.initialized = False
        self.height_maps = {}  # Store height map data for callbacks
        self.callback_registry = []  # Store registered callbacks

        # Initialize Polyscope - the backend parameter must be a string
        ps.init(backend=backend if backend is not None else "")
        ps.set_program_name("TMD Visualizer")
        ps.set_window_size(width, height)

        if background_color:
            ps.set_ground_plane_mode("none")
            ps.set_background_color(background_color)

        self.initialized = True

    def _register_callback(self, callback_fn):
        """Register a callback function.."""
        self.callback_registry.append(callback_fn)

    def _create_default_callback(self):
        """Create the default callback for height map sliders.."""

        def default_callback():
            """Handle UI callbacks for height map sliders.."""
            for name, data in self.height_maps.items():
                mesh = data["mesh"]
                current_scale = data["current_scale"]

                # Create a slider for each height map
                _, new_scale = psim.SliderFloat(
                    f"{name} Height Scale",
                    current_scale,
                    v_min=data["min_scale"],
                    v_max=data["max_scale"],
                )

                # Update height if scale changed
                if new_scale != current_scale:
                    # Update the stored value
                    data["current_scale"] = new_scale

                    # Update the vertex positions
                    new_vertices = data["vertices"].copy()
                    new_vertices[:, 2] = data["height_map"].flatten() * new_scale
                    mesh.update_vertex_positions(new_vertices)

                    # Update the stored vertices
                    data["vertices"] = new_vertices

                # Add a separator between controls
                psim.Separator()

        return default_callback

    def plot_height_map(
        self,
        height_map: np.ndarray,
        x_range: Tuple[float, float] = None,
        y_range: Tuple[float, float] = None,
        name: str = "height_map",
        enabled: bool = True,
        add_height_slider: bool = True,
        min_scale: float = 0.0,
        max_scale: float = 10.0,
        initial_scale: float = 1.0,
        edge_width: float = 0.0,
        smooth_shade: bool = True,
    ):
        """.

        Plot a height map as a 3D surface.

        Args:
            height_map: 2D numpy array with height values
            x_range: Optional (min, max) range for x coordinates
            y_range: Optional (min, max) range for y coordinates
            name: Name for the surface in Polyscope
            enabled: Whether the surface is initially visible
            add_height_slider: Whether to add a height scale slider
            min_scale: Minimum value for height scale slider
            max_scale: Maximum value for height scale slider
            initial_scale: Initial height scale
            edge_width: Width of edges (0 for no edges)
            smooth_shade: Whether to use smooth shading

        Returns:
            Polyscope surface mesh object
        """
        if not self.initialized:
            raise RuntimeError("Polyscope not initialized. Call init() first.")

        # Get height map dimensions
        h, w = height_map.shape

        # Create coordinate grids
        if x_range is None:
            x_range = (0, w - 1)
        if y_range is None:
            y_range = (0, h - 1)

        x = np.linspace(x_range[0], x_range[1], w)
        y = np.linspace(y_range[0], y_range[1], h)
        X, Y = np.meshgrid(x, y)

        # Create vertices (x,y,z)
        vertices = np.zeros((w * h, 3))
        vertices[:, 0] = X.flatten()
        vertices[:, 1] = Y.flatten()
        vertices[:, 2] = height_map.flatten() * initial_scale

        # Create faces (triangles)
        faces = []
        for i in range(h - 1):
            for j in range(w - 1):
                v0 = i * w + j
                v1 = i * w + (j + 1)
                v2 = (i + 1) * w + j
                v3 = (i + 1) * w + (j + 1)

                # Two triangles per grid cell
                faces.append([v0, v1, v3])
                faces.append([v0, v3, v2])

        faces = np.array(faces)

        # Register mesh in Polyscope
        mesh = ps.register_surface_mesh(
            name, vertices, faces, smooth_shade=smooth_shade, enabled=enabled
        )

        # Add height values as a scalar field
        mesh.add_scalar_quantity("height", vertices[:, 2], enabled=True, cmap="viridis")

        # Set edge width
        if edge_width > 0:
            mesh.set_edge_width(edge_width)

        # Store height map information for the slider callback
        if add_height_slider:
            self.height_maps[name] = {
                "mesh": mesh,
                "height_map": height_map,
                "vertices": vertices.copy(),
                "min_scale": min_scale,
                "max_scale": max_scale,
                "current_scale": initial_scale,
            }

            # Register callback if this is the first height map
            if len(self.height_maps) == 1 and add_height_slider:
                default_callback = self._create_default_callback()
                ps.set_user_callback(default_callback)

        return mesh

    def plot_point_cloud(
        self,
        points: np.ndarray,
        values: Optional[np.ndarray] = None,
        name: str = "point_cloud",
        point_size: float = 0.01,
        enabled: bool = True,
        cmap: str = "viridis",
    ):
        """.

        Plot a 3D point cloud.

        Args:
            points: Nx3 array of point coordinates
            values: Optional array of scalar values for coloring points
            name: Name for the point cloud in Polyscope
            point_size: Size of points
            enabled: Whether the point cloud is initially visible
            cmap: Colormap for scalar values

        Returns:
            Polyscope point cloud object
        """
        if not self.initialized:
            raise RuntimeError("Polyscope not initialized. Call init() first.")

        # Register point cloud in Polyscope
        point_cloud = ps.register_point_cloud(name, points, enabled=enabled)

        # Set point size
        point_cloud.set_radius(point_size, relative=False)

        # Add values as scalar quantity if provided
        if values is not None:
            point_cloud.add_scalar_quantity("values", values, enabled=True, cmap=cmap)

        return point_cloud

    def plot_surface(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        vertex_values: Optional[np.ndarray] = None,
        name: str = "surface",
        enabled: bool = True,
        smooth_shade: bool = True,
    ):
        """.

        Plot a generic 3D surface mesh.

        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices
            vertex_values: Optional array of scalar values for coloring vertices
            name: Name for the surface in Polyscope
            enabled: Whether the surface is initially visible
            smooth_shade: Whether to use smooth shading

        Returns:
            Polyscope surface mesh object
        """
        if not self.initialized:
            raise RuntimeError("Polyscope not initialized. Call init() first.")

        # Register surface mesh in Polyscope
        mesh = ps.register_surface_mesh(
            name, vertices, faces, smooth_shade=smooth_shade, enabled=enabled
        )

        # Add vertex values as scalar quantity if provided
        if vertex_values is not None:
            mesh.add_scalar_quantity(
                "values", vertex_values, enabled=True, cmap="viridis"
            )

        return mesh

    def add_callback(self, callback_fn):
        """.

        Add a custom UI callback function.

        Args:
            callback_fn: Callback function for UI interaction
        """
        ps.set_user_callback(callback_fn)

    def show(self):
        """Show the visualization and start the Polyscope UI.."""
        if not self.initialized:
            raise RuntimeError("Polyscope not initialized. Call init() first.")
        ps.show()

    def screenshot(self, filename="screenshot.png"):
        """.

        Take a screenshot of the current view.

        Args:
            filename: Output filename for the screenshot

        Returns:
            Path to the saved screenshot
        """
        if not self.initialized:
            raise RuntimeError("Polyscope not initialized. Call init() first.")
        ps.screenshot(filename)
        return filename

    def set_camera_params(self, center=None, zoom=None, rotation=None):
        """.

        Set camera parameters.

        Args:
            center: (x,y,z) camera target point
            zoom: Zoom level
            rotation: Camera rotation as quaternion (w,x,y,z)
        """
        if center is not None:
            ps.look_at(center)
        if zoom is not None:
            ps.set_view_projection_mode("perspective")
            ps.set_automatical_view_camera(False)
            # Actually adjusting zoom is not directly available in Polyscope API
        if rotation is not None:
            ps.set_view_rotation(rotation)
