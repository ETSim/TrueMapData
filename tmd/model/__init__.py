"""
TMD Model Exporters Package

This module exposes the TMD model exporting components:
  - Base classes: ModelExporter
  - Factory classes: ModelExporterFactory
  - Concrete exporters for different file formats (STL, OBJ, PLY, GLTF, USD, NVBD)
  - Model utility functions for mesh generation and conversion
"""

# Import base classes
try:
    from .base import ModelExporter, export_heightmap_to_model
except ImportError:
    # Fall back to older paths
    try:
        from tmd.exporters.model.base import ModelExporter, export_heightmap_to_model
    except ImportError:
        pass

# Import factory class
try:
    from .factory import ModelExporterFactory
except ImportError:
    # Fall back to older paths
    try:
        from tmd.exporters.model.factory import ModelExporterFactory
    except ImportError:
        pass

# Make get_registered_exporters function available directly
def get_registered_exporters():
    """
    Get a dictionary of available exporters and their status.
    
    Returns:
        Dict[str, bool]: Dictionary with exporter names as keys and 
                        availability status as values
    """
    try:
        return ModelExporterFactory.list_available_formats()
    except (NameError, AttributeError):
        # If ModelExporterFactory isn't available, return empty dict
        return {}

# Get available exporters
def get_available_exporters():
    """
    Get list of available exporter format names.
    
    Returns:
        List[str]: Names of all available exporter formats
    """
    formats = get_registered_exporters()
    return [name for name, available in formats.items() if available]

# Make get_best_exporter function available for auto-selection
def get_best_exporter(preference_order=None):
    """
    Get the best available exporter based on preference order.
    
    Args:
        preference_order: List of exporter names in order of preference
                        (default: ["stl", "obj", "ply", "glb", "usdz", "nvbd"])
    
    Returns:
        Exporter class or None if no exporters are available
    """
    if preference_order is None:
        preference_order = ["stl", "obj", "ply", "glb", "usdz", "nvbd"]
    
    available = get_registered_exporters()
    
    for format_name in preference_order:
        if format_name in available and available[format_name]:
            try:
                return ModelExporterFactory.get_exporter(format_name)
            except (NameError, AttributeError):
                return None
    
    return None

# Import specific exporters when available
try:
    from .stl import STLExporter
except ImportError:
    try:
        from tmd.exporters.model.stl import STLExporter
    except ImportError:
        pass

try:
    from .obj import OBJExporter
except ImportError:
    try:
        from tmd.exporters.model.obj import OBJExporter
    except ImportError:
        pass

try:
    from .ply import PLYExporter
except ImportError:
    try:
        from tmd.exporters.model.ply import PLYExporter
    except ImportError:
        pass

try:
    from .gltf import GLTFExporter
except ImportError:
    try:
        from tmd.exporters.model.gltf import GLTFExporter
    except ImportError:
        pass

try:
    from .usd import USDExporter
except ImportError:
    try:
        from tmd.exporters.model.usd import USDExporter
    except ImportError:
        pass

try:
    from .nvbd import NVBDExporter
except ImportError:
    try:
        from tmd.exporters.model.nvbd import NVBDExporter
    except ImportError:
        pass

# Import utility functions
try:
    from .mesh_utils import (
        calculate_vertex_normals,
        calculate_face_normals,
        calculate_heightmap_normals,
        ensure_watertight_mesh,
        optimize_mesh,
        validate_heightmap,
        generate_uv_coordinates
    )
except ImportError:
    try:
        from tmd.exporters.model.mesh_utils import (
            calculate_vertex_normals,
            calculate_face_normals,
            calculate_heightmap_normals,
            ensure_watertight_mesh,
            optimize_mesh,
            validate_heightmap,
            generate_uv_coordinates
        )
    except ImportError:
        pass

# Define __all__ for explicit exports
__all__ = [
    'ModelExporter',
    'ModelExporterFactory',
    'export_heightmap_to_model',
    'get_registered_exporters',
    'get_available_exporters',
    'get_best_exporter',
    'calculate_vertex_normals',
    'calculate_face_normals',
    'calculate_heightmap_normals',
    'ensure_watertight_mesh',
    'optimize_mesh',
    'validate_heightmap',
    'generate_uv_coordinates',
    'STLExporter',
    'OBJExporter',
    'PLYExporter',
    'GLTFExporter',
    'USDExporter',
    'NVBDExporter'
]