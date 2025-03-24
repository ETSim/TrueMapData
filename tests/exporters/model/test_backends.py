""".

Unit tests for the model exporter backends.

Tests functionality of different exporter backends and their interfaces.
"""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import os
import tempfile

from tmd.exporters.model.backends import (
    MeshIOExporter,
    PyMeshExporter,
    TrimeshExporter,
    OpenCascadeExporter,
    ThreeJSExporter,
    register_backend,
    get_backend
)

class TestModelBackends(unittest.TestCase):
    """Test cases for model exporter backends.."""
    
    def setUp(self):
        """Set up test fixtures.."""
        # Create sample height map
        self.height_map = np.random.random((10, 10))
        
        # Create temp directory for output files
        self.temp_dir = tempfile.mkdtemp(prefix="tmd_test_backends_")
    
    def tearDown(self):
        """Clean up test fixtures.."""
        # Clean up temp files
        for root, dirs, files in os.walk(self.temp_dir, topdown=False):
            for file in files:
                os.unlink(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))
        os.rmdir(self.temp_dir)
    
    def test_meshio_exporter_initialization(self):
        """Test MeshIOExporter initialization.."""
        exporter = MeshIOExporter()
        self.assertIsInstance(exporter, MeshIOExporter)
        
        # Check that supported formats were populated
        self.assertIn("obj", exporter.supported_formats)
        self.assertIn("stl", exporter.supported_formats)
    
    @patch('meshio.write')
    @patch('meshio.Mesh')
    def test_meshio_exporter_export(self, mock_mesh, mock_write):
        """Test MeshIOExporter export method.."""
        # Setup mocks
        mock_mesh.return_value = MagicMock()
        
        # Create exporter
        exporter = MeshIOExporter()
        
        # Create output filename
        output_file = os.path.join(self.temp_dir, "test_export.obj")
        
        # Call export method
        result = exporter.export(
            height_map=self.height_map,
            filename=output_file,
            format="obj",
            x_length=1.0,
            y_length=1.0,
            z_scale=1.0
        )
        
        # Verify mocks were called
        mock_mesh.assert_called_once()
        mock_write.assert_called_once()
        self.assertEqual(result, output_file)
    
    def test_pymesh_exporter_initialization(self):
        """Test PyMeshExporter initialization.."""
        exporter = PyMeshExporter()
        self.assertIsInstance(exporter, PyMeshExporter)
        
        # Check that supported formats were populated
        self.assertIn("obj", exporter.supported_formats)
        self.assertIn("stl", exporter.supported_formats)
        self.assertIn("off", exporter.supported_formats)
    
    @patch('tmd.exporters.model.backends.pymesh')
    def test_pymesh_exporter_export(self, mock_pymesh):
        """Test PyMeshExporter export method.."""
        # Setup mocks
        mock_mesh = MagicMock()
        mock_pymesh.meshio.load_mesh.return_value = mock_mesh
        
        # Create exporter
        exporter = PyMeshExporter()
        
        # Create output filename
        output_file = os.path.join(self.temp_dir, "test_export.obj")
        
        # Call export method
        result = exporter.export(
            height_map=self.height_map,
            filename=output_file,
            format="obj",
            x_length=1.0,
            y_length=1.0,
            z_scale=1.0
        )
        
        # Verify mocks were called
        mock_pymesh.meshio.save_mesh.assert_called_once()
        self.assertEqual(result, output_file)
    
    def test_trimesh_exporter_initialization(self):
        """Test TrimeshExporter initialization.."""
        exporter = TrimeshExporter()
        self.assertIsInstance(exporter, TrimeshExporter)
        
        # Check that supported formats were populated
        self.assertIn("obj", exporter.supported_formats)
        self.assertIn("stl", exporter.supported_formats)
        self.assertIn("glb", exporter.supported_formats)
    
    @patch('tmd.exporters.model.backends.trimesh')
    def test_trimesh_exporter_export(self, mock_trimesh):
        """Test TrimeshExporter export method.."""
        # Setup mocks
        mock_mesh = MagicMock()
        mock_trimesh.Trimesh.return_value = mock_mesh
        
        # Create exporter
        exporter = TrimeshExporter()
        
        # Create output filename
        output_file = os.path.join(self.temp_dir, "test_export.obj")
        
        # Call export method
        result = exporter.export(
            height_map=self.height_map,
            filename=output_file,
            format="obj",
            x_length=1.0,
            y_length=1.0,
            z_scale=1.0
        )
        
        # Verify mocks were called
        mock_mesh.export.assert_called_once()
        self.assertEqual(result, output_file)
    
    def test_opencascade_exporter_initialization(self):
        """Test OpenCascadeExporter initialization.."""
        exporter = OpenCascadeExporter()
        self.assertIsInstance(exporter, OpenCascadeExporter)
        
        # Check that supported formats were populated
        self.assertIn("step", exporter.supported_formats)
        self.assertIn("iges", exporter.supported_formats)
        self.assertIn("brep", exporter.supported_formats)
    
    @patch('tmd.exporters.model.backends.OCC')
    def test_opencascade_exporter_export(self, mock_occ):
        """Test OpenCascadeExporter export method.."""
        # Setup mocks
        mock_shape = MagicMock()
        mock_occ.BRepPrimAPI_MakePrism.return_value = mock_shape
        
        # Create exporter
        exporter = OpenCascadeExporter()
        
        # Create output filename
        output_file = os.path.join(self.temp_dir, "test_export.step")
        
        # Call export method
        result = exporter.export(
            height_map=self.height_map,
            filename=output_file,
            format="step",
            x_length=1.0,
            y_length=1.0,
            z_scale=1.0
        )
        
        # Verify mocks were called correctly
        self.assertEqual(result, output_file)
    
    def test_threejs_exporter_initialization(self):
        """Test ThreeJSExporter initialization.."""
        exporter = ThreeJSExporter()
        self.assertIsInstance(exporter, ThreeJSExporter)
        
        # Check that supported formats were populated
        self.assertIn("json", exporter.supported_formats)
        self.assertIn("glb", exporter.supported_formats)
    
    @patch('tmd.exporters.model.backends.convert_heightmap_to_threejs')
    def test_threejs_exporter_export(self, mock_convert):
        """Test ThreeJSExporter export method.."""
        # Setup mocks
        mock_convert.return_value = "output.json"
        
        # Create exporter
        exporter = ThreeJSExporter()
        
        # Create output filename
        output_file = os.path.join(self.temp_dir, "test_export.json")
        
        # Call export method
        result = exporter.export(
            height_map=self.height_map,
            filename=output_file,
            format="json",
            x_length=1.0,
            y_length=1.0,
            z_scale=1.0,
            add_texture=True
        )
        
        # Verify mocks were called
        mock_convert.assert_called_once()
        self.assertEqual(result, "output.json")
    
    def test_register_and_get_backend(self):
        """Test registering and retrieving a custom backend.."""
        # Create a custom backend class
        class CustomExporter:
            supported_formats = ["custom"]
            def export(self, *args, **kwargs):
                return "custom_export.custom"
        
        # Register the backend
        register_backend("custom", CustomExporter)
        
        # Retrieve the backend
        backend = get_backend("custom")
        
        # Verify it's our custom backend
        self.assertIsInstance(backend, CustomExporter)
        
        # Test getting a backend by format
        backend = get_backend(format="custom")
        self.assertIsInstance(backend, CustomExporter)
    
    def test_get_nonexistent_backend(self):
        """Test retrieving a non-existent backend.."""
        # This should return the default backend
        backend = get_backend("nonexistent")
        self.assertIsInstance(backend, MeshIOExporter)

if __name__ == "__main__":
    unittest.main()
