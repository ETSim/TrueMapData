"""Unit tests for TMD model base module."""

import unittest
import numpy as np
import os
import sys
from unittest.mock import patch, MagicMock

# Add the project root to the path to import tmd modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from tmd.exporters.model.base import (
    create_mesh_from_heightmap,
    _add_base_to_mesh,
    calculate_vertex_normals
)


class TestModelBase(unittest.TestCase):
    """Test class for model base functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create simple test heightmaps
        self.heightmap_flat = np.zeros((5, 5), dtype=np.float32)
        
        self.heightmap_slope = np.zeros((5, 5), dtype=np.float32)
        for i in range(5):
            self.heightmap_slope[i, :] = i / 4.0
            
        self.heightmap_peak = np.zeros((5, 5), dtype=np.float32)
        for i in range(5):
            for j in range(5):
                self.heightmap_peak[i, j] = 1.0 - ((i-2)**2 + (j-2)**2) / 8.0
                if self.heightmap_peak[i, j] < 0:
                    self.heightmap_peak[i, j] = 0
    
    def test_create_mesh_from_heightmap(self):
        """Test basic mesh creation from heightmap."""
        # Test with flat heightmap
        vertices, faces = create_mesh_from_heightmap(self.heightmap_flat)
        
        # Should have 5x5=25 vertices
        self.assertEqual(len(vertices), 25)
        
        # Should have (5-1)x(5-1)x2=32 triangles
        self.assertEqual(len(faces), 32)
        
        # All z values should be 0 for flat heightmap
        for vertex in vertices:
            self.assertEqual(vertex[2], 0.0)
    
    def test_mesh_scaling(self):
        """Test mesh scaling parameters."""
        # Test with z scaling
        z_scale = 2.0
        vertices, _ = create_mesh_from_heightmap(
            self.heightmap_slope,
            z_scale=z_scale
        )
        
        # Check that z values are scaled correctly
        max_z = max(vertex[2] for vertex in vertices)
        self.assertEqual(max_z, 1.0 * z_scale)
        
        # Test with x/y scaling
        x_length, y_length = 10.0, 5.0
        vertices, _ = create_mesh_from_heightmap(
            self.heightmap_flat,
            x_length=x_length,
            y_length=y_length
        )
        
        # Check x and y dimensions
        max_x = max(vertex[0] for vertex in vertices)
        max_y = max(vertex[1] for vertex in vertices)
        self.assertEqual(max_x, x_length)
        self.assertEqual(max_y, y_length)
    
    def test_add_base_to_mesh(self):
        """Test adding a base to a mesh with minimal triangles."""
        # Create a simple mesh without base
        vertices, faces = create_mesh_from_heightmap(self.heightmap_flat)
        vertex_count = len(vertices)
        face_count = len(faces)
        
        # Add base with height 1.0
        base_height = 1.0
        new_vertices, new_faces = _add_base_to_mesh(vertices, faces, base_height)
        
        # The base should add 5 new vertices (center + 4 corners)
        self.assertEqual(len(new_vertices), vertex_count + 5)
        
        # Base vertices should have z = -base_height
        base_min_z = min(vertex[2] for vertex in new_vertices)
        self.assertEqual(base_min_z, -base_height)
        
        # Base should include a center vertex
        center_found = False
        x_min = min(v[0] for v in vertices)
        x_max = max(v[0] for v in vertices)
        y_min = min(v[1] for v in vertices)
        y_max = max(v[1] for v in vertices)
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        for v in new_vertices[vertex_count:]:
            # Check if this is approximately the center vertex at base_z
            if (abs(v[0] - center_x) < 1e-5 and
                abs(v[1] - center_y) < 1e-5 and
                abs(v[2] - (-base_height)) < 1e-5):
                center_found = True
                break
        
        self.assertTrue(center_found, "Base should include a center vertex")
        
        # Check corner vertices exist at base_z
        corners_found = 0
        for v in new_vertices[vertex_count:]:
            # Check if this is a corner vertex at base_z
            is_corner = False
            if abs(v[2] - (-base_height)) < 1e-5:
                if (abs(v[0] - x_min) < 1e-5 and abs(v[1] - y_min) < 1e-5) or \
                   (abs(v[0] - x_max) < 1e-5 and abs(v[1] - y_min) < 1e-5) or \
                   (abs(v[0] - x_max) < 1e-5 and abs(v[1] - y_max) < 1e-5) or \
                   (abs(v[0] - x_min) < 1e-5 and abs(v[1] - y_max) < 1e-5):
                    corners_found += 1
        
        self.assertEqual(corners_found, 4, "Base should have 4 corner vertices")
        
        # Base should add minimal triangles (4 for the base + triangles for side walls)
        new_base_faces = new_faces[face_count:]
        self.assertGreaterEqual(len(new_base_faces), 4, "Base should have at least 4 triangles")
        
        # Check that all new triangles are valid
        for face in new_faces:
            # All faces should be valid with 3 distinct vertices
            self.assertEqual(len(face), 3)
            self.assertEqual(len(set(face)), 3)
    
    def test_calculate_vertex_normals(self):
        """Test normal calculation for vertices."""
        # Create a simple mesh
        vertices, faces = create_mesh_from_heightmap(self.heightmap_peak)
        
        # Calculate normals
        normals = calculate_vertex_normals(vertices, faces)
        
        # Should have one normal per vertex
        self.assertEqual(len(normals), len(vertices))
        
        # All normals should be unit vectors
        for normal in normals:
            magnitude = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
            self.assertAlmostEqual(magnitude, 1.0, places=5)
            
        # For heightmap, most normals should point upward (z > 0)
        upward_normals = [n for n in normals if n[2] > 0]
        self.assertEqual(len(upward_normals), len(normals))

    def test_create_mesh_with_base(self):
        """Test creating a mesh with base in one step."""
        base_height = 0.5
        vertices, faces = create_mesh_from_heightmap(
            self.heightmap_peak,
            z_scale=1.0,
            base_height=base_height
        )
        
        # Check that base vertices exist with correct z value
        min_z = min(vertex[2] for vertex in vertices)
        self.assertEqual(min_z, -base_height)


if __name__ == '__main__':
    unittest.main()
