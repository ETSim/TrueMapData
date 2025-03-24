""".

Unit tests for mesh utility functions.
"""

import unittest
import numpy as np

from tmd.exporters.model.mesh_utils import (
    calculate_vertex_normals,
    optimize_mesh,
    triangulate_quad_mesh
)


class TestMeshUtils(unittest.TestCase):
    """Test cases for mesh utility functions.."""
    
    def setUp(self):
        """Set up test data.."""
        # Create a simple cube mesh for testing
        self.cube_vertices = [
            [0, 0, 0],  # 0
            [1, 0, 0],  # 1
            [1, 1, 0],  # 2
            [0, 1, 0],  # 3
            [0, 0, 1],  # 4
            [1, 0, 1],  # 5
            [1, 1, 1],  # 6
            [0, 1, 1],  # 7
        ]
        
        # Triangulated faces of the cube
        self.cube_faces = [
            [0, 1, 2], [0, 2, 3],  # Bottom face
            [4, 5, 6], [4, 6, 7],  # Top face
            [0, 1, 5], [0, 5, 4],  # Front face
            [3, 2, 6], [3, 6, 7],  # Back face
            [0, 3, 7], [0, 7, 4],  # Left face
            [1, 2, 6], [1, 6, 5],  # Right face
        ]
        
        # Create a quad mesh
        self.quads = [
            [0, 1, 2, 3],  # Bottom face
            [4, 5, 6, 7],  # Top face
            [0, 1, 5, 4],  # Front face
            [3, 2, 6, 7],  # Back face
            [0, 3, 7, 4],  # Left face
            [1, 2, 6, 5],  # Right face
        ]
    
    def test_calculate_vertex_normals(self):
        """Test calculation of vertex normals.."""
        normals = calculate_vertex_normals(self.cube_vertices, self.cube_faces)
        
        # Should have one normal per vertex
        self.assertEqual(len(normals), len(self.cube_vertices))
        
        # Each normal should be a unit vector
        for normal in normals:
            norm = np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)
            self.assertAlmostEqual(norm, 1.0, places=5)
            
        # Check normals for cube corners - should point in expected directions
        # For example, corner 0 (0,0,0) should have normal approximately (-1,-1,-1) normalized
        corner0_normal = np.array(normals[0])
        self.assertTrue(all(corner0_normal < 0))
    
    def test_optimize_mesh(self):
        """Test mesh optimization by merging duplicate vertices.."""
        # Create a mesh with duplicate vertices
        vertices = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],  # Duplicate of vertex 0
            [1, 0, 0],  # Duplicate of vertex 1
            [0.999, 0.001, 0]  # Almost duplicate of vertex 1 (within tolerance)
        ]
        
        faces = [
            [0, 1, 2],
            [3, 4, 2],
            [3, 5, 2]
        ]
        
        # Optimize mesh
        optimized_vertices, optimized_faces = optimize_mesh(vertices, faces, tolerance=0.01)
        
        # Should have fewer vertices
        self.assertLess(len(optimized_vertices), len(vertices))
        
        # Should have 3 unique vertices (0,0,0), (1,0,0), (0,1,0)
        self.assertEqual(len(optimized_vertices), 3)
    
    def test_triangulate_quad_mesh(self):
        """Test triangulation of a quad mesh.."""
        triangles = triangulate_quad_mesh(self.cube_vertices, self.quads)
        
        # Should have twice as many triangles as quads
        self.assertEqual(len(triangles), 2 * len(self.quads))
        
        # Each triangle should have 3 vertices
        for triangle in triangles:
            self.assertEqual(len(triangle), 3)
        
        # Check that some specific triangulation patterns are followed
        # For example, the first quad [0,1,2,3] should become triangles [0,1,2] and [0,2,3]
        self.assertIn([0, 1, 2], triangles)
        self.assertIn([0, 2, 3], triangles)


if __name__ == "__main__":
    unittest.main()
