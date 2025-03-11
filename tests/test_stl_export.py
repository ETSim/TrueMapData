"""
Unit tests for the STL export module.

These tests verify the functionality of exporting height maps to STL files
in both ASCII and binary formats.
"""
import os
import struct
import unittest
import tempfile
import numpy as np
import re
from pathlib import Path

from tmd.exporters.stl import convert_heightmap_to_stl


class TestSTLExport(unittest.TestCase):
    """Test cases for STL export functionality."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary directory for output files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        
        # Create a simple height map for testing
        self.small_height_map = np.array([
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 0.5, 1.0]
        ])
        
        # Create a larger height map with a gradient
        size = 10
        x = np.linspace(0, 1, size)
        y = np.linspace(0, 1, size)
        X, Y = np.meshgrid(x, y)
        self.gradient_height_map = X + Y

        # Create a height map with a peak in the middle
        self.peak_height_map = np.zeros((20, 20))
        for i in range(20):
            for j in range(20):
                dx = (i - 10) / 5
                dy = (j - 10) / 5
                self.peak_height_map[i, j] = np.exp(-(dx**2 + dy**2))

    def tearDown(self):
        """Clean up temporary directory after tests."""
        self.temp_dir.cleanup()

    def test_convert_small_height_map_ascii(self):
        """Test converting a small height map to ASCII STL."""
        output_file = os.path.join(self.output_dir, "small_ascii.stl")
        
        # Convert height map to STL
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=output_file,
            x_length=1.0,
            y_length=1.0,
            z_scale=1.0,
            ascii=True
        )
        
        # Check if file exists
        self.assertTrue(os.path.exists(output_file))
        
        # Check file size is reasonable (should be more than 100 bytes)
        file_size = os.path.getsize(output_file)
        self.assertGreater(file_size, 100)
        
        # Verify the file is in ASCII format
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertTrue(content.startswith("solid displacement"))
            self.assertTrue(content.endswith("endsolid displacement\n"))

            # Calculate expected number of triangles
            rows, cols = self.small_height_map.shape
            expected_triangles = 2 * (rows - 1) * (cols - 1)
            
            # Count "facet normal" entries to verify triangle count
            facet_count = content.count("facet normal")
            self.assertEqual(facet_count, expected_triangles)

    def test_convert_small_height_map_binary(self):
        """Test converting a small height map to binary STL."""
        output_file = os.path.join(self.output_dir, "small_binary.stl")
        
        # Convert height map to STL
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=output_file,
            x_length=1.0,
            y_length=1.0,
            z_scale=1.0,
            ascii=False
        )
        
        # Check if file exists
        self.assertTrue(os.path.exists(output_file))
        
        # Verify the file is in binary format by checking the structure
        with open(output_file, 'rb') as f:
            # Read header (80 bytes)
            header = f.read(80)
            self.assertEqual(header[:34], b"TMD Processor Generated Binary STL")
            
            # Read number of triangles (4 bytes)
            triangle_count_bytes = f.read(4)
            triangle_count = struct.unpack("<I", triangle_count_bytes)[0]
            
            # Calculate expected number of triangles
            rows, cols = self.small_height_map.shape
            expected_triangles = 2 * (rows - 1) * (cols - 1)
            self.assertEqual(triangle_count, expected_triangles)
            
            # Check if file size matches expected size
            # Each triangle uses 50 bytes (12 for normal, 36 for vertices, 2 for attribute)
            expected_file_size = 80 + 4 + (50 * triangle_count)
            self.assertEqual(os.path.getsize(output_file), expected_file_size)

    def test_convert_gradient_height_map(self):
        """Test converting a gradient height map to STL."""
        output_file = os.path.join(self.output_dir, "gradient.stl")
        
        # Convert height map to STL
        convert_heightmap_to_stl(
            height_map=self.gradient_height_map,
            filename=output_file,
            ascii=True
        )
        
        # Check if file exists and has reasonable size
        self.assertTrue(os.path.exists(output_file))
        self.assertGreater(os.path.getsize(output_file), 1000)  # Should be bigger for this map
        
        # Calculate expected number of triangles
        rows, cols = self.gradient_height_map.shape
        expected_triangles = 2 * (rows - 1) * (cols - 1)
        
        # Verify triangle count in the file
        with open(output_file, 'r') as f:
            content = f.read()
            facet_count = content.count("facet normal")
            self.assertEqual(facet_count, expected_triangles)

    def test_heightmap_z_scaling(self):
        """Test converting a height map with different z-scaling values."""
        # Create two output files with different z scaling
        output_file1 = os.path.join(self.output_dir, "peak_z1.stl")
        output_file2 = os.path.join(self.output_dir, "peak_z5.stl")
        
        # Convert with z_scale = 1.0
        convert_heightmap_to_stl(
            height_map=self.peak_height_map,
            filename=output_file1,
            z_scale=1.0,
            ascii=True
        )
        
        # Convert with z_scale = 5.0
        convert_heightmap_to_stl(
            height_map=self.peak_height_map,
            filename=output_file2,
            z_scale=5.0,
            ascii=True
        )
        
        # Check that both files exist
        self.assertTrue(os.path.exists(output_file1))
        self.assertTrue(os.path.exists(output_file2))
        
        # Read contents to check z values
        with open(output_file1, 'r') as f1, open(output_file2, 'r') as f2:
            content1 = f1.read()
            content2 = f2.read()
            
            # Find a vertex with non-zero z value in file1
            match1 = re.search(r'vertex\s+\S+\s+\S+\s+(\S+)', content1)
            z_value1 = float(match1.group(1))
            
            # Find corresponding vertex in file2
            match2 = re.search(r'vertex\s+\S+\s+\S+\s+(\S+)', content2)
            z_value2 = float(match2.group(1))
            
            # With z_scale=5, the z values in file2 should be larger
            if z_value1 != 0:  # Avoid comparing zeros
                self.assertGreater(abs(z_value2), abs(z_value1))

    def test_custom_physical_dimensions(self):
        """Test specifying custom physical dimensions for the model."""
        # Create output files with different physical dimensions
        output_file1 = os.path.join(self.output_dir, "dim_default.stl")
        output_file2 = os.path.join(self.output_dir, "dim_custom.stl")
        
        # Convert with default dimensions
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=output_file1,
            ascii=True
        )
        
        # Convert with custom dimensions
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=output_file2,
            x_length=10.0,  # 10x larger
            y_length=10.0,  # 10x larger
            ascii=True
        )
        
        # Check both files exist
        self.assertTrue(os.path.exists(output_file1))
        self.assertTrue(os.path.exists(output_file2))
        
        # Parse files to find vertex coordinates that are non-zero
        with open(output_file1, 'r') as f1, open(output_file2, 'r') as f2:
            content1 = f1.read()
            content2 = f2.read()
            
            # Find vertices with non-zero values (avoid division by zero)
            matches1 = re.findall(r'vertex\s+(\S+)\s+(\S+)', content1)
            matches2 = re.findall(r'vertex\s+(\S+)\s+(\S+)', content2)
            
            # Find a non-zero x value in the first file
            x_value1 = None
            for match in matches1:
                val = float(match[0])
                if abs(val) > 0.001:  # Find a non-zero value
                    x_value1 = val
                    break
                    
            # Find the corresponding value in the second file
            if x_value1 is not None:
                x_value2 = float(matches2[matches1.index(match)][0])
                
                # The values in file2 should be approximately 10x larger
                self.assertAlmostEqual(x_value2 / x_value1, 10.0, delta=0.1)
            else:
                # Just check they're different if we couldn't find non-zero values
                self.assertNotEqual(content1, content2)

    def test_tiny_heightmap(self):
        """Test behavior with very small height maps (< 2x2)."""
        # Create a 1x1 height map
        tiny_map = np.array([[1.0]])
        output_file = os.path.join(self.output_dir, "tiny.stl")
        
        # This should print a message but not create a file
        convert_heightmap_to_stl(
            height_map=tiny_map,
            filename=output_file,
            ascii=True
        )
        
        # Verify file was not created (too small to generate triangles)
        self.assertFalse(os.path.exists(output_file))

    def test_with_offset(self):
        """Test applying x/y offsets to the model."""
        # Create output files with different offsets
        output_file1 = os.path.join(self.output_dir, "no_offset.stl")
        output_file2 = os.path.join(self.output_dir, "with_offset.stl")
        
        # Convert without offset
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=output_file1,
            ascii=True
        )
        
        # Convert with offset
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=output_file2,
            x_offset=10.0,
            y_offset=20.0,
            ascii=True
        )
        
        # Check both files exist
        self.assertTrue(os.path.exists(output_file1))
        self.assertTrue(os.path.exists(output_file2))
        
        # Parse files to find vertex coordinates
        with open(output_file1, 'r') as f1, open(output_file2, 'r') as f2:
            content1 = f1.read()
            content2 = f2.read()
            
            # Find first vertex in each file
            match1 = re.search(r'vertex\s+(\S+)\s+(\S+)', content1)
            x1 = float(match1.group(1))
            y1 = float(match1.group(2))
            
            match2 = re.search(r'vertex\s+(\S+)\s+(\S+)', content2)
            x2 = float(match2.group(1))
            y2 = float(match2.group(2))
            
            # The second file's coordinates should be offset by (10, 20)
            self.assertAlmostEqual(x2, x1 + 10.0, places=5)
            self.assertAlmostEqual(y2, y1 + 20.0, places=5)

    def test_normal_vector_calculation(self):
        """Test that normal vectors are calculated correctly."""
        # Use a simple height map with a very clear slope for testing normal vectors
        simple_map = np.array([
            [0.0, 0.0],
            [0.0, 1.0]
        ])
        output_file = os.path.join(self.output_dir, "simple_normals.stl")
        
        # Convert to STL
        convert_heightmap_to_stl(
            height_map=simple_map,
            filename=output_file,
            ascii=True
        )
        
        # Check file exists
        self.assertTrue(os.path.exists(output_file))
        
        # Read the file and check the normal vector
        with open(output_file, 'r') as f:
            content = f.read()
            
            # Look for any normal vector - with a more generalized pattern
            match = re.search(r'facet normal\s+([^\s]+)\s+([^\s]+)\s+([^\s]+)', content)
            self.assertIsNotNone(match, "No normal vector found in STL file")
            
            # Extract components if match exists
            if match:
                nx = float(match.group(1))
                ny = float(match.group(2))
                nz = float(match.group(3))
                
                # Verify normal is normalized (length = 1)
                normal_length = np.sqrt(nx**2 + ny**2 + nz**2)
                self.assertAlmostEqual(normal_length, 1.0, places=5)

    def test_binary_file_structure(self):
        """Test detailed binary STL file structure."""
        output_file = os.path.join(self.output_dir, "binary_structure.stl")
        
        # Convert to binary STL
        convert_heightmap_to_stl(
            height_map=self.small_height_map,
            filename=output_file,
            ascii=False
        )
        
        # Check file exists
        self.assertTrue(os.path.exists(output_file))
        
        # Read and verify binary structure in detail
        with open(output_file, 'rb') as f:
            # Header (80 bytes)
            header = f.read(80)
            
            # Triangle count (4 bytes)
            triangle_count_bytes = f.read(4)
            triangle_count = struct.unpack("<I", triangle_count_bytes)[0]
            
            # Calculate expected triangle count
            rows, cols = self.small_height_map.shape
            expected_triangles = 2 * (rows - 1) * (cols - 1)
            self.assertEqual(triangle_count, expected_triangles)
            
            # Read first triangle data (50 bytes)
            # Normal vector (3 floats, 12 bytes)
            normal_bytes = f.read(12)
            nx, ny, nz = struct.unpack("<fff", normal_bytes)
            
            # Normal should be normalized (length = 1)
            normal_length = np.sqrt(nx**2 + ny**2 + nz**2)
            self.assertAlmostEqual(normal_length, 1.0, places=5)
            
            # Vertices (9 floats, 36 bytes)
            vertex_bytes = f.read(36)
            vertices = struct.unpack("<fffffffff", vertex_bytes)
            
            # Attribute byte count (2 bytes)
            attr_bytes = f.read(2)
            attr_count = struct.unpack("<H", attr_bytes)[0]
            self.assertEqual(attr_count, 0)  # Should be 0
            
            # Verify total file size is correct
            # 80 byte header + 4 byte triangle count + (50 bytes * triangle_count)
            expected_size = 80 + 4 + (50 * triangle_count)
            self.assertEqual(os.path.getsize(output_file), expected_size)


if __name__ == "__main__":
    unittest.main()
