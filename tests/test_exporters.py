"""Tests for TMD data exporters."""
import unittest
import os
import tempfile
import numpy as np
from PIL import Image


class TestExporters(unittest.TestCase):
    """Test the exporter functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a test height map
        self.height_map = np.zeros((50, 70), dtype=np.float32)
        
        # Fill with a pattern
        for i in range(50):
            for j in range(70):
                self.height_map[i, j] = np.sin(i/10) * np.cos(j/10) * 0.5 + 0.5
        
        self.data_dict = {
            'height_map': self.height_map,
            'width': 70,
            'height': 50,
            'x_length': 1.0,
            'y_length': 0.7,
            'x_offset': 0,
            'y_offset': 0,
            'comment': 'Test data',
            'header': 'Test TMD'
        }
        
        # Create a temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after tests."""
        # Remove all files in the temporary directory
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        
        # Remove the directory
        os.rmdir(self.temp_dir)
    
    def test_export_to_npy(self):
        """Test exporting to NumPy's .npy format."""
        from tmd.exporters.numpy import export_to_npy
        
        # Export to .npy
        npy_path = os.path.join(self.temp_dir, 'test_export.npy')
        result_path = export_to_npy(self.height_map, npy_path)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(result_path))
        
        # Load the file and check contents
        loaded = np.load(result_path)
        np.testing.assert_array_equal(loaded, self.height_map)
        
        # Test with compression
        npz_path = os.path.join(self.temp_dir, 'test_export_compressed')
        result_path_compressed = export_to_npy(self.height_map, npz_path, compress=True)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(result_path_compressed))
        
        # Load the file and check contents
        loaded = np.load(result_path_compressed)['height_map']
        np.testing.assert_array_equal(loaded, self.height_map)
    
    def test_export_to_npz(self):
        """Test exporting to NumPy's .npz format."""
        from tmd.exporters.numpy import export_to_npz
        
        # Export to .npz
        npz_path = os.path.join(self.temp_dir, 'test_export.npz')
        result_path = export_to_npz(self.data_dict, npz_path)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(result_path))
        
        # Load the file and check contents
        loaded = np.load(result_path)
        
        # Check all keys
        for key, value in self.data_dict.items():
            if key == 'height_map':
                np.testing.assert_array_equal(loaded[key], value)
            elif isinstance(value, (str)):
                # String values are converted to arrays
                self.assertEqual(loaded[key][0], value)
    
    def test_export_metadata_txt(self):
        """Test exporting metadata to a text file."""
        from tmd.exporters.numpy import export_metadata_txt
        
        # Export metadata
        txt_path = os.path.join(self.temp_dir, 'test_metadata.txt')
        result_path = export_metadata_txt(self.data_dict, txt_path)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(result_path))
        
        # Check content
        with open(result_path, 'r') as f:
            content = f.read()
            
        # Verify key metadata is included
        self.assertIn('width: 70', content)
        self.assertIn('height: 50', content)
        self.assertIn('comment: Test data', content)
    
    def test_export_to_stl(self):
        """Test exporting to STL format."""
        from tmd.exporters.stl import convert_heightmap_to_stl
        
        # Export to ASCII STL
        ascii_stl_path = os.path.join(self.temp_dir, 'test_ascii.stl')
        convert_heightmap_to_stl(self.height_map, ascii_stl_path, ascii=True)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(ascii_stl_path))
        
        # Check the first line for the "solid" keyword
        with open(ascii_stl_path, 'r') as f:
            first_line = f.readline().strip()
            self.assertTrue(first_line.startswith('solid'))
        
        # Export to binary STL
        binary_stl_path = os.path.join(self.temp_dir, 'test_binary.stl')
        convert_heightmap_to_stl(self.height_map, binary_stl_path, ascii=False)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(binary_stl_path))
        
        # Check file size (binary STL should be smaller than ASCII for this data)
        binary_size = os.path.getsize(binary_stl_path)
        self.assertGreater(binary_size, 84)  # Header (80) + Triangle count (4)
    
    def test_export_to_sdf(self):
        """Test exporting to SDF format."""
        from tmd.exporters.sdf import export_to_sdf, read_sdf_file
        
        # Export to SDF
        sdf_path = os.path.join(self.temp_dir, 'test_export.sdf')
        result_path = export_to_sdf(self.height_map, sdf_path, metadata=self.data_dict)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(result_path))
        
        # Read back the SDF file
        loaded_sdf, loaded_metadata = read_sdf_file(result_path)
        
        # Check dimensions
        self.assertEqual(loaded_sdf.shape, self.height_map.shape)
        
        # Check some metadata is present
        self.assertGreater(len(loaded_metadata), 0)
    
    def test_displacement_map_export(self):
        """Test exporting to displacement map image."""
        from tmd.exporters.image import convert_heightmap_to_displacement_map
        
        # Export to displacement map
        disp_path = os.path.join(self.temp_dir, 'test_displacement.png')
        result_img = convert_heightmap_to_displacement_map(self.height_map, disp_path)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(disp_path))
        
        # Check that it returned a PIL Image
        self.assertIsInstance(result_img, Image.Image)
        
        # Check dimensions
        self.assertEqual(result_img.size, (self.height_map.shape[1], self.height_map.shape[0]))
    
    def test_normal_map_export(self):
        """Test exporting to normal map image."""
        from tmd.exporters.image import convert_heightmap_to_normal_map
        
        # Export to normal map
        normal_path = os.path.join(self.temp_dir, 'test_normal.png')
        result_img = convert_heightmap_to_normal_map(self.height_map, normal_path)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(normal_path))
        
        # Check that it returned a PIL Image
        self.assertIsInstance(result_img, Image.Image)
        
        # Check dimensions
        self.assertEqual(result_img.size, (self.height_map.shape[1], self.height_map.shape[0]))
        
        # Normal maps should be RGB
        self.assertEqual(result_img.mode, 'RGB')
    
    def test_plotters_matplotlib(self):
        """Test matplotlib plotting functions (basic object checks only)."""
        try:
            from tmd.plotters.matplotlib import (
                plot_height_map_matplotlib, 
                plot_2d_heatmap_matplotlib,
                plot_x_profile_matplotlib
            )
            
            # Test if function exists (might not execute if matplotlib is not available)
            self.assertTrue(callable(plot_height_map_matplotlib))
            self.assertTrue(callable(plot_2d_heatmap_matplotlib))
            self.assertTrue(callable(plot_x_profile_matplotlib))
            
        except ImportError:
            self.skipTest("Matplotlib not installed")
    
    def test_plotters_seaborn(self):
        """Test seaborn plotting functions (basic object checks only)."""
        try:
            from tmd.plotters.seaborn import plot_height_map_seaborn, plot_2d_heatmap_seaborn
            
            # Test if function exists (might not execute if seaborn is not available)
            self.assertTrue(callable(plot_height_map_seaborn))
            self.assertTrue(callable(plot_2d_heatmap_seaborn))
            
        except ImportError:
            self.skipTest("Seaborn not installed")
    
    def test_bump_map_export(self):
        """Test exporting to bump map image."""
        from tmd.exporters.image import convert_heightmap_to_bump_map
        
        # Export to bump map
        bump_path = os.path.join(self.temp_dir, 'test_bump.png')
        result_img = convert_heightmap_to_bump_map(self.height_map, bump_path)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(bump_path))
        
        # Check that it returned a PIL Image
        self.assertIsInstance(result_img, Image.Image)
        
        # Check dimensions
        self.assertEqual(result_img.size, (self.height_map.shape[1], self.height_map.shape[0]))
    
    def test_bsrf_map_export(self):
        """Test exporting to BSRF map image."""
        from tmd.exporters.image import convert_heightmap_to_bsrf_map
        
        # Export to BSRF map
        bsrf_path = os.path.join(self.temp_dir, 'test_bsrf.png')
        result_img = convert_heightmap_to_bsrf_map(self.height_map, bsrf_path)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(bsrf_path))
        
        # Check that it returned a PIL Image
        self.assertIsInstance(result_img, Image.Image)
        
        # Check dimensions
        self.assertEqual(result_img.size, (self.height_map.shape[1], self.height_map.shape[0]))
    
    def test_ao_map_export(self):
        """Test exporting to ambient occlusion map image."""
        from tmd.exporters.image import convert_heightmap_to_ao_map
        
        # Use a smaller height map for this test to speed it up
        small_map = self.height_map[0:10, 0:10]
        
        # Export to AO map
        ao_path = os.path.join(self.temp_dir, 'test_ao.png')
        result_img = convert_heightmap_to_ao_map(small_map, ao_path, samples=4)  # Use fewer samples for speed
        
        # Check that the file was created
        self.assertTrue(os.path.exists(ao_path))
        
        # Check that it returned a PIL Image
        self.assertIsInstance(result_img, Image.Image)
        
        # Check dimensions
        self.assertEqual(result_img.size, (small_map.shape[1], small_map.shape[0]))
    
    def test_multi_channel_map_export(self):
        """Test exporting to multi-channel map image."""
        from tmd.exporters.image import convert_heightmap_to_multi_channel_map
        
        # Export to multi-channel map (RGBE mode)
        rgbe_path = os.path.join(self.temp_dir, 'test_rgbe.png')
        result_img = convert_heightmap_to_multi_channel_map(self.height_map, rgbe_path, channel_type="rgbe")
        
        # Check that the file was created
        self.assertTrue(os.path.exists(rgbe_path))
        
        # Check that it returned a PIL Image
        self.assertIsInstance(result_img, Image.Image)
        
        # Check dimensions
        self.assertEqual(result_img.size, (self.height_map.shape[1], self.height_map.shape[0]))
        
        # Check mode (should be RGBA)
        self.assertEqual(result_img.mode, 'RGBA')
        
        # Test RG mode
        rg_path = os.path.join(self.temp_dir, 'test_rg.png')
        result_img2 = convert_heightmap_to_multi_channel_map(self.height_map, rg_path, channel_type="rg")
        
        # Check that the file was created
        self.assertTrue(os.path.exists(rg_path))
        
        # Check mode (should be RGB)
        self.assertEqual(result_img2.mode, 'RGB')
        
        # Test invalid mode
        with self.assertRaises(ValueError):
            convert_heightmap_to_multi_channel_map(self.height_map, 
                                                  os.path.join(self.temp_dir, 'invalid.png'), 
                                                  channel_type="invalid")


if __name__ == '__main__':
    unittest.main()