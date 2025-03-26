"""Unit tests for TMD material set module."""

import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import tempfile
import shutil
from PIL import Image

from tmd.exporters.image.material_set import (
    create_pbr_material_set,
    export_pbr_material_set,
    create_roughness_map,
    create_diffuse_map
)
from tests.resources import create_sample_height_map


class TestMaterialSet(unittest.TestCase):
    """Test class for material set functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample height maps
        self.sample_height_map = create_sample_height_map(pattern="peak")
        self.sample_height_map_with_nan = create_sample_height_map(pattern="with_nan")
        
        # Test output directories
        self.output_dir = os.path.join(self.temp_dir, "materials")
        
        # Expected output files
        self.expected_files = {
            'diffuse': os.path.join(self.output_dir, "material_diffuse.png"),
            'normal': os.path.join(self.output_dir, "material_normal.png"),
            'roughness': os.path.join(self.output_dir, "material_roughness.png"),
            'metallic': os.path.join(self.output_dir, "material_metallic.png"),
            'ao': os.path.join(self.output_dir, "material_ao.png"),
            'height': os.path.join(self.output_dir, "material_height.png")
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_create_pbr_material_set_basic(self):
        """Test creating a basic PBR material set."""
        # Create material set
        materials = create_pbr_material_set(self.sample_height_map)
        
        # Check that all required maps were created
        self.assertIn('diffuse', materials)
        self.assertIn('normal', materials)
        self.assertIn('roughness', materials)
        self.assertIn('metallic', materials)
        self.assertIn('ao', materials)
        self.assertIn('height', materials)
        
        # Check dimensions and types
        self.assertEqual(materials['diffuse'].shape, self.sample_height_map.shape + (3,))
        self.assertEqual(materials['normal'].shape, self.sample_height_map.shape + (3,))
        self.assertEqual(materials['roughness'].shape, self.sample_height_map.shape)
        self.assertEqual(materials['metallic'].shape, self.sample_height_map.shape)
        self.assertEqual(materials['ao'].shape, self.sample_height_map.shape)
        self.assertEqual(materials['height'].shape, self.sample_height_map.shape)
    
    def test_create_pbr_material_set_with_nan(self):
        """Test creating PBR material set from height map with NaN values."""
        # Create material set
        materials = create_pbr_material_set(self.sample_height_map_with_nan)
        
        # Check that no maps have NaN values
        for map_name, map_data in materials.items():
            self.assertFalse(np.any(np.isnan(map_data)), f"NaNs found in {map_name} map")
    
    def test_create_pbr_material_set_parameters(self):
        """Test creating PBR material set with custom parameters."""
        # Create material set with custom parameters
        materials = create_pbr_material_set(
            self.sample_height_map,
            z_scale=2.0,
            roughness_min=0.1,
            roughness_max=0.9,
            invert_roughness=True,
            default_metallic=0.5,
            base_color=(0.8, 0.2, 0.2)  # Reddish color
        )
        
        # Check effects of parameters
        # Metallic value should match default_metallic
        self.assertAlmostEqual(np.mean(materials['metallic']), 0.5)
        
        # Base color should match the given color
        self.assertAlmostEqual(np.mean(materials['diffuse'][:, :, 0]), 0.8)  # R
        self.assertAlmostEqual(np.mean(materials['diffuse'][:, :, 1]), 0.2)  # G
        self.assertAlmostEqual(np.mean(materials['diffuse'][:, :, 2]), 0.2)  # B
    
    @patch('PIL.Image.fromarray')
    @patch('os.makedirs')
    def test_export_pbr_material_set(self, mock_makedirs, mock_fromarray):
        """Test exporting PBR material sets."""
        # Set up PIL mock
        mock_img = MagicMock()
        mock_fromarray.return_value = mock_img
        
        # Create and export material set
        result = export_pbr_material_set(
            self.sample_height_map,
            self.output_dir
        )
        
        # Check that output directory was created
        mock_makedirs.assert_called()
        
        # Check that PIL was used to save each map type
        self.assertEqual(mock_fromarray.call_count, 6)  # Once for each map type
        self.assertEqual(mock_img.save.call_count, 6)
        
        # Check that all expected files were returned
        for map_type in ['diffuse', 'normal', 'roughness', 'metallic', 'ao', 'height']:
            self.assertIn(map_type, result)
            self.assertEqual(result[map_type], self.expected_files[map_type])
    
    def test_actual_export(self):
        """Test actual file export (integration test)."""
        # Create and export material set
        result = export_pbr_material_set(
            self.sample_height_map,
            self.output_dir,
            base_name="test"
        )
        
        # Check that directory was created
        self.assertTrue(os.path.isdir(self.output_dir))
        
        # Check that files were created
        for map_type, file_path in result.items():
            self.assertTrue(os.path.exists(file_path), f"File not found: {file_path}")
            
            # Try opening the file to verify it's a valid image
            try:
                img = Image.open(file_path)
                img.verify()
                
                # Check dimensions match source height map
                img = Image.open(file_path)
                self.assertEqual(img.size, (self.sample_height_map.shape[1], self.sample_height_map.shape[0]))
                
                # Check mode is appropriate for the map type
                if map_type in ['diffuse', 'normal']:
                    self.assertEqual(img.mode, 'RGB')
                else:
                    self.assertEqual(img.mode, 'L')
            except Exception as e:
                self.fail(f"Failed to verify image {file_path}: {e}")
    
    def test_create_roughness_map(self):
        """Test creating roughness map from height map."""
        # Create a roughness map
        roughness = create_roughness_map(
            self.sample_height_map,
            min_val=0.2,
            max_val=0.8,
            invert=False
        )
        
        # Check dimensions and range
        self.assertEqual(roughness.shape, self.sample_height_map.shape)
        self.assertTrue(np.all(roughness >= 0.2) and np.all(roughness <= 0.8))
        
        # Create inverted roughness map
        roughness_inv = create_roughness_map(
            self.sample_height_map,
            min_val=0.2,
            max_val=0.8,
            invert=True
        )
        
        # Check that inverting has an effect
        self.assertNotEqual(np.mean(roughness), np.mean(roughness_inv))
    
    def test_create_diffuse_map(self):
        """Test creating diffuse map from height map."""
        # Create a diffuse map with default gray color
        diffuse_gray = create_diffuse_map(self.sample_height_map)
        
        # Check dimensions and values
        self.assertEqual(diffuse_gray.shape, self.sample_height_map.shape + (3,))
        self.assertAlmostEqual(np.mean(diffuse_gray), 0.5)
        
        # Create diffuse map with custom color
        diffuse_blue = create_diffuse_map(
            self.sample_height_map,
            base_color=(0.1, 0.2, 0.8)  # Blue
        )
        
        # Check color values
        self.assertAlmostEqual(np.mean(diffuse_blue[:, :, 0]), 0.1)  # R
        self.assertAlmostEqual(np.mean(diffuse_blue[:, :, 1]), 0.2)  # G
        self.assertAlmostEqual(np.mean(diffuse_blue[:, :, 2]), 0.8)  # B


if __name__ == '__main__':
    unittest.main()
