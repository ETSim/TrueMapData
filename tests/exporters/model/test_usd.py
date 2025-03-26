"""Unit tests for TMD USD module."""

import unittest
import numpy as np
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add the project root to the path to import tmd modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from tmd.exporters.model.usd import convert_heightmap_to_usd, convert_heightmap_to_usdz, _create_texture_from_heightmap


class TestUsd(unittest.TestCase):
    """Test class for USD export functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test heightmaps
        self.heightmap_flat = np.zeros((10, 10), dtype=np.float32)
        
        self.heightmap_slope = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            for j in range(10):
                self.heightmap_slope[i, j] = i / 10.0
        
        self.heightmap_peak = np.zeros((10, 10), dtype=np.float32)
        for i in range(10):
            for j in range(10):
                self.heightmap_peak[i, j] = 1.0 - ((i-5)**2 + (j-5)**2) / 50.0
                if self.heightmap_peak[i, j] < 0:
                    self.heightmap_peak[i, j] = 0
    
    @patch('tmd.exporters.model.usd.HAS_USD', False)
    def test_missing_dependencies(self):
        """Test behavior when USD dependencies are missing."""
        # Should return None if USD modules not available
        result = convert_heightmap_to_usd(self.heightmap_flat, "test.usda")
        self.assertIsNone(result)
        
        # USDZ exporter should also return None
        result = convert_heightmap_to_usdz(self.heightmap_flat, "test.usdz")
        self.assertIsNone(result)
    
    def test_input_validation(self):
        """Test input validation for USD export."""
        with patch('tmd.exporters.model.usd.HAS_USD', True):
            # Test with None heightmap
            result = convert_heightmap_to_usd(None, "test.usda")
            self.assertIsNone(result)
            
            # Test with empty heightmap
            result = convert_heightmap_to_usd(np.array([]), "test.usda")
            self.assertIsNone(result)
            
            # Test with too small heightmap
            result = convert_heightmap_to_usd(np.array([[1]]), "test.usda")
            self.assertIsNone(result)
    
    @patch('tmd.exporters.model.usd.HAS_USD', True)
    @patch('pxr.Usd.Stage.CreateNew')
    @patch('tmd.exporters.model.usd.create_mesh_from_heightmap')
    @patch('tmd.exporters.model.usd.ensure_directory_exists')
    def test_usd_export_basic(self, mock_ensure_dir, mock_create_mesh, mock_create_stage):
        """Test basic USD export functionality."""
        # Setup mocks
        mock_ensure_dir.return_value = True
        mock_stage = MagicMock()
        mock_create_stage.return_value = mock_stage
        
        # Mock the mesh creation
        vertices = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0]
        ]
        faces = [
            [0, 1, 2],
            [1, 3, 2]
        ]
        mock_create_mesh.return_value = (vertices, faces)
        
        # Mock USD API
        mock_mesh = MagicMock()
        mock_mesh_define = MagicMock(return_value=mock_mesh)
        
        with patch('pxr.UsdGeom.Mesh.Define', mock_mesh_define):
            # Test export
            result = convert_heightmap_to_usd(
                self.heightmap_flat,
                filename="test.usda",
                z_scale=1.0,
                base_height=0.0
            )
            
            # Check that stage was created and mesh was defined
            mock_create_stage.assert_called_once_with("test.usda")
            mock_mesh_define.assert_called_once()
            
            # Check points and faces were set
            mock_mesh.CreatePointsAttr.assert_called_once()
            mock_mesh.CreateFaceVertexCountsAttr.assert_called_once()
            mock_mesh.CreateFaceVertexIndicesAttr.assert_called_once()
            
            # Check that stage was saved
            mock_stage.Save.assert_called_once()
            
            # Result should be the filename
            self.assertEqual(result, "test.usda")
    
    @patch('tmd.exporters.model.usd.HAS_USD', True)
    @patch('pxr.Usd.Stage.CreateNew')
    @patch('tmd.exporters.model.usd.create_mesh_from_heightmap')
    @patch('tmd.exporters.model.usd._create_texture_from_heightmap')
    @patch('cv2.imwrite')
    @patch('tempfile.NamedTemporaryFile')
    def test_usd_with_texture(self, mock_temp_file, mock_imwrite, mock_create_texture, 
                              mock_create_mesh, mock_create_stage):
        """Test USD export with texture."""
        # Setup mocks
        mock_stage = MagicMock()
        mock_create_stage.return_value = mock_stage
        
        # Mock the mesh creation
        vertices = [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0]
        ]
        faces = [
            [0, 1, 2],
            [1, 3, 2]
        ]
        mock_create_mesh.return_value = (vertices, faces)
        
        # Mock texture creation
        mock_create_texture.return_value = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Mock temporary file
        mock_temp_file_instance = MagicMock()
        mock_temp_file_instance.name = "/tmp/texture.png"
        mock_temp_file().__enter__.return_value = mock_temp_file_instance
        
        # Mock USD geometry and shader modules
        mock_mesh = MagicMock()
        mock_material = MagicMock()
        mock_shader = MagicMock()
        
        with patch('pxr.UsdGeom.Mesh.Define', return_value=mock_mesh), \
             patch('pxr.UsdShade.Material.Define', return_value=mock_material), \
             patch('pxr.UsdShade.Shader.Define', return_value=mock_shader), \
             patch('os.path.exists', return_value=True), \
             patch('os.remove'):
            
            # Test export with texture
            result = convert_heightmap_to_usd(
                self.heightmap_flat,
                filename="test.usda",
                z_scale=1.0,
                add_texture=True
            )
            
            # Check that texture was created
            mock_create_texture.assert_called_once()
            mock_imwrite.assert_called_once()
            
            # Should create primvar for texture coordinates
            mock_mesh.CreatePrimvar.assert_called_once()
            
            # Result should be the filename
            self.assertEqual(result, "test.usda")
    
    @patch('tmd.exporters.model.usd.HAS_USD', True)
    @patch('tmd.exporters.model.usd.convert_heightmap_to_usd')
    @patch('pxr.UsdUtils.CreateNewARKitUsdzPackage')
    @patch('os.path.exists', return_value=True)
    @patch('tempfile.mkdtemp', return_value="/tmp/usd_test")
    @patch('shutil.rmtree')
    def test_usdz_export(self, mock_rmtree, mock_mkdtemp, mock_path_exists, 
                         mock_create_usdz, mock_convert_usd):
        """Test USDZ export functionality."""
        # Setup mocks
        mock_convert_usd.return_value = "/tmp/usd_test/temp.usda"
        
        # Test USDZ export
        result = convert_heightmap_to_usdz(
            self.heightmap_flat,
            filename="test.usdz",
            z_scale=2.0,
            base_height=0.1
        )
        
        # Check that USD was created first
        mock_convert_usd.assert_called_once()
        self.assertEqual(mock_convert_usd.call_args[1]['filename'], "/tmp/usd_test/temp.usda")
        self.assertEqual(mock_convert_usd.call_args[1]['z_scale'], 2.0)
        self.assertEqual(mock_convert_usd.call_args[1]['base_height'], 0.1)
        
        # Check that USDZ package was created
        mock_create_usdz.assert_called_once_with("/tmp/usd_test/temp.usda", "test.usdz")
        
        # Check that temp directory was cleaned up
        mock_rmtree.assert_called_once_with("/tmp/usd_test")
        
        # Result should be the filename
        self.assertEqual(result, "test.usdz")
    
    @patch('tmd.exporters.model.usd.HAS_USD', True)
    @patch('tmd.exporters.model.usd.convert_heightmap_to_usd')
    @patch('pxr.UsdUtils.CreateNewARKitUsdzPackage')
    @patch('os.path.exists', return_value=False)
    def test_usdz_export_failure(self, mock_path_exists, mock_create_usdz, mock_convert_usd):
        """Test USDZ export failure handling."""
        # Setup mocks
        mock_convert_usd.return_value = "temp.usda"
        
        # Test USDZ export with failure (file doesn't exist after creation)
        result = convert_heightmap_to_usdz(
            self.heightmap_flat,
            filename="test.usdz"
        )
        
        # Check that USDZ package was attempted
        mock_create_usdz.assert_called_once()
        
        # Result should be None due to failure
        self.assertIsNone(result)
    
    def test_create_texture_from_heightmap(self):
        """Test heightmap to texture conversion."""
        # Test with OpenCV and matplotlib available
        with patch('cv2.resize', return_value=self.heightmap_flat), \
             patch('cv2.cvtColor', return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
             patch('matplotlib.cm.get_cmap', return_value=lambda x: np.zeros((10, 10, 4))):
            
            texture = _create_texture_from_heightmap(
                self.heightmap_flat,
                resolution=(10, 10)
            )
            
            # Should return a 3-channel image
            self.assertEqual(texture.shape, (10, 10, 3))
            self.assertEqual(texture.dtype, np.uint8)
        
        # Test with ImportError (fallback to grayscale)
        with patch('cv2.cvtColor', return_value=np.zeros((10, 10, 3), dtype=np.uint8)), \
             patch('matplotlib.cm.get_cmap', side_effect=ImportError):
            
            texture = _create_texture_from_heightmap(
                self.heightmap_flat,
                resolution=(10, 10)
            )
            
            # Should still return a 3-channel image
            self.assertEqual(texture.shape, (10, 10, 3))
    
    @unittest.skipIf(not os.getenv('RUN_INTEGRATION_TESTS') or 
                    not hasattr(sys.modules.get('tmd.exporters.model.usd', None), 'HAS_USD') or 
                    not getattr(sys.modules['tmd.exporters.model.usd'], 'HAS_USD', False),
                    "Integration tests disabled or USD modules not available")
    def test_integration_usd_export(self):
        """Integration test for USD export (requires USD modules)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_file = os.path.join(tmp_dir, "test_output.usda")
            
            # Test USD export
            result = convert_heightmap_to_usd(
                self.heightmap_peak,
                filename=output_file,
                z_scale=1.0
            )
            
            # Check that file exists
            self.assertTrue(os.path.isfile(output_file))
            
            # Basic check of file content
            with open(output_file, 'r') as f:
                content = f.read()
                self.assertIn("def Mesh", content)  # Should contain a mesh definition


if __name__ == '__main__':
    unittest.main()
