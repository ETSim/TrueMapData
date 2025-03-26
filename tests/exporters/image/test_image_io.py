"""Unit tests for the image IO module."""

import unittest
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

from tmd.exporters.image.image_io import (
    load_heightmap,
    load_image_pil,
    load_image_opencv,
    load_image_npy,
    load_image_npz,
    save_heightmap,
    save_image,
    load_normal_map,
    load_mask,
    _normalize_array,
    ImageType
)


class TestImageIO(unittest.TestCase):
    """Test class for image IO functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample arrays for testing
        self.sample_array_2d = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9]
        ], dtype=np.float32)
        
        self.sample_array_3d = np.array([
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            [[0.7, 0.8, 0.9], [0.1, 0.2, 0.3]]
        ], dtype=np.float32)
        
        # Test file paths
        self.npy_path = os.path.join(self.temp_dir, "test_data.npy")
        self.npz_path = os.path.join(self.temp_dir, "test_data.npz")
        self.png_path = os.path.join(self.temp_dir, "test_data.png")
        self.exr_path = os.path.join(self.temp_dir, "test_data.exr")
        self.normal_path = os.path.join(self.temp_dir, "normal_map.png")
        self.height_path = os.path.join(self.temp_dir, "height_map.png")
        self.mask_path = os.path.join(self.temp_dir, "mask.png")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and its contents
        shutil.rmtree(self.temp_dir)
    
    def test_normalize_array(self):
        """Test normalizing array values."""
        # Test with default parameters
        result = _normalize_array(self.sample_array_2d)
        self.assertEqual(result.dtype, np.float32)
        self.assertAlmostEqual(np.min(result), 0.0)
        self.assertAlmostEqual(np.max(result), 1.0)
        
        # Test with custom range and type
        result = _normalize_array(
            self.sample_array_2d,
            target_type=np.uint8,
            target_range=(0, 255)
        )
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(np.min(result), 0)
        self.assertEqual(np.max(result), 255)
        
        # Test with flat array
        flat_array = np.full((3, 3), 5.0)
        result = _normalize_array(flat_array)
        self.assertEqual(np.mean(result), 0.0)  # Default is 0.0 for uniform
    
    @patch('numpy.save')
    @patch('os.makedirs')
    def test_save_image_npy(self, mock_makedirs, mock_save):
        """Test saving image as NPY file."""
        # Save as NPY file
        save_image(self.sample_array_2d, self.npy_path)
        
        # Check that numpy.save was called
        mock_save.assert_called_once()
        args, _ = mock_save.call_args
        self.assertEqual(args[0], self.npy_path)
        np.testing.assert_array_equal(args[1], self.sample_array_2d)
    
    @patch('numpy.savez_compressed')
    @patch('os.makedirs')
    def test_save_image_npz(self, mock_makedirs, mock_savez):
        """Test saving image as NPZ file."""
        # Save as NPZ file
        save_image(self.sample_array_2d, self.npz_path)
        
        # Check that numpy.savez_compressed was called
        mock_savez.assert_called_once()
        args, kwargs = mock_savez.call_args
        self.assertEqual(args[0], self.npz_path)
        self.assertIn('height_map', kwargs)
        np.testing.assert_array_equal(kwargs['height_map'], self.sample_array_2d)
    
    @patch('tmd.exporters.image.image_io.has_opencv', True)
    @patch('cv2.imwrite')
    @patch('os.makedirs')
    def test_save_image_opencv(self, mock_makedirs, mock_imwrite):
        """Test saving image with OpenCV."""
        # Save with OpenCV
        save_image(self.sample_array_2d, self.png_path)
        
        # Check that cv2.imwrite was called
        mock_imwrite.assert_called_once()
        args, _ = mock_imwrite.call_args
        self.assertEqual(args[0], self.png_path)
    
    @patch('tmd.exporters.image.image_io.has_opencv', False)
    @patch('tmd.exporters.image.image_io.has_pil', True)
    @patch('PIL.Image.fromarray')
    @patch('os.makedirs')
    def test_save_image_pil(self, mock_makedirs, mock_fromarray):
        """Test saving image with PIL."""
        # Set up PIL mock
        mock_img = MagicMock()
        mock_fromarray.return_value = mock_img
        
        # Save with PIL
        save_image(self.sample_array_2d, self.png_path)
        
        # Check that PIL was used
        mock_fromarray.assert_called_once()
        mock_img.save.assert_called_once_with(self.png_path)
    
    @patch('tmd.exporters.image.image_io.has_opencv', False)
    @patch('tmd.exporters.image.image_io.has_pil', False)
    def test_save_image_no_libraries(self):
        """Test saving image with no libraries available."""
        with self.assertRaises(ImportError):
            save_image(self.sample_array_2d, self.png_path)
    
    @patch('os.path.exists')
    def test_load_image_file_not_found(self, mock_exists):
        """Test error handling when file not found."""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            load_image("nonexistent_file.png")
    
    @patch('os.path.exists')
    @patch('numpy.load')
    def test_load_image_npy(self, mock_load, mock_exists):
        """Test loading NPY file."""
        # Set up mocks
        mock_exists.return_value = True
        mock_load.return_value = self.sample_array_2d
        
        # Load NPY file
        result = load_image(self.npy_path)
        
        # Check result
        mock_load.assert_called_once_with(self.npy_path)
        np.testing.assert_array_equal(result, self.sample_array_2d)
    
    @patch('os.path.exists')
    @patch('numpy.load')
    def test_load_image_npz(self, mock_load, mock_exists):
        """Test loading NPZ file."""
        # Set up mocks
        mock_exists.return_value = True
        mock_npz = MagicMock()
        mock_npz.__contains__ = lambda self, key: key == 'height_map'
        mock_npz.__getitem__ = lambda self, key: self.sample_array_2d if key == 'height_map' else None
        mock_load.return_value = mock_npz
        
        # Load NPZ file
        result = load_image(self.npz_path)
        
        # Check result
        mock_load.assert_called_once_with(self.npz_path)
        self.assertIsInstance(result, np.ndarray)
    
    @patch('os.path.exists')
    @patch('tmd.exporters.image.image_io.has_opencv', True)
    @patch('cv2.imread')
    def test_load_image_opencv(self, mock_imread, mock_exists):
        """Test loading image with OpenCV."""
        # Set up mocks
        mock_exists.return_value = True
        mock_imread.return_value = self.sample_array_2d
        
        # Load image
        result = load_image(self.png_path, image_type=ImageType.HEIGHTMAP)
        
        # Check result
        mock_imread.assert_called_once()
        np.testing.assert_array_equal(result, self.sample_array_2d)
    
    @patch('os.path.exists')
    @patch('tmd.exporters.image.image_io.has_opencv', False)
    @patch('tmd.exporters.image.image_io.has_pil', True)
    @patch('PIL.Image.open')
    def test_load_image_pil(self, mock_open, mock_exists):
        """Test loading image with PIL."""
        # Set up mocks
        mock_exists.return_value = True
        mock_img = MagicMock()
        mock_open.return_value = mock_img
        mock_img.mode = 'L'
        mock_img.__array__ = lambda: self.sample_array_2d
        
        # Load image
        result = load_image(self.png_path, image_type=ImageType.HEIGHTMAP)
        
        # Check result
        mock_open.assert_called_once_with(self.png_path)
    
    @patch('os.path.exists')
    @patch('tmd.exporters.image.image_io.has_opencv', False)
    @patch('tmd.exporters.image.image_io.has_pil', False)
    def test_load_image_no_libraries(self, mock_exists):
        """Test loading image with no libraries available."""
        mock_exists.return_value = True
        
        with self.assertRaises(ImportError):
            load_image(self.png_path)
    
    @patch('tmd.exporters.image.image_io.load_image')
    def test_load_heightmap(self, mock_load):
        """Test the load_heightmap wrapper."""
        # Call load_heightmap
        load_heightmap(self.height_path, normalize=True)
        
        # Check that load_image was called correctly
        mock_load.assert_called_once_with(
            self.height_path,
            image_type=ImageType.HEIGHTMAP,
            normalize=True
        )
    
    @patch('tmd.exporters.image.image_io.save_image')
    def test_save_heightmap(self, mock_save):
        """Test the save_heightmap wrapper."""
        # Call save_heightmap
        save_heightmap(self.sample_array_2d, self.height_path, normalize=True)
        
        # Check that save_image was called correctly
        mock_save.assert_called_once_with(
            self.sample_array_2d,
            self.height_path,
            normalize=True
        )
    
    @patch('tmd.exporters.image.image_io.load_image')
    def test_load_normal_map(self, mock_load):
        """Test the load_normal_map wrapper."""
        # Call load_normal_map
        load_normal_map(self.normal_path, normalize=True)
        
        # Check that load_image was called correctly
        mock_load.assert_called_once_with(
            self.normal_path,
            image_type=ImageType.NORMAL_MAP,
            normalize=True
        )
    
    @patch('tmd.exporters.image.image_io.load_image')
    def test_load_mask(self, mock_load):
        """Test the load_mask wrapper."""
        # Set up mock to return a test array
        mock_load.return_value = np.array([[127, 255], [0, 200]])
        
        # Call load_mask
        mask = load_mask(self.mask_path)
        
        # Check that load_image was called correctly
        mock_load.assert_called_once_with(
            self.mask_path,
            image_type=ImageType.MASK,
            normalize=False
        )
        
        # Check the result is a boolean array
        self.assertEqual(mask.dtype, bool)
        np.testing.assert_array_equal(mask, np.array([[0, 1], [0, 1]]))


if __name__ == '__main__':
    unittest.main()
