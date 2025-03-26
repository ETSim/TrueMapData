"""Tests for the TMD processor module."""

import unittest
import os
import tempfile
import shutil
import numpy as np
from unittest.mock import patch, MagicMock

from tmd.processor import TMDProcessor

class TestTMDProcessor(unittest.TestCase):
    """Test the TMDProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock TMD file for testing
        self.test_file = os.path.join(self.temp_dir, 'test_file.tmd')
        with open(self.test_file, 'wb') as f:
            # Write a minimal TMD header
            f.write(b'TMD\0')  # magic
            f.write((1).to_bytes(4, byteorder='little'))  # version
            f.write((10).to_bytes(4, byteorder='little'))  # width
            f.write((10).to_bytes(4, byteorder='little'))  # height
            # Add comment field
            comment = b'Test file\0'
            f.write(len(comment).to_bytes(4, byteorder='little'))  # comment length
            f.write(comment)
            # Add some dummy height data
            height_data = np.ones((10, 10), dtype=np.float32) * 0.1
            f.write(height_data.tobytes())
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and all its content
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test TMDProcessor initialization."""
        processor = TMDProcessor(self.test_file)
        self.assertEqual(processor.filepath, self.test_file)
        self.assertEqual(processor.version, 1)
        self.assertIsNone(processor.height_map)
        self.assertFalse(processor.debug)
        
    def test_set_debug(self):
        """Test setting debug mode."""
        processor = TMDProcessor(self.test_file)
        
        # Test default value
        self.assertFalse(processor.debug)
        
        # Test setting to True
        result = processor.set_debug(True)
        self.assertTrue(processor.debug)
        self.assertEqual(result, processor, "Method should return self for chaining")
        
        # Test setting to False
        result = processor.set_debug(False)
        self.assertFalse(processor.debug)
        self.assertEqual(result, processor, "Method should return self for chaining")
    
    def test_print_file_header(self):
        """Test printing file header."""
        processor = TMDProcessor(self.test_file)
        header_info = processor.print_file_header()
        
        # Verify header information
        self.assertEqual(header_info["magic"], "TMD")
        self.assertEqual(header_info["version"], 1)
        self.assertEqual(header_info["width"], 10)
        self.assertEqual(header_info["height"], 10)
    
    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        non_existent_file = os.path.join(self.temp_dir, 'non_existent.tmd')
        with self.assertRaises(FileNotFoundError):
            TMDProcessor(non_existent_file)
    
    def test_process(self):
        """Test processing the TMD file."""
        processor = TMDProcessor(self.test_file)
        result = processor.process()
        
        # Check that the result contains expected keys
        self.assertIn("metadata", result)
        self.assertIn("height_map", result)
        
        # Check some metadata values
        self.assertEqual(result["metadata"]["version"], 1)
        self.assertEqual(result["metadata"]["width"], 10)
        self.assertEqual(result["metadata"]["height"], 10)
        self.assertEqual(result["metadata"]["comment"], "Test file")
        
        # Check height map dimensions and values
        self.assertEqual(result["height_map"].shape, (10, 10))
        np.testing.assert_allclose(result["height_map"], np.ones((10, 10)) * 0.1)
    
    def test_get_methods(self):
        """Test getting height map, metadata, and stats."""
        processor = TMDProcessor(self.test_file)
        
        # Test before processing
        height_map = processor.get_height_map()
        self.assertIsNotNone(height_map)
        self.assertEqual(height_map.shape, (10, 10))
        
        # Get and check metadata
        metadata = processor.get_metadata()
        self.assertEqual(metadata["version"], 1)
        self.assertEqual(metadata["comment"], "Test file")
        
        # Get stats
        stats = processor.get_stats()
        self.assertIn("min", stats)
        self.assertIn("max", stats)
        self.assertIn("mean", stats)
        self.assertIn("median", stats)
        self.assertIn("std", stats)
        self.assertIn("shape", stats)
        self.assertEqual(stats["min"], 0.1)
        self.assertEqual(stats["max"], 0.1)
        
    @patch('tmd.utils.metadata.export_metadata')
    def test_export_metadata(self, mock_export_metadata):
        """Test exporting metadata to file."""
        mock_export_metadata.return_value = "output_path.txt"
        
        processor = TMDProcessor(self.test_file)
        result = processor.export_metadata("output_path.txt")
        
        # Check that export_metadata was called
        mock_export_metadata.assert_called_once()
        self.assertEqual(result, "output_path.txt")
        
        # Test with default output path
        processor.export_metadata()
        self.assertEqual(mock_export_metadata.call_count, 2)

if __name__ == '__main__':
    unittest.main()
