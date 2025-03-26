"""Unit tests for TMD metadata utility module."""

import unittest
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, mock_open, MagicMock

from tmd.utils.metadata import (
    compute_stats,
    export_metadata,
    export_metadata_txt,
    extract_metadata
)
from tmd.utils.utils import create_sample_height_map


class TestMetadataUtility(unittest.TestCase):
    """Test class for metadata utility functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temp directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test height maps with different properties
        self.flat_map = np.ones((10, 15), dtype=np.float32) * 0.5
        self.gradient_map = np.linspace(0, 1, 10*15).reshape(10, 15).astype(np.float32)
        self.sample_map = create_sample_height_map(width=20, height=15, pattern='peak')
        
        # Create a map with NaN values for testing stats with missing data
        self.nan_map = self.sample_map.copy()
        self.nan_map[5:8, 8:12] = np.nan
        
        # Create test metadata
        self.test_metadata = {
            'version': 2,
            'width': 20,
            'height': 15,
            'x_length': 10.0,
            'y_length': 8.0,
            'x_offset': 1.0,
            'y_offset': 0.5,
            'comment': 'Test metadata'
        }
        
        # Define paths for output files
        self.metadata_path = os.path.join(self.temp_dir, 'metadata.txt')
        self.tmd_path = os.path.join(self.temp_dir, 'test.tmd')
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_compute_stats_basic(self):
        """Test computing statistics on a simple height map."""
        # Test with flat map (all values are 0.5)
        stats = compute_stats(self.flat_map)
        
        # Verify basic stats
        self.assertEqual(stats['min'], 0.5)
        self.assertEqual(stats['max'], 0.5)
        self.assertEqual(stats['mean'], 0.5)
        self.assertEqual(stats['median'], 0.5)
        self.assertEqual(stats['std'], 0.0)
        self.assertEqual(stats['shape'], (10, 15))
        self.assertEqual(stats['non_nan'], 150)  # 10x15
        self.assertEqual(stats['nan_count'], 0)
    
    def test_compute_stats_gradient(self):
        """Test computing statistics on a gradient height map."""
        # Test with gradient map (values from 0 to 1)
        stats = compute_stats(self.gradient_map)
        
        # Verify stats for gradient
        self.assertAlmostEqual(stats['min'], 0.0)
        self.assertAlmostEqual(stats['max'], 1.0)
        self.assertAlmostEqual(stats['mean'], 0.5, places=2)
        self.assertAlmostEqual(stats['median'], 0.5, places=2)
        self.assertGreater(stats['std'], 0.0)  # Should have non-zero std dev
        self.assertEqual(stats['shape'], (10, 15))
    
    def test_compute_stats_with_nans(self):
        """Test computing statistics on height map with NaN values."""
        # Test with map containing NaN values
        stats = compute_stats(self.nan_map)
        
        # Count how many NaN values we inserted
        expected_nan_count = 3 * 4  # 3 rows, 4 columns of NaNs
        expected_non_nan_count = 20 * 15 - expected_nan_count
        
        # Verify NaN handling
        self.assertEqual(stats['nan_count'], expected_nan_count)
        self.assertEqual(stats['non_nan'], expected_non_nan_count)
        
        # Stats should ignore NaN values
        self.assertFalse(np.isnan(stats['mean']))
        self.assertFalse(np.isnan(stats['min']))
        self.assertFalse(np.isnan(stats['max']))
    
    def test_export_metadata(self):
        """Test exporting metadata to a text file."""
        # Compute stats for exporting
        stats = compute_stats(self.sample_map)
        
        # Export metadata
        output_path = export_metadata(
            self.test_metadata, 
            stats, 
            self.metadata_path
        )
        
        # Verify output path
        self.assertEqual(output_path, self.metadata_path)
        self.assertTrue(os.path.exists(self.metadata_path))
        
        # Check file content
        with open(self.metadata_path, 'r') as f:
            content = f.read()
            
            # Verify key metadata is in the file
            self.assertIn('version: 2', content)
            self.assertIn('width: 20', content)
            self.assertIn('height: 15', content)
            
            # Verify stats are in the file
            self.assertIn('min:', content)
            self.assertIn('max:', content)
            self.assertIn('mean:', content)
    
    def test_export_metadata_txt(self):
        """Test exporting metadata to a simple text format."""
        # Create test data dictionary with height map
        data_dict = self.test_metadata.copy()
        data_dict['height_map'] = self.sample_map
        
        # Export to text file
        output_path = export_metadata_txt(
            data_dict, 
            filename=self.metadata_path
        )
        
        # Verify output
        self.assertEqual(output_path, self.metadata_path)
        self.assertTrue(os.path.exists(self.metadata_path))
        
        # Check file content
        with open(self.metadata_path, 'r') as f:
            content = f.read()
            
            # Verify key metadata and stats sections
            self.assertIn('TMD File Metadata', content)
            self.assertIn('Height Map Statistics', content)
            self.assertIn('Shape:', content)
            
            # Check that specific metadata values are included
            self.assertIn('version: 2', content)
            self.assertIn('width: 20', content)
    
    @patch('tmd.utils.metadata.TMDProcessor')
    def test_extract_metadata_tmd_file(self, mock_processor):
        """Test extracting metadata from a TMD file."""
        # Setup mock processor
        mock_instance = MagicMock()
        mock_processor.return_value = mock_instance
        mock_instance.process.return_value = {'metadata': self.test_metadata}
        
        # Call the function with a TMD file
        metadata = extract_metadata('test_file.tmd')
        
        # Verify processor was initialized correctly
        mock_processor.assert_called_once_with('test_file.tmd')
        
        # Verify mock process was called
        mock_instance.process.assert_called_once()
        
        # Verify metadata was extracted correctly
        self.assertEqual(metadata, self.test_metadata)
    
    def test_extract_metadata_non_tmd_file(self):
        """Test extracting metadata from a non-TMD file."""
        # Call with a non-TMD file
        metadata = extract_metadata('test_file.txt')
        
        # Should return empty dict
        self.assertEqual(metadata, {})


if __name__ == '__main__':
    unittest.main()
