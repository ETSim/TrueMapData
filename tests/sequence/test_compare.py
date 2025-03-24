"""
Unit tests for the TMDSequenceComparator class.
"""

import unittest
import os
import numpy as np
from unittest.mock import patch, MagicMock

from tmd.sequence.compare import TMDSequenceComparator, compare_heightmaps
from tmd.sequence.sequence import TMDSequence

class TestTMDSequenceComparator(unittest.TestCase):
    """Test TMDSequenceComparator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.comparator = TMDSequenceComparator()
        
        # Create mock sequences
        self.seq1 = TMDSequence(name="Sequence 1")
        self.seq2 = TMDSequence(name="Sequence 2")
        
        # Add sample frames to sequences
        frame1 = np.ones((10, 10))
        frame2 = np.ones((10, 10)) * 2  # Different values
        
        self.seq1.add_frame(frame1, timestamp="Frame 1")
        self.seq1.add_frame(frame1, timestamp="Frame 2")
        
        self.seq2.add_frame(frame2, timestamp="Frame 1")
        self.seq2.add_frame(frame2, timestamp="Frame 2")
        
        # Add sequences to comparator
        self.comparator.add_sequence(self.seq1)
        self.comparator.add_sequence(self.seq2)
    
    def test_add_sequence(self):
        """Test adding a sequence to the comparator."""
        # Test adding a sequence with custom name
        seq3 = TMDSequence(name="Original Name")
        index = self.comparator.add_sequence(seq3, name="Custom Name")
        
        # Verify it was added
        self.assertEqual(index, 2)  # Should be the third sequence
        self.assertEqual(len(self.comparator.sequences), 3)
        self.assertEqual(self.comparator.sequence_names[2], "Custom Name")
        
        # Test adding something that's not a TMDSequence
        invalid_index = self.comparator.add_sequence("Not a sequence")
        self.assertEqual(invalid_index, -1)  # Should indicate failure
        self.assertEqual(len(self.comparator.sequences), 3)  # Count shouldn't change
    
    def test_calculate_frame_wise_differences(self):
        """Test calculating frame-wise differences between sequences."""
        # Calculate differences
        differences = self.comparator.calculate_frame_wise_differences()
        
        # Should have differences for one pair of sequences
        self.assertEqual(len(differences), 1)  # Number of pairs
        
        # Get the pair key (should be (0, 1))
        pair_key = (0, 1)
        self.assertIn(pair_key, differences)
        
        # Get differences for the pair
        pair_diffs = differences[pair_key]
        
        # Should have differences for both frames
        self.assertEqual(len(pair_diffs), 2)
        
        # Each difference should be: seq2 - seq1 = 2 - 1 = 1
        np.testing.assert_array_equal(pair_diffs[0], np.ones((10, 10)))
        np.testing.assert_array_equal(pair_diffs[1], np.ones((10, 10)))
    
    def test_calculate_statistical_differences(self):
        """Test calculating statistical differences between sequences."""
        # Force sequences to have statistics
        with patch.object(self.seq1, 'calculate_statistics') as mock1:
            with patch.object(self.seq2, 'calculate_statistics') as mock2:
                # Set up mock return values
                mock1.return_value = {
                    'timestamps': ["Frame 1", "Frame 2"],
                    'min': [0.8, 0.9],
                    'max': [1.2, 1.1],
                    'mean': [1.0, 1.0],
                    'std': [0.1, 0.05]
                }
                
                mock2.return_value = {
                    'timestamps': ["Frame 1", "Frame 2"],
                    'min': [1.8, 1.9],
                    'max': [2.2, 2.1],
                    'mean': [2.0, 2.0],
                    'std': [0.1, 0.05]
                }
                
                # Calculate differences
                stat_diffs = self.comparator.calculate_statistical_differences()
                
                # Should have differences for one pair of sequences
                self.assertEqual(len(stat_diffs), 1)  # Number of pairs
                
                # Check pair key (should be (0, 1))
                pair_key = (0, 1)
                self.assertIn(pair_key, stat_diffs)
                
                # Get differences for the pair
                pair_stats = stat_diffs[pair_key]
                
                # Should have difference stats for each measure
                self.assertIn('mean_abs_diff', pair_stats)
                self.assertIn('mean_rel_diff', pair_stats)
                
                # Check values: mean diff = seq2_mean - seq1_mean = 2.0 - 1.0 = 1.0
                np.testing.assert_array_equal(pair_stats['mean_abs_diff'], [1.0, 1.0])
    
    def test_visualize_frame_differences(self):
        """Test visualizing frame differences."""
        # Mock the matplotlib functions
        with patch('matplotlib.pyplot.figure'):
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.close'):
                    # Mock calculate_frame_wise_differences to return known data
                    with patch.object(self.comparator, 'calculate_frame_wise_differences') as mock_calc:
                        mock_calc.return_value = {
                            (0, 1): [np.ones((10, 10))]
                        }
                        
                        # Call the visualization function
                        figures = self.comparator.visualize_frame_differences(
                            output_dir="/tmp",
                            show=False
                        )
                        
                        # Should have created at least one figure
                        self.assertGreater(len(figures), 0)
    
    def test_visualize_statistical_comparison(self):
        """Test visualizing statistical comparisons."""
        # Mock the matplotlib functions
        with patch('matplotlib.pyplot.figure'):
            with patch('matplotlib.pyplot.savefig'):
                with patch('matplotlib.pyplot.close'):
                    # Mock calculate_statistical_differences to return known data
                    with patch.object(self.comparator, 'calculate_statistical_differences') as mock_calc:
                        mock_calc.return_value = {
                            (0, 1): {
                                'timestamps': ["Frame 1", "Frame 2"],
                                'mean_abs_diff': [1.0, 1.0],
                                'mean_rel_diff': [100.0, 100.0],
                                'std_abs_diff': [0.0, 0.0],
                                'std_rel_diff': [0.0, 0.0]
                            }
                        }
                        
                        # Call the visualization function
                        figures = self.comparator.visualize_statistical_comparison(
                            output_dir="/tmp",
                            metrics=['mean', 'std'],
                            show=False
                        )
                        
                        # Should have created at least one figure
                        self.assertGreater(len(figures), 0)
    
    def test_export_difference_report(self):
        """Test exporting difference report."""
        # Mock pandas and ExcelWriter
        with patch('pandas.DataFrame'):
            with patch('pandas.ExcelWriter') as mock_writer:
                # Create a mock writer with a close method
                writer_instance = MagicMock()
                mock_writer.return_value = writer_instance
                
                # Mock calculation methods to return valid data
                with patch.object(self.comparator, 'calculate_frame_wise_differences') as mock_frame:
                    with patch.object(self.comparator, 'calculate_statistical_differences') as mock_stats:
                        mock_frame.return_value = {
                            (0, 1): [np.ones((10, 10))]
                        }
                        mock_stats.return_value = {
                            (0, 1): {
                                'timestamps': ["Frame 1"],
                                'mean_abs_diff': [1.0],
                                'mean_rel_diff': [100.0]
                            }
                        }
                        
                        # Call export function
                        result = self.comparator.export_difference_report(
                            output_file="/tmp/report.xlsx"
                        )
                        
                        # Should have returned the path
                        self.assertIsNotNone(result)
                        
                        # Verify writer was used
                        self.assertTrue(writer_instance.close.called)

    def test_len(self):
        """Test the __len__ method."""
        self.assertEqual(len(self.comparator), 2)
        
        # Add another sequence
        self.comparator.add_sequence(TMDSequence())
        self.assertEqual(len(self.comparator), 3)

class TestCompareHeightmaps(unittest.TestCase):
    """Test compare_heightmaps function."""
    
    def test_compare_identical_maps(self):
        """Test comparing identical heightmaps."""
        # Create identical maps
        map1 = np.ones((10, 10))
        map2 = np.ones((10, 10))
        
        # Compare
        result = compare_heightmaps(map1, map2)
        
        # Check structure and some values
        self.assertIn('difference', result)
        self.assertIn('metrics', result)
        
        # Difference should be zeros
        np.testing.assert_array_equal(result['difference'], np.zeros((10, 10)))
        
        # RMSE should be zero
        self.assertEqual(result['metrics']['rmse'], 0.0)
    
    def test_compare_different_maps(self):
        """Test comparing different heightmaps."""
        # Create maps with a difference of 1.0
        map1 = np.ones((10, 10))
        map2 = np.ones((10, 10)) * 2.0
        
        # Compare
        result = compare_heightmaps(map1, map2)
        
        # Check difference is 1.0 everywhere
        np.testing.assert_array_equal(result['difference'], np.ones((10, 10)))
        
        # RMSE should be 1.0
        self.assertEqual(result['metrics']['rmse'], 1.0)
    
    def test_compare_different_shapes(self):
        """Test comparing heightmaps with different shapes."""
        # Create maps with different shapes
        map1 = np.ones((10, 10))
        map2 = np.ones((12, 8))
        
        # Compare
        result = compare_heightmaps(map1, map2)
        
        # Should have cropped to smaller dimensions
        self.assertEqual(result['difference'].shape, (10, 8))

if __name__ == "__main__":
    unittest.main()
