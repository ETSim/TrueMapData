"""
Unit tests for the TMDSequence class.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

from tmd.sequence.sequence import TMDSequence

class TestTMDSequence(unittest.TestCase):
    """Test TMDSequence class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sequence = TMDSequence(name="Test Sequence")
        
        # Create sample frames
        self.frame1 = np.ones((10, 10))
        self.frame2 = np.ones((10, 10)) * 2
        self.frame3 = np.ones((10, 10)) * 3
        
        # Add frames to sequence
        self.sequence.add_frame(self.frame1, timestamp="Frame 1")
        self.sequence.add_frame(self.frame2, timestamp="Frame 2")
        self.sequence.add_frame(self.frame3, timestamp="Frame 3")
    
    def test_add_frame(self):
        """Test adding a frame to the sequence."""
        # Test adding a frame with metadata
        new_frame = np.ones((10, 10)) * 4
        metadata = {"key": "value"}
        
        index = self.sequence.add_frame(new_frame, timestamp="Frame 4", metadata=metadata)
        
        # Check that frame was added correctly
        self.assertEqual(index, 3)  # Should be 4th frame (index 3)
        self.assertEqual(len(self.sequence.frames), 4)
        self.assertEqual(len(self.sequence.frame_timestamps), 4)
        
        # Add empty/null frame
        null_index = self.sequence.add_frame(None)
        self.assertEqual(null_index, -1)  # Should indicate failure
        self.assertEqual(len(self.sequence.frames), 4)  # Count shouldn't change
    
    def test_get_frame(self):
        """Test retrieving a frame by index."""
        # Get an existing frame
        frame = self.sequence.get_frame(1)
        np.testing.assert_array_equal(frame, self.frame2)
        
        # Try getting a non-existent frame
        none_frame = self.sequence.get_frame(10)
        self.assertIsNone(none_frame)
    
    def test_get_timestamp(self):
        """Test retrieving a timestamp by index."""
        # Get an existing timestamp
        ts = self.sequence.get_timestamp(0)
        self.assertEqual(ts, "Frame 1")
        
        # Try getting a non-existent timestamp
        none_ts = self.sequence.get_timestamp(10)
        self.assertIsNone(none_ts)
    
    def test_get_all_timestamps(self):
        """Test retrieving all timestamps."""
        timestamps = self.sequence.get_all_timestamps()
        self.assertEqual(timestamps, ["Frame 1", "Frame 2", "Frame 3"])
        
        # Ensure it's a copy
        timestamps.append("New timestamp")
        self.assertEqual(len(self.sequence.frame_timestamps), 3)
    
    def test_get_all_frames(self):
        """Test retrieving all frames."""
        frames = self.sequence.get_all_frames()
        self.assertEqual(len(frames), 3)
        
        # Ensure it's a copy
        frames.append(np.ones((10, 10)))
        self.assertEqual(len(self.sequence.frames), 3)
    
    def test_len(self):
        """Test the __len__ method."""
        self.assertEqual(len(self.sequence), 3)
        
        # Add a frame
        self.sequence.add_frame(np.ones((10, 10)))
        self.assertEqual(len(self.sequence), 4)
    
    def test_getitem(self):
        """Test the __getitem__ method."""
        frame = self.sequence[1]
        np.testing.assert_array_equal(frame, self.frame2)
        
        # Test out of bounds
        with self.assertRaises(IndexError):
            _ = self.sequence[10]
    
    def test_calculate_statistics(self):
        """Test calculating statistics."""
        # Calculate statistics
        stats = self.sequence.calculate_statistics()
        
        # Check that we have expected keys and correct values
        self.assertIn('min', stats)
        self.assertIn('max', stats)
        self.assertIn('mean', stats)
        self.assertEqual(len(stats['min']), 3)
        self.assertEqual(len(stats['max']), 3)
        self.assertEqual(len(stats['mean']), 3)
        
        # Check values (frame1=1, frame2=2, frame3=3)
        self.assertEqual(stats['min'][0], 1.0)
        self.assertEqual(stats['max'][1], 2.0)
        self.assertEqual(stats['mean'][2], 3.0)
        
        # Handle empty sequence
        empty_seq = TMDSequence(name="Empty")
        empty_stats = empty_seq.calculate_statistics()
        self.assertEqual(empty_stats, {})
    
    def test_calculate_differences(self):
        """Test calculating differences between frames."""
        # Calculate differences
        diffs = self.sequence.calculate_differences()
        
        # Should be 2 differences (between 3 frames)
        self.assertEqual(len(diffs), 2)
        
        # diff1 = frame2 - frame1 = 2 - 1 = 1
        np.testing.assert_array_equal(diffs[0], np.ones((10, 10)))
        
        # diff2 = frame3 - frame2 = 3 - 2 = 1
        np.testing.assert_array_equal(diffs[1], np.ones((10, 10)))
        
        # Handle sequence with fewer than 2 frames
        short_seq = TMDSequence(name="Short")
        short_seq.add_frame(np.ones((10, 10)))
        short_diffs = short_seq.calculate_differences()
        self.assertEqual(short_diffs, [])

    def test_set_and_get_transformation(self):
        """Test setting and getting transformations."""
        # Create a transformation
        transform = {"rotation": 90, "scale": 1.5}
        
        # Set transformation
        success = self.sequence.set_transformation(1, transform)
        self.assertTrue(success)
        
        # Get transformation
        stored_transform = self.sequence.get_transformation(1)
        self.assertEqual(stored_transform, transform)
        
        # Try invalid index
        invalid_success = self.sequence.set_transformation(10, transform)
        self.assertFalse(invalid_success)
        invalid_transform = self.sequence.get_transformation(10)
        self.assertIsNone(invalid_transform)

    def test_apply_transformations(self):
        """Test applying transformations to frames."""
        # Set up transformations
        self.sequence.set_transformation(0, {"scaling": [1.0, 1.0, 2.0]})  # Double z height
        self.sequence.set_transformation(1, {"scaling": [1.0, 1.0, 3.0]})  # Triple z height
        
        # Transform
        transformed_frames = self.sequence.apply_transformations()
        
        # frame1: 1 * 2 = 2
        np.testing.assert_array_equal(transformed_frames[0], np.ones((10, 10)) * 2)
        
        # frame2: 2 * 3 = 6
        np.testing.assert_array_equal(transformed_frames[1], np.ones((10, 10)) * 6)

if __name__ == "__main__":
    unittest.main()
