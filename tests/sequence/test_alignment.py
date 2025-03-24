"""
Tests for the TMD sequence alignment functionality.
"""

import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory

from tmd.sequence.alignement import (
    align_heightmaps,
    align_sequence_to_reference,
    visualize_alignment
)

# Helper function to create test data
def create_test_heightmap(shape=(64, 64), offset=(0, 0), pattern="slope"):
    """Create a test heightmap with various patterns."""
    height_map = np.zeros(shape, dtype=np.float32)
    
    if pattern == "slope":
        # Create a sloped surface
        x = np.linspace(0, 1, shape[1])
        y = np.linspace(0, 1, shape[0])
        X, Y = np.meshgrid(x, y)
        height_map = X + Y
    elif pattern == "peak":
        # Create a central peak
        x = np.linspace(-3, 3, shape[1])
        y = np.linspace(-3, 3, shape[0])
        X, Y = np.meshgrid(x, y)
        height_map = np.exp(-(X**2 + Y**2)/2)
    elif pattern == "random":
        # Random noise with fixed seed for reproducibility
        np.random.seed(42)
        height_map = np.random.random(shape)
        
    # Apply offset by rolling the array
    if offset[0] != 0 or offset[1] != 0:
        height_map = np.roll(height_map, offset, axis=(0, 1))
        
    return height_map

class TestAlignHeightmaps:
    """Tests for the align_heightmaps function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a source and target heightmap with an offset
        self.source = create_test_heightmap(shape=(64, 64), pattern="peak")
        self.target = create_test_heightmap(shape=(64, 64), offset=(5, 3), pattern="peak")
        
        # Create a mock for OpenCV
        self.cv2_patcher = patch('tmd.sequence.alignement.cv2')
        self.mock_cv2 = self.cv2_patcher.start()
        
        # Set up return values for OpenCV functions
        self.mock_cv2.MOTION_TRANSLATION = 0
        self.mock_cv2.TERM_CRITERIA_EPS = 1
        self.mock_cv2.TERM_CRITERIA_COUNT = 2
        self.mock_cv2.INTER_CUBIC = 2
        self.mock_cv2.WARP_INVERSE_MAP = 16
        self.mock_cv2.BORDER_REPLICATE = 1
        self.mock_cv2.INTER_AREA = 3
        
        # Make warpAffine return the input (mock implementation)
        self.mock_cv2.warpAffine.side_effect = lambda src, matrix, size, **kwargs: src
        self.mock_cv2.resize.side_effect = lambda img, size, **kwargs: img
        
        # Create a transform matrix for ECC
        self.warp_matrix = np.eye(2, 3, dtype=np.float32)
        self.warp_matrix[0, 2] = 5  # x translation
        self.warp_matrix[1, 2] = 3  # y translation
        
        # Mock findTransformECC
        self.mock_cv2.findTransformECC.return_value = (None, self.warp_matrix)
        
    def teardown_method(self):
        """Clean up after test."""
        self.cv2_patcher.stop()
    
    def test_align_with_ecc(self):
        """Test alignment using ECC method."""
        # Test the function
        aligned, transform = align_heightmaps(
            self.source, 
            self.target, 
            method='ecc'
        )
        
        # Verify the shape and transformation matrix
        assert aligned.shape == self.target.shape
        assert np.allclose(transform, self.warp_matrix)
        
        # Verify OpenCV was called correctly
        self.mock_cv2.findTransformECC.assert_called_once()
    
    def test_align_with_orb(self):
        """Test alignment using ORB features."""
        # Mock the ORB feature detector
        mock_detector = MagicMock()
        self.mock_cv2.ORB_create.return_value = mock_detector
        
        # Set up mock for feature detection
        keypoints1 = [MagicMock() for _ in range(5)]
        keypoints2 = [MagicMock() for _ in range(5)]
        descriptors1 = np.random.randn(5, 32).astype(np.float32)
        descriptors2 = np.random.randn(5, 32).astype(np.float32)
        
        # Configure keypoints
        for i, kp in enumerate(keypoints1):
            kp.pt = (i*10, i*5)
        for i, kp in enumerate(keypoints2):
            kp.pt = (i*10+5, i*5+3)
            
        # Configure detectAndCompute to return keypoints and descriptors
        mock_detector.detectAndCompute.side_effect = [
            (keypoints1, descriptors1),
            (keypoints2, descriptors2)
        ]
        
        # Mock the matcher
        mock_matcher = MagicMock()
        self.mock_cv2.BFMatcher_create.return_value = mock_matcher
        
        # Set up matches
        good_match = MagicMock()
        good_match.queryIdx = 0
        good_match.trainIdx = 0
        good_match.distance = 10
        
        another_match = MagicMock()
        another_match.distance = 20
        
        # Create 5 pairs of matches
        mock_matcher.knnMatch.return_value = [(good_match, another_match) for _ in range(5)]
        
        # Mock homography estimation
        self.mock_cv2.findHomography.return_value = (self.warp_matrix, None)
        
        # Test the alignment with ORB
        aligned, transform = align_heightmaps(
            self.source, 
            self.target, 
            method='orb'
        )
        
        # Verify results
        assert aligned.shape == self.target.shape
        assert self.mock_cv2.ORB_create.called
        assert mock_detector.detectAndCompute.call_count == 2
        assert mock_matcher.knnMatch.called
    
    def test_align_with_template(self):
        """Test alignment using template matching."""
        # Mock template matching
        result = np.zeros((10, 10), dtype=np.float32)
        result[5, 3] = 1.0  # Max at (5, 3)
        self.mock_cv2.matchTemplate.return_value = result
        self.mock_cv2.minMaxLoc.return_value = (0, 1.0, (0, 0), (3, 5))
        
        aligned, transform = align_heightmaps(
            self.source, 
            self.target, 
            method='template'
        )
        
        # Verify the shape of results
        assert aligned.shape == self.target.shape
        assert transform.shape == (2, 3)
        
        # Check if the transform matrix contains the expected translation
        assert transform[0, 2] == 3
        assert transform[1, 2] == 5
        
    def test_align_different_shapes(self):
        """Test alignment with heightmaps of different shapes."""
        # Create a larger target heightmap
        larger_target = create_test_heightmap(shape=(80, 80), offset=(10, 5), pattern="peak")
        
        # Test the alignment
        aligned, transform = align_heightmaps(self.source, larger_target)
        
        # Verify the mock resize was called
        self.mock_cv2.resize.assert_called_once()
        
        # Verify warpAffine was called with the right dimensions
        _, args, _ = self.mock_cv2.warpAffine.mock_calls[0]
        assert args[2] == (larger_target.shape[1], larger_target.shape[0])
    
    def test_align_with_unknown_method(self):
        """Test alignment with an unknown method."""
        # Test with an invalid method
        aligned, transform = align_heightmaps(
            self.source, 
            self.target, 
            method='invalid_method'
        )
        
        # Should default to identity transform
        assert np.allclose(transform, np.eye(2, 3))
        assert aligned.shape == self.target.shape


class TestAlignSequence:
    """Tests for the align_sequence_to_reference function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create test sequences with different offsets
        self.source_sequence = [
            create_test_heightmap(shape=(64, 64), offset=(0, 0), pattern="peak"),
            create_test_heightmap(shape=(64, 64), offset=(1, 1), pattern="peak"),
            create_test_heightmap(shape=(64, 64), offset=(2, 2), pattern="peak")
        ]
        
        self.reference_sequence = [
            create_test_heightmap(shape=(64, 64), offset=(5, 3), pattern="peak"),
            create_test_heightmap(shape=(64, 64), offset=(6, 4), pattern="peak"),
            create_test_heightmap(shape=(64, 64), offset=(7, 5), pattern="peak")
        ]
        
    def test_align_sequence_all_frames(self):
        """Test aligning a sequence using all frames."""
        # Mock the align_heightmaps function to return a known transformation
        with patch('tmd.sequence.alignement.align_heightmaps') as mock_align:
            # Set up mock to return aligned heightmap and transformation
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            warp_matrix[0, 2] = 5  # x translation
            warp_matrix[1, 2] = 3  # y translation
            
            mock_align.side_effect = lambda src, target, **kwargs: (src, warp_matrix)
            
            # Test the function
            aligned_seq, transforms = align_sequence_to_reference(
                self.source_sequence, 
                self.reference_sequence
            )
            
            # Verify results
            assert len(aligned_seq) == len(self.source_sequence)
            assert len(transforms) == len(self.source_sequence)
            assert all(np.allclose(t, warp_matrix) for t in transforms)
            
            # Verify align_heightmaps was called for each frame
            assert mock_align.call_count == len(self.source_sequence)
            
    def test_align_sequence_specific_frames(self):
        """Test aligning a sequence using specific frame indices."""
        # Mock the align_heightmaps function
        with patch('tmd.sequence.alignement.align_heightmaps') as mock_align:
            # Set up mock
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            warp_matrix[0, 2] = 5
            mock_align.side_effect = lambda src, target, **kwargs: (src, warp_matrix)
            
            # Test with specific indices
            aligned_seq, transforms = align_sequence_to_reference(
                self.source_sequence,
                self.reference_sequence,
                frame_indices=[0, 2]  # Only use first and last frame
            )
            
            # Verify mock was called with the right frames
            assert mock_align.call_count == 2
            
            # Verify all frames were transformed with the average matrix
            assert len(aligned_seq) == len(self.source_sequence)
            assert len(transforms) == len(self.source_sequence)
            
    def test_align_sequence_invalid_indices(self):
        """Test aligning with invalid frame indices."""
        with patch('tmd.sequence.alignement.align_heightmaps') as mock_align:
            mock_align.side_effect = lambda src, target, **kwargs: (src, np.eye(2, 3))
            
            # Test with out of bounds indices
            aligned_seq, transforms = align_sequence_to_reference(
                self.source_sequence,
                self.reference_sequence,
                frame_indices=[-1, 100]  # Invalid indices
            )
            
            # Should not call align_heightmaps since no valid indices
            assert mock_align.call_count == 0
            
            # Should return source sequence with identity transforms
            assert len(aligned_seq) == len(self.source_sequence)
            assert all(np.allclose(t, np.eye(2, 3)) for t in transforms)
            
    def test_align_sequence_empty(self):
        """Test aligning with empty sequences."""
        # Test with empty source sequence
        empty_source = []
        aligned_seq, transforms = align_sequence_to_reference(
            empty_source, 
            self.reference_sequence
        )
        
        # Should return empty lists
        assert aligned_seq == empty_source
        assert transforms == []
        
        # Test with empty reference sequence
        empty_reference = []
        aligned_seq, transforms = align_sequence_to_reference(
            self.source_sequence, 
            empty_reference
        )
        
        # Should return source sequence with identity transforms
        assert len(aligned_seq) == len(self.source_sequence)
        assert len(transforms) == len(self.source_sequence)
        assert all(np.allclose(t, np.eye(2, 3)) for t in transforms)


class TestVisualizeAlignment:
    """Tests for the visualize_alignment function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.source = create_test_heightmap(shape=(64, 64), pattern="peak")
        self.target = create_test_heightmap(shape=(64, 64), offset=(5, 3), pattern="peak")
        self.aligned = create_test_heightmap(shape=(64, 64), offset=(5, 3), pattern="peak")
        
        # Create a temporary directory for test outputs
        self.temp_dir = TemporaryDirectory()
        
    def teardown_method(self):
        """Clean up after test."""
        self.temp_dir.cleanup()
        
    def test_visualization(self):
        """Test the visualization function."""
        # Create output path in temporary directory
        output_path = os.path.join(self.temp_dir.name, "alignment_test.png")
        
        # Mock plt.show to avoid displaying during tests
        with patch('matplotlib.pyplot.show'):
            # Test the visualization function
            fig = visualize_alignment(
                self.source, 
                self.target, 
                self.aligned,
                output_path=output_path,
                show=True
            )
            
            # Verify the figure was created and saved
            assert isinstance(fig, plt.Figure)
            assert os.path.exists(output_path)
            
            # Verify the figure has the expected subplots
            assert len(fig.axes) == 3
            
            # Close the figure to release memory
            plt.close(fig)
    
    def test_visualization_no_output(self):
        """Test visualization without saving to file."""
        # Mock plt.show to avoid displaying during tests
        with patch('matplotlib.pyplot.show'):
            # Test without output path
            fig = visualize_alignment(
                self.source, 
                self.target, 
                self.aligned,
                show=True
            )
            
            # Verify the figure was created
            assert isinstance(fig, plt.Figure)
            
            # Close the figure
            plt.close(fig)
