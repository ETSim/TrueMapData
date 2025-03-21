import unittest
import os
import numpy as np
import tempfile
from PIL import Image
import json

from tmd.exporters.image import (
    convert_heightmap_to_displacement_map,
    convert_heightmap_to_normal_map,
    convert_heightmap_to_bump_map,
    generate_roughness_map,
    generate_maps_from_tmd,
    generate_all_maps,
    generate_hillshade
)
from tmd.utils.utils import create_sample_height_map

class TestImageExporter(unittest.TestCase):
    """Test cases for image export functionality."""

    def setUp(self):
        # Create a temporary directory for test outputs
        self.output_dir = tempfile.mkdtemp()
        
        # Create sample height maps for testing
        self.small_height_map = create_sample_height_map(width=20, height=20, pattern="waves")
        
        # Sample metadata for testing
        self.sample_metadata = {
            "normal_strength": 2.0,
            "bump_strength": 1.5,
            "roughness_scale": 1.2,
            "terrain_type": "mountain",
            "units": "µm",
            "x_length": 15.0,
            "y_length": 15.0
        }

    def tearDown(self):
        # Clean up temporary files
        for file in os.listdir(self.output_dir):
            os.remove(os.path.join(self.output_dir, file))
        os.rmdir(self.output_dir)

    def test_displacement_map_generation(self):
        """Test displacement map generation with physical units."""
        output_file = os.path.join(self.output_dir, "displacement.png")
        result = convert_heightmap_to_displacement_map(
            self.small_height_map, 
            filename=output_file, 
            units="µm"
        )
        
        self.assertTrue(os.path.exists(output_file))
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, (20, 20))
        self.assertEqual(result.mode, "L")  # Grayscale mode

    def test_normal_map_generation(self):
        """Test normal map generation with strength parameter."""
        output_file = os.path.join(self.output_dir, "normal.png")
        result = convert_heightmap_to_normal_map(
            self.small_height_map, 
            filename=output_file,
            strength=2.0
        )
        
        self.assertTrue(os.path.exists(output_file))
        self.assertEqual(result.mode, "RGB")  # RGB mode for normal maps

    def test_bump_map_generation(self):
        """Test bump map generation with strength and blur parameters."""
        output_file = os.path.join(self.output_dir, "bump.png")
        result = convert_heightmap_to_bump_map(
            self.small_height_map, 
            filename=output_file,
            strength=1.5,
            blur_radius=0.8
        )
        
        self.assertTrue(os.path.exists(output_file))
        self.assertEqual(result.mode, "L")  # Grayscale mode

    def test_roughness_map_generation(self):
        """Test roughness map generation with scale parameter."""
        # Use significantly different scale values to ensure clear difference
        result = generate_roughness_map(
            self.small_height_map, 
            kernel_size=3,
            scale=2.0  # Increased from 1.5
        )
        
        # Check shape and type
        self.assertEqual(result.shape, self.small_height_map.shape)
        self.assertEqual(result.dtype, np.uint8)
        
        # Higher scale should produce higher values on average
        low_scale_result = generate_roughness_map(self.small_height_map, scale=0.5)
        
        # Add a small delta to account for floating point precision
        mean_high = float(np.mean(result))
        mean_low = float(np.mean(low_scale_result))
        self.assertGreater(mean_high, mean_low, 
                           f"Higher scale should produce higher mean value: {mean_high} vs {mean_low}")

    def test_maps_with_physical_units(self):
        """Test that maps include proper physical dimensions."""
        result = generate_maps_from_tmd(
            self.small_height_map, 
            self.sample_metadata,
            output_dir=self.output_dir
        )
        
        # Check metadata JSON file
        with open(os.path.join(self.output_dir, "map_metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Verify physical dimensions
        dimensions = metadata["physical_dimensions"]
        self.assertEqual(dimensions["width"], 15.0)
        self.assertEqual(dimensions["height"], 15.0)
        self.assertEqual(dimensions["units"], "µm")
        
        # Verify roughness measurement exists
        self.assertIn("roughness", metadata)
        self.assertIn("rms", metadata["roughness"])
        
        # Verify slope measurement exists
        self.assertIn("slope", metadata)
        self.assertIn("max_angle", metadata["slope"])
        
        # Check that files with physical units in the name exist
        roughness_files = [f for f in os.listdir(self.output_dir) if f.startswith("roughness_RMS_")]
        self.assertTrue(any("µm" in f for f in roughness_files))

    def test_hillshade_generation(self):
        """Test hillshade image generation with different light angles."""
        # Default parameters
        output_file = os.path.join(self.output_dir, "hillshade_default.png")
        result = generate_hillshade(
            self.small_height_map, 
            filename=output_file
        )
        
        self.assertTrue(os.path.exists(output_file))
        self.assertEqual(result.mode, "L")  # Grayscale mode
        
        # Test with different light angles
        angles = [(30, 45), (60, 135), (75, 225), (15, 315)]
        for altitude, azimuth in angles:
            output_file = os.path.join(self.output_dir, f"hillshade_alt{altitude}_az{azimuth}.png")
            result = generate_hillshade(
                self.small_height_map, 
                filename=output_file,
                altitude=altitude,
                azimuth=azimuth
            )
            self.assertTrue(os.path.exists(output_file))
        
        # Test with z_factor to exaggerate features
        output_file = os.path.join(self.output_dir, "hillshade_exaggerated.png")
        result = generate_hillshade(
            self.small_height_map, 
            filename=output_file,
            z_factor=3.0
        )
        self.assertTrue(os.path.exists(output_file))
        
        # Verify different light angles produce different images
        images = []
        for altitude, azimuth in angles:
            file_path = os.path.join(self.output_dir, f"hillshade_alt{altitude}_az{azimuth}.png")
            with Image.open(file_path) as img:
                images.append(np.array(img))
        
        # Check that at least some of the images are different from each other
        differences = []
        for i in range(len(images) - 1):
            diff = np.mean(np.abs(images[i].astype(np.float32) - images[i+1].astype(np.float32)))
            differences.append(diff)
        
        # At least one pair of images should be noticeably different
        self.assertTrue(any(d > 5.0 for d in differences), "Light angle changes should affect hillshade appearance")

    def test_all_maps_generation(self):
        """Test the complete map suite generation."""
        maps = generate_all_maps(self.small_height_map, output_dir=self.output_dir)
        
        # Verify essential maps are generated, now including hillshade
        essential_maps = ["displacement", "normal", "roughness", "orm", "edge", "hillshade"]
        for map_type in essential_maps:
            self.assertIn(map_type, maps)
            
        # Check that hillshade file was created
        hillshade_path = os.path.join(self.output_dir, "hillshade.png")
        self.assertTrue(os.path.exists(hillshade_path))
        
        # Check that files were created with appropriate names
        file_count = len(os.listdir(self.output_dir))
        self.assertGreaterEqual(file_count, len(essential_maps))
        
        # Test that map_metadata.json was created and is valid JSON
        metadata_path = os.path.join(self.output_dir, "map_metadata.json")
        self.assertTrue(os.path.exists(metadata_path))
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.assertIsInstance(metadata, dict)

if __name__ == "__main__":
    unittest.main()
