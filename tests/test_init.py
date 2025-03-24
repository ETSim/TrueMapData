""".

Unit tests for TMD package initialization and version.
"""

import unittest
import importlib


class TestInit(unittest.TestCase):
    """Test cases for the TMD package initialization.."""
    
    def test_version(self):
        """Test that the version is correctly defined.."""
        import tmd
        self.assertTrue(hasattr(tmd, "__version__"))
        self.assertIsInstance(tmd.__version__, str)
        
        # Version should follow semantic versioning format
        import re
        version_pattern = r'^\d+\.\d+\.\d+(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$'
        self.assertTrue(re.match(version_pattern, tmd.__version__), 
                        f"Version '{tmd.__version__}' does not follow semantic versioning")
    
    def test_import_submodules(self):
        """Test that all submodules can be imported without errors.."""
        # Core components
        import tmd.processor
        import tmd.utils
        import tmd.exporters
        import tmd.plotters
        
        # Utility components
        import tmd.utils.filters
        import tmd.utils.metadata
        import tmd.utils.processing
        import tmd.utils.utils
        
        # Exporters
        import tmd.exporters.compression
        import tmd.exporters.image
        import tmd.exporters.model
        import tmd.exporters.shader
        
        # Make sure the imports worked by checking key attributes
        self.assertTrue(hasattr(tmd.processor, "TMDProcessor"))
        self.assertTrue(hasattr(tmd.exporters.model, "convert_heightmap_to_stl"))
        self.assertTrue(hasattr(tmd.utils.processing, "crop_height_map"))


if __name__ == "__main__":
    unittest.main()
