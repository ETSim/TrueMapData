"""Advanced tests for the TMD utility functions."""
import unittest
import numpy as np
import io
import tempfile
import os
import struct
from tmd.utils.utils import try_read_dimensions, get_header_offset


class TestAdvancedUtilFunctions(unittest.TestCase):
    """Test the advanced utility functions."""
    
    def test_try_read_dimensions(self):
        """Test the try_read_dimensions function."""
        # Create a test binary file with known dimensions
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Write dummy header
            tmp.write(b'X' * 32)
            
            # Write dimensions at offset 32
            tmp.write(struct.pack('<II', 640, 480))
            tmp_file = tmp.name
        
        try:
            # Test reading dimensions
            with open(tmp_file, 'rb') as f:
                f.seek(32)
                dimensions = try_read_dimensions(f, endian='<')
                self.assertEqual(dimensions, (640, 480))
            
            # Test with invalid dimensions
            with open(tmp_file, 'rb') as f:
                f.seek(0)  # Wrong position
                dimensions = try_read_dimensions(f, endian='<')
                self.assertIsNone(dimensions)
            
            # Test with wrong endianness
            with open(tmp_file, 'rb') as f:
                f.seek(32)
                dimensions = try_read_dimensions(f, endian='>')
                self.assertNotEqual(dimensions, (640, 480))
        finally:
            os.unlink(tmp_file)
    
    def test_get_header_offset(self):
        """Test the get_header_offset function."""
        # Test version 1
        offset_v1 = get_header_offset(version=1)
        self.assertEqual(offset_v1, 32)
        
        # Test version 2
        offset_v2 = get_header_offset(version=2)
        self.assertEqual(offset_v2, 64)
        
        # Test fallback behavior for unknown version
        offset_unknown = get_header_offset(version=99)
        self.assertEqual(offset_unknown, 64)  # Should default to v2 behavior


if __name__ == '__main__':
    unittest.main()
