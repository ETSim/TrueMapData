"""Unit tests for TMD files utility module."""

import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import time

from tmd.utils.files import (
    generate_unique_filename,
    list_files_with_extension,
    ensure_directory_exists,
    get_filename_without_extension,
    get_directory_from_filepath,
    find_files_by_pattern
)


class TestFilesUtility(unittest.TestCase):
    """Test class for files utility functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test file paths
        self.test_file = os.path.join(self.temp_dir, "test_file.txt")
        self.test_tmd_file = os.path.join(self.temp_dir, "test_data.tmd")
        self.test_png_file = os.path.join(self.temp_dir, "test_image.png")
        
        # Create a subdirectory
        self.sub_dir = os.path.join(self.temp_dir, "subdir")
        os.makedirs(self.sub_dir)
        
        # Create some test files
        with open(self.test_file, "w") as f:
            f.write("Test content")
        
        with open(self.test_tmd_file, "w") as f:
            f.write("TMD test content")
        
        with open(self.test_png_file, "w") as f:
            f.write("PNG test content")
            
        # Create a file in the subdirectory
        self.sub_file = os.path.join(self.sub_dir, "sub_file.tmd")
        with open(self.sub_file, "w") as f:
            f.write("Subdirectory TMD file")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory and all its contents
        shutil.rmtree(self.temp_dir)
    
    def test_generate_unique_filename(self):
        """Test generating unique filenames."""
        # Test with non-existent file (should return original)
        non_existent = os.path.join(self.temp_dir, "nonexistent.txt")
        result = generate_unique_filename(non_existent)
        self.assertEqual(result, non_existent)
        
        # Test with existing file (should append _1)
        result = generate_unique_filename(self.test_file)
        expected = os.path.join(self.temp_dir, "test_file_1.txt")
        self.assertEqual(result, expected)
        
        # Test with file that already has a numeric suffix
        with open(os.path.join(self.temp_dir, "document_5.txt"), "w") as f:
            f.write("Test")
        
        result = generate_unique_filename(os.path.join(self.temp_dir, "document_5.txt"))
        expected = os.path.join(self.temp_dir, "document_1.txt")
        self.assertEqual(result, expected)
        
        # Test with multiple existing files
        for i in range(1, 4):
            with open(os.path.join(self.temp_dir, f"multi_{i}.txt"), "w") as f:
                f.write("Test")
                
        result = generate_unique_filename(os.path.join(self.temp_dir, "multi_1.txt"))
        expected = os.path.join(self.temp_dir, "multi_4.txt")
        self.assertEqual(result, expected)
    
    def test_list_files_with_extension(self):
        """Test listing files with specific extensions."""
        # Test with .tmd extension, non-recursive
        tmd_files = list_files_with_extension(self.temp_dir, ".tmd")
        self.assertEqual(len(tmd_files), 1)
        self.assertEqual(os.path.basename(tmd_files[0]), "test_data.tmd")
        
        # Test with extension without dot
        tmd_files2 = list_files_with_extension(self.temp_dir, "tmd")
        self.assertEqual(tmd_files, tmd_files2)
        
        # Test with recursive search
        all_tmd_files = list_files_with_extension(self.temp_dir, ".tmd", recursive=True)
        self.assertEqual(len(all_tmd_files), 2)
        self.assertTrue(any("subdir" in f for f in all_tmd_files))
        
        # Test with non-existent extension
        no_files = list_files_with_extension(self.temp_dir, ".xyz")
        self.assertEqual(len(no_files), 0)
    
    def test_ensure_directory_exists(self):
        """Test ensuring directories exist."""
        # Test with existing directory
        result = ensure_directory_exists(self.temp_dir)
        self.assertTrue(result)
        
        # Test with new directory
        new_dir = os.path.join(self.temp_dir, "new_directory")
        result = ensure_directory_exists(new_dir)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(new_dir))
        
        # Test with nested directories
        nested_dir = os.path.join(self.temp_dir, "level1", "level2", "level3")
        result = ensure_directory_exists(nested_dir)
        self.assertTrue(result)
        self.assertTrue(os.path.exists(nested_dir))
        
        # Test with invalid path (simulate failure)
        with patch("os.makedirs") as mock_makedirs:
            mock_makedirs.side_effect = PermissionError("Access denied")
            result = ensure_directory_exists("/invalid/path")
            self.assertFalse(result)
    
    def test_get_filename_without_extension(self):
        """Test extracting filename without extension."""
        # Test with simple filename
        result = get_filename_without_extension(self.test_file)
        self.assertEqual(result, "test_file")
        
        # Test with path containing directories
        result = get_filename_without_extension(self.sub_file)
        self.assertEqual(result, "sub_file")
        
        # Test with filename having multiple dots
        complex_file = os.path.join(self.temp_dir, "data.backup.txt")
        result = get_filename_without_extension(complex_file)
        self.assertEqual(result, "data.backup")
        
        # Test with filename having no extension
        no_ext_file = os.path.join(self.temp_dir, "noextension")
        result = get_filename_without_extension(no_ext_file)
        self.assertEqual(result, "noextension")
    
    def test_get_directory_from_filepath(self):
        """Test extracting directory from filepath."""
        # Test with file in root directory
        result = get_directory_from_filepath(self.test_file)
        self.assertEqual(result, self.temp_dir)
        
        # Test with file in subdirectory
        result = get_directory_from_filepath(self.sub_file)
        self.assertEqual(result, self.sub_dir)
        
        # Test with just a filename (should return empty string)
        result = get_directory_from_filepath("just_filename.txt")
        self.assertEqual(result, "")
    
    def test_find_files_by_pattern(self):
        """Test finding files by pattern."""
        # Create additional test files for pattern matching
        with open(os.path.join(self.temp_dir, "report_2023.txt"), "w") as f:
            f.write("Report 2023")
            
        with open(os.path.join(self.temp_dir, "report_2024.txt"), "w") as f:
            f.write("Report 2024")
            
        # Test with simple wildcard pattern
        results = find_files_by_pattern(self.temp_dir, "*.txt")
        self.assertEqual(len(results), 3)  # test_file.txt, report_2023.txt, report_2024.txt
        
        # Test with specific pattern
        results = find_files_by_pattern(self.temp_dir, "report_*.txt")
        self.assertEqual(len(results), 2)
        self.assertTrue(all("report_" in os.path.basename(f) for f in results))
        
        # Test with subdirectories (non-recursive)
        with open(os.path.join(self.sub_dir, "report_sub.txt"), "w") as f:
            f.write("Sub report")
            
        results = find_files_by_pattern(self.temp_dir, "*.txt")
        self.assertEqual(len(results), 3)  # Should not find files in subdirectory
        
        # Test with recursive search
        results = find_files_by_pattern(self.temp_dir, "**/*.txt", recursive=True)
        self.assertEqual(len(results), 4)  # Should find all .txt files
        self.assertTrue(any("subdir" in f for f in results))


if __name__ == '__main__':
    unittest.main()
