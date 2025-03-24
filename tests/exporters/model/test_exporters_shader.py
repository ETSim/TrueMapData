""".

Unit tests for the TMD shader export module.

These tests verify the functionality of the shader export functions.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tmd.exporters.shader import (
    export_heightmap_to_fragment_shader,
    export_heightmap_to_shader_pack,
    _generate_glsl_shader,
    _generate_hlsl_shader,
    _generate_wgsl_shader
)


class TestShaderExport(unittest.TestCase):
    """Test case for shader export functionality.."""
    
    def setUp(self):
        """Set up test environment.."""
        # Create a temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp(prefix="tmd_shader_test_")
        
        # Create a small height map for faster testing
        self.height_map = np.zeros((50, 50), dtype=np.float32)
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        self.X, self.Y = np.meshgrid(x, y)
        self.height_map = np.sin(self.X) * np.cos(self.Y)
    
    def tearDown(self):
        """Clean up after tests.."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def test_export_glsl_shader(self):
        """Test exporting height map to GLSL shader.."""
        output_file = os.path.join(self.test_dir, "test_shader.glsl")
        
        # Test with texture
        result = export_heightmap_to_fragment_shader(
            self.height_map,
            output_file=output_file,
            shader_language="glsl",
            include_texture=True,
            normalize=True
        )
        
        # Check result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output_file))
        self.assertTrue(os.path.exists(os.path.splitext(output_file)[0] + "_texture.png"))
        
        # Check file content
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertIn("// Height Map Fragment Shader", content)
            self.assertIn("uniform sampler2D heightMap", content)
    
    def test_export_hlsl_shader(self):
        """Test exporting height map to HLSL shader.."""
        output_file = os.path.join(self.test_dir, "test_shader.hlsl")
        
        # Test with texture
        result = export_heightmap_to_fragment_shader(
            self.height_map,
            output_file=output_file,
            shader_language="hlsl",
            include_texture=True,
            normalize=True
        )
        
        # Check result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output_file))
        
        # Check file content
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertIn("// Height Map Pixel Shader (HLSL)", content)
            self.assertIn("Texture2D heightMap", content)
    
    def test_export_wgsl_shader(self):
        """Test exporting height map to WGSL shader.."""
        output_file = os.path.join(self.test_dir, "test_shader.wgsl")
        
        # Test with texture
        result = export_heightmap_to_fragment_shader(
            self.height_map,
            output_file=output_file,
            shader_language="wgsl",
            include_texture=True,
            normalize=True
        )
        
        # Check result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output_file))
        
        # Check file content
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertIn("// Height Map Shader (WGSL for WebGPU)", content)
            self.assertIn("var heightMap: texture_2d<f32>", content)
    
    def test_export_without_texture(self):
        """Test exporting shader without texture.."""
        output_file = os.path.join(self.test_dir, "test_no_texture.glsl")
        
        # Test without texture
        result = export_heightmap_to_fragment_shader(
            self.height_map,
            output_file=output_file,
            shader_language="glsl",
            include_texture=False,
            normalize=True
        )
        
        # Check result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output_file))
        self.assertFalse(os.path.exists(os.path.splitext(output_file)[0] + "_texture.png"))
        
        # Check file content
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertIn("// Height Map Fragment Shader", content)
            self.assertIn("Height map encoded directly in shader", content)
    
    def test_shader_pack_generation(self):
        """Test generating a shader pack with multiple formats.."""
        output_dir = os.path.join(self.test_dir, "shader_pack")
        
        # Test creating a shader pack
        result = export_heightmap_to_shader_pack(
            self.height_map,
            output_dir=output_dir,
            base_name="test_heightmap",
            include_texture=True,
            normalize=True
        )
        
        # Check results
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output_dir))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "test_heightmap.glsl")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "test_heightmap.hlsl")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "test_heightmap.wgsl")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "test_heightmap_texture.png")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "README.md")))
    
    def test_invalid_shader_language(self):
        """Test handling of invalid shader language.."""
        output_file = os.path.join(self.test_dir, "test_invalid.shader")
        
        # Test with invalid shader language
        with self.assertRaises(ValueError):
            export_heightmap_to_fragment_shader(
                self.height_map,
                output_file=output_file,
                shader_language="invalid",
                include_texture=True
            )
    
    def test_custom_template(self):
        """Test using a custom shader template.."""
        output_file = os.path.join(self.test_dir, "test_custom.glsl")
        
        # Create a custom template with placeholders
        template = """// Custom Shader Template.

// Scale: {{SCALE}}
// Texture: {{TEXTURE_FILENAME}}

void main() {
    // Custom code
}
"""
        
        # Test with custom template
        result = export_heightmap_to_fragment_shader(
            self.height_map,
            output_file=output_file,
            shader_language="glsl",
            include_texture=True,
            template=template,
            scale=2.5
        )
        
        # Check result
        self.assertIsNotNone(result)
        self.assertTrue(os.path.exists(output_file))
        
        # Check file content
        with open(output_file, 'r') as f:
            content = f.read()
            self.assertIn("// Custom Shader Template", content)
            self.assertIn("// Scale: 2.5", content)
            self.assertIn("// Custom code", content)
    
    def test_shader_generation_functions(self):
        """Test internal shader generation functions.."""
        # Test GLSL generation
        glsl_code = _generate_glsl_shader(self.height_map, "texture.png", 2.0)
        self.assertIn("#version 330 core", glsl_code)
        self.assertIn("uniform float heightScale = 2.000000", glsl_code)
        
        # Test HLSL generation
        hlsl_code = _generate_hlsl_shader(self.height_map, "texture.png", 2.0)
        self.assertIn("// Height Map Pixel Shader (HLSL)", hlsl_code)
        self.assertIn("float height = heightMap.Sample", hlsl_code)
        
        # Test WGSL generation
        wgsl_code = _generate_wgsl_shader(self.height_map, "texture.png", 2.0)
        self.assertIn("// Height Map Shader (WGSL for WebGPU)", wgsl_code)
        self.assertIn("@fragment", wgsl_code)


if __name__ == "__main__":
    unittest.main()
