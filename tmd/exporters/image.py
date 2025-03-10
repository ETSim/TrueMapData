"""
Functions for exporting TMD height maps to image formats like displacement and normal maps.
"""
import numpy as np
from PIL import Image
import os
from scipy import ndimage

def convert_heightmap_to_displacement_map(height_map, filename="displacement_map.png"):
    """
    Converts the height map into a grayscale displacement map (PNG).
    
    Args:
        height_map: 2D numpy array of height values
        filename: Name of the output PNG file
        
    Returns:
        PIL Image object of the displacement map
    """
    hmin = float(height_map.min())
    hmax = float(height_map.max())
    norm = (height_map - hmin) / (hmax - hmin) * 255.0
    norm = norm.astype(np.uint8)
    im = Image.fromarray(norm)
    im.save(filename)
    print(f"Displacement map saved to {filename}")
    return im

def convert_heightmap_to_normal_map(height_map, filename="normal_map.png", strength=1.0):
    """
    Converts the height map to a normal map (RGB) for use in 3D rendering and games.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Name of the output PNG file
        strength: Factor to control the strength of normals (higher makes details more pronounced)
        
    Returns:
        PIL Image object of the normal map
    """
    # Ensure we have a clean float array to work with
    height_map = height_map.astype(np.float32)
    
    # Get the dimensions
    rows, cols = height_map.shape
    
    # Prepare the normal map array (RGB)
    normal_map = np.zeros((rows, cols, 3), dtype=np.uint8)
    
    # Calculate the sampling step size
    dx = 1.0
    dy = 1.0
    
    # X and Y gradients using Sobel filter approximation
    for y in range(1, rows-1):
        for x in range(1, cols-1):
            # Compute derivatives in x and y directions
            dzdx = (height_map[y, x+1] - height_map[y, x-1]) / (2.0 * dx)
            dzdy = (height_map[y+1, x] - height_map[y-1, x]) / (2.0 * dy)
            
            # Apply strength factor
            dzdx *= strength
            dzdy *= strength
            
            # Compute the normal vector
            normal = np.array([-dzdx, -dzdy, 1.0])
            
            # Normalize the vector
            norm = np.sqrt(np.sum(normal * normal))
            if norm > 0:
                normal = normal / norm
            
            # Convert from [-1, 1] range to [0, 255] for RGB
            normal_map[y, x, 0] = int((normal[0] * 0.5 + 0.5) * 255)
            normal_map[y, x, 1] = int((normal[1] * 0.5 + 0.5) * 255)
            normal_map[y, x, 2] = int((normal[2] * 0.5 + 0.5) * 255)
    
    # Handle borders (duplicate edge pixels)
    normal_map[0, :, :] = normal_map[1, :, :]
    normal_map[-1, :, :] = normal_map[-2, :, :]
    normal_map[:, 0, :] = normal_map[:, 1, :]
    normal_map[:, -1, :] = normal_map[:, -2, :]
    
    # Create and save the image
    im = Image.fromarray(normal_map)
    im.save(filename)
    print(f"Normal map saved to {filename}")
    return im

def convert_heightmap_to_bsrf_map(height_map, filename="bsrf_map.png", scale=1.0, bias=0.5):
    """
    Converts the height map to a BSRF (Bidirectional Scattering Reflection Function) map.
    This is often used in material editing software for surface reflectivity.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Name of the output PNG file
        scale: Scale factor to adjust the intensity of the effect
        bias: Bias value to adjust the mid-point of the effect (0.0-1.0)
        
    Returns:
        PIL Image object of the BSRF map
    """
    # Ensure we have a clean float array
    height_map = height_map.astype(np.float32)
    
    # Normalize height map to 0-1
    h_min = np.min(height_map)
    h_max = np.max(height_map)
    height_norm = (height_map - h_min) / (h_max - h_min) if h_max > h_min else np.zeros_like(height_map)
    
    # Apply scale and bias
    bsrf_map = np.clip(height_norm * scale + bias, 0.0, 1.0)
    
    # Convert to 8-bit
    bsrf_map = (bsrf_map * 255).astype(np.uint8)
    
    # Create and save image
    im = Image.fromarray(bsrf_map)
    im.save(filename)
    print(f"BSRF map saved to {filename}")
    return im

def convert_heightmap_to_bump_map(height_map, filename="bump_map.png", strength=1.0, blur_radius=1.0):
    """
    Converts the height map to a bump map with optional blurring.
    A bump map is similar to a normal map but represents height directly rather than surface normals.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Name of the output PNG file
        strength: Strength factor for bump effect
        blur_radius: Radius for Gaussian blur to smooth the result
        
    Returns:
        PIL Image object of the bump map
    """
    # Ensure we have a clean float array
    height_map = height_map.astype(np.float32)
    
    # Normalize and apply strength
    h_min = np.min(height_map)
    h_max = np.max(height_map)
    bump_map = (height_map - h_min) / (h_max - h_min) if h_max > h_min else np.zeros_like(height_map)
    bump_map *= strength
    
    # Apply Gaussian blur if requested
    if blur_radius > 0:
        bump_map = ndimage.gaussian_filter(bump_map, sigma=blur_radius)
    
    # Renormalize after blur
    b_min = np.min(bump_map)
    b_max = np.max(bump_map)
    bump_map = (bump_map - b_min) / (b_max - b_min) if b_max > b_min else bump_map
    
    # Convert to 8-bit
    bump_map = (bump_map * 255).astype(np.uint8)
    
    # Create and save image
    im = Image.fromarray(bump_map)
    im.save(filename)
    print(f"Bump map saved to {filename}")
    return im

def convert_heightmap_to_ao_map(height_map, filename="ao_map.png", strength=1.0, samples=16, radius=3.0):
    """
    Converts the height map to an ambient occlusion (AO) map.
    AO maps represent how exposed each point is to ambient lighting.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Name of the output PNG file
        strength: Strength factor for AO effect
        samples: Number of sample directions
        radius: Maximum distance to check for occlusions
        
    Returns:
        PIL Image object of the AO map
    """
    # Ensure we have a clean float array
    height_map = height_map.astype(np.float32)
    
    # Normalize the height map
    h_min = np.min(height_map)
    h_max = np.max(height_map)
    height_norm = (height_map - h_min) / (h_max - h_min) if h_max > h_min else np.zeros_like(height_map)
    
    # Calculate ambient occlusion
    rows, cols = height_map.shape
    ao_map = np.ones((rows, cols), dtype=np.float32)
    
    # Simple screen space ambient occlusion approximation
    for i in range(rows):
        for j in range(cols):
            occlusion = 0.0
            sample_count = 0
            
            # Sample points in a radius around current point
            for s in range(samples):
                # Calculate sample angle and distance
                angle = 2.0 * np.pi * s / samples
                for r in range(1, int(radius) + 1):
                    # Calculate sample coordinates
                    si = int(i + r * np.sin(angle))
                    sj = int(j + r * np.cos(angle))
                    
                    # Check if sample is within bounds
                    if 0 <= si < rows and 0 <= sj < cols:
                        # Compare heights - if sample is higher, it occludes
                        h_diff = height_norm[si, sj] - height_norm[i, j]
                        if h_diff > 0:
                            # Weight by distance and height difference
                            occlusion += h_diff * (1.0 - r/radius)
                            sample_count += 1
            
            # Calculate average occlusion
            if sample_count > 0:
                ao_map[i, j] = 1.0 - (occlusion / sample_count) * strength
    
    # Ensure range 0-1
    ao_map = np.clip(ao_map, 0, 1)
    
    # Convert to 8-bit
    ao_map = (ao_map * 255).astype(np.uint8)
    
    # Create and save image
    im = Image.fromarray(ao_map)
    im.save(filename)
    print(f"Ambient occlusion map saved to {filename}")
    return im

def convert_heightmap_to_multi_channel_map(height_map, filename="material_map.png", channel_type="rgbe"):
    """
    Converts the height map to a multi-channel image encoding different surface properties.
    Common formats use RGB channels for normals and alpha for height.
    
    Args:
        height_map: 2D numpy array of height values
        filename: Name of the output PNG file
        channel_type: Type of encoding: "rgbe" (RGB+Height), "rg" (Red/Green normal), etc.
        
    Returns:
        PIL Image object of the multi-channel map
    """
    # Ensure we have a clean float array
    height_map = height_map.astype(np.float32)
    
    # Get dimensions
    rows, cols = height_map.shape
    
    if channel_type.lower() == "rgbe":
        # Create RGBA image (normal map + height in alpha)
        multi_map = np.zeros((rows, cols, 4), dtype=np.uint8)
        
        # Generate normal map data for RGB channels
        for y in range(1, rows-1):
            for x in range(1, cols-1):
                # Compute derivatives
                dzdx = (height_map[y, x+1] - height_map[y, x-1]) / 2.0
                dzdy = (height_map[y+1, x] - height_map[y-1, x]) / 2.0
                
                # Compute normal
                normal = np.array([-dzdx, -dzdy, 1.0])
                norm_val = np.sqrt(np.sum(normal * normal))
                if norm_val > 0:
                    normal = normal / norm_val
                
                # Map from [-1,1] to [0,255] for RGB with safe clipping
                r_val = np.clip(int((normal[0] * 0.5 + 0.5) * 255), 0, 255)
                g_val = np.clip(int((normal[1] * 0.5 + 0.5) * 255), 0, 255)
                b_val = np.clip(int((normal[2] * 0.5 + 0.5) * 255), 0, 255)
                
                multi_map[y, x, 0] = r_val  # R
                multi_map[y, x, 1] = g_val  # G
                multi_map[y, x, 2] = b_val  # B
        
        # Handle borders
        multi_map[0, :, :3] = multi_map[1, :, :3]
        multi_map[-1, :, :3] = multi_map[-2, :, :3]
        multi_map[:, 0, :3] = multi_map[:, 1, :3]
        multi_map[:, -1, :3] = multi_map[:, -2, :3]
        
        # Add height to alpha channel
        h_min = np.min(height_map)
        h_max = np.max(height_map)
        height_norm = (height_map - h_min) / (h_max - h_min) if h_max > h_min else np.zeros_like(height_map)
        multi_map[:, :, 3] = (height_norm * 255).astype(np.uint8)  # A
        
        # Create and save RGBA image
        im = Image.fromarray(multi_map, mode='RGBA')
        
    elif channel_type.lower() == "rg":
        # Create RG image (normal map using only red and green channels - compact)
        multi_map = np.zeros((rows, cols, 3), dtype=np.uint8)
        
        # Generate normal map data for RG channels
        for y in range(1, rows-1):
            for x in range(1, cols-1):
                # Compute derivatives
                dzdx = (height_map[y, x+1] - height_map[y, x-1]) / 2.0
                dzdy = (height_map[y+1, x] - height_map[y-1, x]) / 2.0
                
                # Map from [-1,1] to [0,255] with safe clipping
                r_val = np.clip(int((dzdx * 0.5 + 0.5) * 255), 0, 255)
                g_val = np.clip(int((dzdy * 0.5 + 0.5) * 255), 0, 255)
                
                multi_map[y, x, 0] = r_val  # R
                multi_map[y, x, 1] = g_val  # G
                multi_map[y, x, 2] = 128    # B (fixed middle value)
        
        # Handle borders
        multi_map[0, :, :] = multi_map[1, :, :]
        multi_map[-1, :, :] = multi_map[-2, :, :]
        multi_map[:, 0, :] = multi_map[:, 1, :]
        multi_map[:, -1, :] = multi_map[:, -2, :]
        
        # Create and save image
        im = Image.fromarray(multi_map)
    
    else:
        raise ValueError(f"Unsupported channel type: {channel_type}")
    
    im.save(filename)
    print(f"Multi-channel map ({channel_type}) saved to {filename}")
    return im
