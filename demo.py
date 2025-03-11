"""
TMD Library Demo - Comprehensive demonstration of TMD library features.

This script demonstrates:
1. Creating and reading TMD files
2. Height map manipulations (crop, rotate, threshold, etc.)
3. Gaussian filtering
4. Cross-section extraction
5. Various export formats (STL, NumPy, images)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import struct

from tmd.processor import TMDProcessor
from tmd.utils.utils import generate_synthetic_tmd, create_sample_height_map
from tmd.processing import (
    crop_height_map, rotate_height_map, threshold_height_map, 
    extract_cross_section, extract_profile_at_percentage
)
from tmd.utils.filter import apply_gaussian_filter
from tmd.exporters.numpy import export_to_npy, export_to_npz, export_metadata_txt
from tmd.exporters.image import (
    convert_heightmap_to_displacement_map, convert_heightmap_to_normal_map,
    convert_heightmap_to_bump_map, convert_heightmap_to_ao_map
)
from tmd.exporters.stl import convert_heightmap_to_stl
from tmd.plotters.matplotlib import (
    plot_height_map_matplotlib, plot_cross_section_matplotlib
)

# Try to import plotly visualization if available
try:
    from tmd.plotters.plotly import plot_height_map_3d, plot_cross_section_plotly
    has_plotly = True
except ImportError:
    has_plotly = False


def main():
    """Main demonstration function showcasing TMD library capabilities."""
    # Create output directory for demo results
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print("TMD (TrueMap Data) Library Demonstration")
    print("=" * 80)
    
    # 1. Generate a synthetic TMD file
    print("\nStep 1: Generating synthetic TMD file...")
    tmd_path = generate_synthetic_tmd(
        output_path=os.path.join(output_dir, "sample.tmd"),
        width=200,
        height=200,
        pattern="combined",
        comment="TMD Library Demo File",
        version=2
    )
    print(f"Created TMD file at: {tmd_path}")
    
    # 2. Process the TMD file
    print("\nStep 2: Processing TMD file...")
    processor = TMDProcessor(tmd_path)
    processor.set_debug(True)
    data = processor.process()
    
    if not data:
        print("Failed to process TMD file. Using synthetic height map directly.")
        height_map = create_sample_height_map(width=200, height=200, pattern="combined")
        data = {
            'height_map': height_map,
            'width': height_map.shape[1],
            'height': height_map.shape[0],
            'x_length': 10.0,
            'y_length': 10.0,
            'x_offset': 0.0,
            'y_offset': 0.0,
            'comment': 'Synthetic height map',
            'header': 'Synthetic TMD'
        }
    
    # Extract height map from processed data
    height_map = data['height_map']
    
    # 3. Basic height map visualization
    print("\nStep 3: Visualizing raw height map...")
    plt.figure(figsize=(10, 8))
    plt.imshow(height_map, cmap='viridis')
    plt.colorbar(label='Height')
    plt.title('Raw Height Map')
    plt.savefig(os.path.join(output_dir, 'raw_height_map.png'), dpi=300)
    plt.close()
    
    # 4. Height map manipulation
    print("\nStep 4: Demonstrating height map manipulation operations...")
    
    # 4.1 Cropping
    print("  - Cropping height map...")
    rows, cols = height_map.shape
    crop_region = (rows//4, 3*rows//4, cols//4, 3*cols//4)  # Middle 50%
    cropped_map = crop_height_map(height_map, crop_region)
    
    # Visualize cropped map
    plt.figure(figsize=(10, 8))
    plt.imshow(cropped_map, cmap='viridis')
    plt.colorbar(label='Height')
    plt.title('Cropped Height Map')
    plt.savefig(os.path.join(output_dir, 'cropped_height_map.png'), dpi=300)
    plt.close()
    
    # 4.2 Rotation
    print("  - Rotating height map...")
    rotated_map = rotate_height_map(height_map, angle=45)
    
    # Visualize rotated map
    plt.figure(figsize=(10, 8))
    plt.imshow(rotated_map, cmap='viridis')
    plt.colorbar(label='Height')
    plt.title('Rotated Height Map (45°)')
    plt.savefig(os.path.join(output_dir, 'rotated_height_map.png'), dpi=300)
    plt.close()
    
    # 4.3 Z-Thresholding
    print("  - Applying threshold to height map...")
    # Calculate threshold values based on height range
    h_min, h_max = height_map.min(), height_map.max()
    h_range = h_max - h_min
    
    # Set thresholds to keep only middle 60% of height values
    min_threshold = h_min + 0.2 * h_range
    max_threshold = h_max - 0.2 * h_range
    
    # Apply threshold
    thresholded_map = threshold_height_map(
        height_map, 
        min_height=min_threshold, 
        max_height=max_threshold
    )
    
    # Visualize thresholded map
    plt.figure(figsize=(10, 8))
    plt.imshow(thresholded_map, cmap='viridis')
    plt.colorbar(label='Height')
    plt.title(f'Thresholded Height Map ({min_threshold:.2f} - {max_threshold:.2f})')
    plt.savefig(os.path.join(output_dir, 'thresholded_height_map.png'), dpi=300)
    plt.close()
    
    # 5. Gaussian filtering
    print("\nStep 5: Applying Gaussian filter...")
    smoothed_map = apply_gaussian_filter(height_map, sigma=2.0)
    
    # Visualize smoothed map
    plt.figure(figsize=(10, 8))
    plt.imshow(smoothed_map, cmap='viridis')
    plt.colorbar(label='Height')
    plt.title('Gaussian Smoothed Height Map (sigma=2.0)')
    plt.savefig(os.path.join(output_dir, 'smoothed_height_map.png'), dpi=300)
    plt.close()
    
    # 6. Cross-section extraction
    print("\nStep 6: Extracting cross-section...")
    
    # Extract cross-section at the middle row
    x_position = height_map.shape[0] // 2
    x_positions, x_heights = extract_cross_section(
        height_map, 
        data, 
        axis='x', 
        position=x_position
    )
    
    # Plot cross-section
    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, x_heights, 'b-')
    plt.title(f'X Cross-Section at Row {x_position}')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Height')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'x_cross_section.png'), dpi=300)
    plt.close()
    
    # 7. Profile extraction at specific percentage
    print("\nStep 7: Extracting profile at 75% position...")
    profile = extract_profile_at_percentage(
        height_map,
        data,
        axis='x',
        percentage=75.0,
        save_path=os.path.join(output_dir, 'profile_75_percent.npy')
    )
    
    # Plot profile
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(profile)), profile, 'r-')
    plt.title('Profile at 75% along Y-axis')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Height')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'profile_75_percent.png'), dpi=300)
    plt.close()
    
    # 8. Export to various formats
    print("\nStep 8: Demonstrating export options...")
    
    # 8.1 Export to NumPy formats
    print("  - Exporting to NumPy formats...")
    npy_path = export_to_npy(height_map, os.path.join(output_dir, 'height_map.npy'))
    npz_path = export_to_npz(data, os.path.join(output_dir, 'tmd_data.npz'))
    txt_path = export_metadata_txt(data, os.path.join(output_dir, 'metadata.txt'))
    
    # 8.2 Export as image maps
    print("  - Converting to image maps...")
    disp_path = os.path.join(output_dir, 'displacement_map.png')
    convert_heightmap_to_displacement_map(height_map, disp_path)
    
    normal_path = os.path.join(output_dir, 'normal_map.png')
    convert_heightmap_to_normal_map(height_map, normal_path)
    
    bump_path = os.path.join(output_dir, 'bump_map.png')
    convert_heightmap_to_bump_map(height_map, bump_path)
    
    # Make a smaller version for AO map (slow computation)
    small_height_map = height_map[::4, ::4]  # Downsample for faster processing
    ao_path = os.path.join(output_dir, 'ao_map.png')
    convert_heightmap_to_ao_map(small_height_map, ao_path, samples=8)
    
    # 8.3 Export as STL for 3D printing
    print("  - Exporting to STL format...")
    stl_path = os.path.join(output_dir, 'height_map.stl')
    convert_heightmap_to_stl(
        height_map,
        filename=stl_path,
        x_length=data['x_length'],
        y_length=data['y_length'],
        z_scale=2.0,  # Exaggerate height for better visualization
        ascii=True    # Use ASCII format for easier viewing
    )
    
    # Create a binary STL version too
    binary_stl_path = os.path.join(output_dir, 'height_map_binary.stl')
    convert_heightmap_to_stl(
        height_map,
        filename=binary_stl_path,
        x_length=data['x_length'],
        y_length=data['y_length'],
        z_scale=2.0,
        ascii=False   # Binary format for smaller file size
    )
    
    # 9. Advanced visualization if available
    if has_plotly:
        print("\nStep 9: Creating advanced visualizations with Plotly...")
        plot_height_map_3d(
            height_map, 
            title="3D Surface Plot of Height Map",
            filename=os.path.join(output_dir, "height_map_3d.html")
        )
        
        # Cross-section with Plotly
        plot_cross_section_plotly(
            height_map,
            data,
            axis='x',
            position=x_position,
            title="Interactive Cross-Section Plot",
            html_filename=os.path.join(output_dir, "cross_section_interactive.html")
        )
    else:
        print("\nStep 9: Skipping advanced visualizations (Plotly not available)")
        
    # Advanced matplotlib visualization
    print("\nCreating 3D surface plot with matplotlib...")
    plot_height_map_matplotlib(
        height_map,
        colorbar_label="Height (normalized)",
        filename=os.path.join(output_dir, "height_map_3d_matplotlib.png")
    )
    
    # Plot cross-section with matplotlib
    plot_cross_section_matplotlib(
        height_map,
        data,
        axis='x',
        position=x_position,
        title="X Cross-Section with Matplotlib",
        filename=os.path.join(output_dir, "cross_section_matplotlib.png")
    )
    
    # Finished!
    print("\n" + "=" * 80)
    print("Demonstration complete! Output files are in the 'output' directory.")
    print("=" * 80)
    print("\nHere's a summary of the files created:")
    print(f"  - Height maps: raw, cropped, rotated, thresholded, smoothed")
    print(f"  - Cross-sections and profiles")
    print(f"  - Export formats:")
    print(f"    - NumPy: {npy_path}, {npz_path}")
    print(f"    - Metadata: {txt_path}")
    print(f"    - Image maps: displacement, normal, bump, ambient occlusion")
    print(f"    - 3D models: {stl_path}, {binary_stl_path}")
    print(f"    - Visualizations: PNG images and HTML files (if Plotly is available)")


if __name__ == "__main__":
    main()
