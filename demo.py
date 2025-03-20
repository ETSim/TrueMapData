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

from tmd.processor import TMDProcessor
from tmd.utils.metadata import export_metadata_txt
from tmd.utils.utils import generate_synthetic_tmd, create_sample_height_map
from tmd.utils.processing import (
    crop_height_map, rotate_height_map, threshold_height_map, 
    extract_cross_section, extract_profile_at_percentage
)
from tmd.utils.filter import apply_gaussian_filter
from tmd.exporters.compression import export_to_npy, export_to_npz
from tmd.exporters.image import (
    convert_heightmap_to_displacement_map, 
    convert_heightmap_to_normal_map,
    convert_heightmap_to_bump_map, 
    convert_heightmap_to_ao_map,
    convert_heightmap_to_multi_channel_map,
    generate_roughness_map,
    generate_all_maps
)
from tmd.exporters.model import convert_heightmap_to_stl, convert_heightmap_to_obj, convert_heightmap_to_plyrom tmd.exporters.model import convert_heightmap_to_stl, convert_heightmap_to_obj, convert_heightmap_to_ply
from typing import Tuple, Optional
from tmd.plotters.matplotlib import (t (
    plot_height_map_matplotlib
)

try:try:
    from tmd.plotters.plotly import plot_height_map_3d, plot_cross_section_plotlyfrom tmd.plotters.plotly import plot_height_map_3d, plot_cross_section_plotly
    has_plotly = True
except ImportError:
    has_plotly = Falselse


# ------------------ STEP FUNCTIONS ------------------# ------------------ STEP FUNCTIONS ------------------

def generate_and_process_tmd(output_dir):def generate_and_process_tmd(output_dir):
    """
    Generates a synthetic TMD file and processes it.erates a synthetic TMD file and processes it.
    
    Returns:Returns:
        data (dict): Processed metadata and height map. (dict): Processed metadata and height map.
        tmd_path (str): Path to the generated TMD file.
    """
    print("\nStep 1: Generating synthetic TMD file...")nt("\nStep 1: Generating synthetic TMD file...")
    tmd_path = os.path.join(output_dir, "sample.tmd")
    # Assuming generate_synthetic_tmd now accepts parameters for width, height, pattern, etc.ameters for width, height, pattern, etc.
    tmd_path = generate_synthetic_tmd(
        output_path=tmd_path,
        width=201,
        height=200,,
        pattern="combined",mbined",
        comment="Created by TrueMap v6", TrueMap v6",
        version=2
    )
    print(f"Created TMD file at: {tmd_path}")rint(f"Created TMD file at: {tmd_path}")

    print("\nStep 2: Processing TMD file...")    print("\nStep 2: Processing TMD file...")
    processor = TMDProcessor(tmd_path)
    processor.set_debug(True)
    data = processor.process())

    if not data:    if not data:
        print("Failed to process TMD file. Using synthetic height map directly.")ailed to process TMD file. Using synthetic height map directly.")
        height_map = create_sample_height_map(width=201, height=200, pattern="combined")bined")
        data = {
            'height_map': height_map,ght_map': height_map,
            'width': height_map.shape[1],[1],
            'height': height_map.shape[0],,
            'x_length': 10.0,
            'y_length': 10.0,
            'x_offset': 0.0,
            'y_offset': 0.0,
            'comment': 'Synthetic height map',hetic height map',
            'header': 'Synthetic TMD'
        }
    return data, tmd_pathn data, tmd_path



def visualize_raw_height_map(height_map, output_dir):def visualize_raw_height_map(height_map, output_dir):
    """
    Visualizes and saves the raw height map as a PNG image.ualizes and saves the raw height map as a PNG image.
    """
    print("\nStep 3: Visualizing raw height map...")nt("\nStep 3: Visualizing raw height map...")
    plt.figure(figsize=(10, 8))
    plt.imshow(height_map, cmap='viridis')='viridis')
    plt.colorbar(label='Height')
    plt.title('Raw Height Map')
    out_path = os.path.join(output_dir, 'raw_height_map.png')put_dir, 'raw_height_map.png')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved raw height map to {out_path}")ed raw height map to {out_path}")



def manipulate_height_map_demo(height_map, output_dir, data):def manipulate_height_map_demo(height_map, output_dir, data):
    """
    Demonstrates cropping, rotation, and thresholding on the height map.onstrates cropping, rotation, and thresholding on the height map.
    """
    print("\nStep 4: Demonstrating height map manipulation operations...")nt("\nStep 4: Demonstrating height map manipulation operations...")

    # 4.1 Cropping: extract the middle 50% region.    # 4.1 Cropping: extract the middle 50% region.
    rows, cols = height_map.shape
    crop_region = (rows // 4, 3 * rows // 4, cols // 4, 3 * cols // 4) rows // 4, cols // 4, 3 * cols // 4)
    cropped_map = crop_height_map(height_map, crop_region)
    out_path = os.path.join(output_dir, 'cropped_height_map.png')p.png')
    plt.figure(figsize=(10, 8))
    plt.imshow(cropped_map, cmap='viridis')p='viridis')
    plt.colorbar(label='Height')
    plt.title('Cropped Height Map')p')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Cropped height map saved to {out_path}")pped height map saved to {out_path}")

    # 4.2 Rotation: rotate by 45 degrees.    # 4.2 Rotation: rotate by 45 degrees.
    rotated_map = rotate_height_map(height_map, angle=45)t_map, angle=45)
    out_path = os.path.join(output_dir, 'rotated_height_map.png')ap.png')
    plt.figure(figsize=(10, 8))
    plt.imshow(rotated_map, cmap='viridis')p='viridis')
    plt.colorbar(label='Height')
    plt.title('Rotated Height Map (45°)')p (45°)')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Rotated height map saved to {out_path}")ated height map saved to {out_path}")

    # 4.3 Thresholding: keep the middle 60% of height values.    # 4.3 Thresholding: keep the middle 60% of height values.
    h_min, h_max = height_map.min(), height_map.max()
    h_range = h_max - h_min
    min_threshold = h_min + 0.2 * h_range 0.2 * h_range
    max_threshold = h_max - 0.2 * h_range
    thresholded_map = threshold_height_map(height_map, min_height=min_threshold, max_height=max_threshold)p(height_map, min_height=min_threshold, max_height=max_threshold)
    out_path = os.path.join(output_dir, 'thresholded_height_map.png')
    plt.figure(figsize=(10, 8))
    plt.imshow(thresholded_map, cmap='viridis') cmap='viridis')
    plt.colorbar(label='Height')
    plt.title(f'Thresholded Height Map ({min_threshold:.2f} - {max_threshold:.2f})')ht Map ({min_threshold:.2f} - {max_threshold:.2f})')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Thresholded height map saved to {out_path}")esholded height map saved to {out_path}")



def gaussian_filter_demo(height_map, output_dir):def gaussian_filter_demo(height_map, output_dir):
    """
    Applies a Gaussian filter to the height map and saves the result.lies a Gaussian filter to the height map and saves the result.
    """
    print("\nStep 5: Applying Gaussian filter...")nt("\nStep 5: Applying Gaussian filter...")
    smoothed_map = apply_gaussian_filter(height_map, sigma=2.0)p, sigma=2.0)
    out_path = os.path.join(output_dir, 'smoothed_height_map.png')g')
    plt.figure(figsize=(10, 8))
    plt.imshow(smoothed_map, cmap='viridis')ap='viridis')
    plt.colorbar(label='Height')
    plt.title('Gaussian Smoothed Height Map (sigma=2.0)') Height Map (sigma=2.0)')
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Smoothed height map saved to {out_path}")othed height map saved to {out_path}")



def cross_section_demo(height_map, data, output_dir):def cross_section_demo(height_map, data, output_dir):
    """
    Extracts and plots a cross-section from the height map.racts and plots a cross-section from the height map.
    """
    print("\nStep 6: Extracting cross-section...")nt("\nStep 6: Extracting cross-section...")
    x_position = height_map.shape[0] // 2
    x_positions, x_heights = extract_cross_section(height_map, data, axis='x', position=x_position)s_section(height_map, data, axis='x', position=x_position)
    out_path = os.path.join(output_dir, 'x_cross_section.png')
    plt.figure(figsize=(10, 6))
    plt.plot(x_positions, x_heights, 'b-')ghts, 'b-')
    plt.title(f'X Cross-Section at Row {x_position}')position}')
    plt.xlabel('X Position (mm)')
    plt.ylabel('Height')
    plt.grid(True, alpha=0.3)=0.3)
    plt.savefig(out_path, dpi=300)=300)
    plt.close()
    print(f"Cross-section saved to {out_path}")ss-section saved to {out_path}")
    return x_position



def profile_demo(height_map, data, output_dir):def profile_demo(height_map, data, output_dir):
    """
    Extracts a profile at a specified percentage along the X axis, saves and plots it.racts a profile at a specified percentage along the X axis, saves and plots it.
    """
    print("\nStep 7: Extracting profile at 75% position...")nt("\nStep 7: Extracting profile at 75% position...")
    profile = extract_profile_at_percentage(height_map, data, axis='x', percentage=75.0,, axis='x', percentage=75.0,
                                            save_path=os.path.join(output_dir, 'profile_75_percent.npy'))75_percent.npy'))
    out_path = os.path.join(output_dir, 'profile_75_percent.png')
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(profile)), profile, 'r-')ile)), profile, 'r-')
    plt.title('Profile at 75% along Y-axis')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Height')
    plt.grid(True, alpha=0.3)=0.3)
    plt.savefig(out_path, dpi=300)=300)
    plt.close()
    print(f"Profile extracted and saved to {out_path}")file extracted and saved to {out_path}")



def export_formats_demo(height_map, data, output_dir):def export_formats_demo(height_map, data, output_dir):
    """
    Exports height map data to various formats: NumPy arrays, metadata text, image maps, STL.orts height map data to various formats: NumPy arrays, metadata text, image maps, STL.
    """
    print("\nStep 8: Demonstrating export options...")nt("\nStep 8: Demonstrating export options...")
    # 8.1 Export to NumPy formats
    npy_path = export_to_npy(height_map, os.path.join(output_dir, 'height_map.npy'))ht_map, os.path.join(output_dir, 'height_map.npy'))
    npz_path = export_to_npz(data, os.path.join(output_dir, 'tmd_data.npz'))
    txt_path = export_metadata_txt(data, os.path.join(output_dir, 'metadata.txt'))txt'))
    print(f"Exported NumPy formats: {npy_path}, {npz_path}")
    print(f"Exported metadata to {txt_path}")

    # 8.2 Export as image maps    # 8.2 Export as image maps
    convert_heightmap_to_displacement_map(height_map, os.path.join(output_dir, 'displacement_map.png'))acement_map(height_map, os.path.join(output_dir, 'displacement_map.png'))
    convert_heightmap_to_normal_map(height_map, os.path.join(output_dir, 'normal_map.png'), strength=1.5)5)
    convert_heightmap_to_bump_map(height_map, os.path.join(output_dir, 'bump_map.png'), strength=1.2, blur_radius=0.8)r_radius=0.8)
    
    # AO map: use a downsampled version for performance# AO map: use a downsampled version for performance
    small_height_map = height_map[::4, ::4]
    convert_heightmap_to_ao_map(small_height_map, os.path.join(output_dir, 'ao_map.png'), samples=8, intensity=1.2)t_map, os.path.join(output_dir, 'ao_map.png'), samples=8, intensity=1.2)
    
    # Create a roughness map# Create a roughness map
    roughness_map = generate_roughness_map(height_map, kernel_size=3, scale=1.5)_roughness_map(height_map, kernel_size=3, scale=1.5)
    plt.imsave(os.path.join(output_dir, 'roughness_map.png'), roughness_map, cmap='gray')p='gray')
    
    # Create multi-channel material map# Create multi-channel material map
    convert_heightmap_to_multi_channel_map(height_map, os.path.join(output_dir, 'material_rgbe.png'), channel_type="rgbe")map(height_map, os.path.join(output_dir, 'material_rgbe.png'), channel_type="rgbe")
    
    # Generate complete material set# Generate complete material set
    material_dir = os.path.join(output_dir, "materials")ut_dir, "materials")
    os.makedirs(material_dir, exist_ok=True)
    metadata = {
        "normal_strength": 1.5,strength": 1.5,
        "bump_strength": 1.2,
        "roughness_scale": 1.0,0,
        "units": "µm",
        "x_length": data.get('x_length', 10.0),ta.get('x_length', 10.0),
        "y_length": data.get('y_length', 10.0),
    }
    generate_all_maps(height_map, output_dir=material_dir)enerate_all_maps(height_map, output_dir=material_dir)
    
    print("Exported image maps.")print("Exported image maps.")

    # 8.3 Export as 3D models    # 8.3 Export as 3D models
    stl_path = os.path.join(output_dir, 'height_map_ascii.stl')utput_dir, 'height_map_ascii.stl')
    convert_heightmap_to_stl(
        height_map,
        filename=stl_path,l_path,
        x_length=data.get('x_length', 10.0),'x_length', 10.0),
        y_length=data.get('y_length', 10.0),
        z_scale=2.0,
        ascii=True
    )
    binary_stl_path = os.path.join(output_dir, 'height_map_binary.stl')inary_stl_path = os.path.join(output_dir, 'height_map_binary.stl')
    convert_heightmap_to_stl(
        height_map,
        filename=binary_stl_path,nary_stl_path,
        x_length=data.get('x_length', 10.0),th', 10.0),
        y_length=data.get('y_length', 10.0),
        z_scale=2.0,
        ascii=False
    )
    obj_path = os.path.join(output_dir, 'height_map.obj')bj_path = os.path.join(output_dir, 'height_map.obj')
    convert_heightmap_to_obj(
        height_map,
        filename=obj_path,j_path,
        x_length=data.get('x_length', 10.0),'x_length', 10.0),
        y_length=data.get('y_length', 10.0),
        z_scale=2.0
    )
    ply_path = os.path.join(output_dir, 'height_map.ply')ly_path = os.path.join(output_dir, 'height_map.ply')
    convert_heightmap_to_ply(
        height_map,
        filename=ply_path,y_path,
        x_length=data.get('x_length', 10.0),'x_length', 10.0),
        y_length=data.get('y_length', 10.0),
        z_scale=2.0
    )
    print(f"Exported 3D models: STL, OBJ, PLY")rint(f"Exported 3D models: STL, OBJ, PLY")



def advanced_visualizations_demo(height_map, data, output_dir, x_position):def advanced_visualizations_demo(height_map, data, output_dir, x_position):
    """
    Creates advanced visualizations using Plotly and matplotlib.ates advanced visualizations using Plotly and matplotlib.
    """
    print("\nStep 9: Creating advanced visualizations with Plotly...")nt("\nStep 9: Creating advanced visualizations with Plotly...")
    if has_plotly:
        plot_height_map_3d(t_map_3d(
                height_map,  
                title="3D Surface Plot of Height Map",rface Plot of Height Map",
                filename=os.path.join(output_dir, "height_map_3d.html")ght_map_3d.html")
        )
    else:
        print("Plotly not available. Skipping interactive 3D visualization.")rint("Plotly not available. Skipping interactive 3D visualization.")
    
    print("\nCreating 3D surface plot with matplotlib...")print("\nCreating 3D surface plot with matplotlib...")
    plot_height_map_matplotlib(
        height_map,
        colorbar_label="Height (normalized)",bel="Height (normalized)",
        filename=os.path.join(output_dir, "height_map_3d_matplotlib.png")ight_map_3d_matplotlib.png")
    )
    print("Advanced visualizations complete.")rint("Advanced visualizations complete.")


def image_processing_showcase(height_map, output_dir):def image_processing_showcase(height_map, output_dir):
    """
    Showcases more advanced image processing capabilities likewcases more advanced image processing capabilities like
    material maps and combining different map types.
    """
    print("\nStep 10: Advanced image processing showcase...")nt("\nStep 10: Advanced image processing showcase...")
    # Create a directory for these examples
    showcase_dir = os.path.join(output_dir, "showcase") "showcase")
    os.makedirs(showcase_dir, exist_ok=True)
    
    # Generate a normal map with different strength values# Generate a normal map with different strength values
    print("Generating normal maps with different strength values...")values...")
    for strength in [0.5, 1.0, 2.0, 3.0]:
        normal_map = convert_heightmap_to_normal_map(_normal_map(
            height_map,
            filename=os.path.join(showcase_dir, f"normal_strength_{strength}.png"),.path.join(showcase_dir, f"normal_strength_{strength}.png"),
            strength=strength
        )
    
    # Generate bump maps with different blur settings# Generate bump maps with different blur settings
    print("Generating bump maps with different blur settings...")ettings...")
    for blur in [0.0, 0.5, 1.0, 2.0]:
        bump_map = convert_heightmap_to_bump_map(to_bump_map(
            height_map,
            filename=os.path.join(showcase_dir, f"bump_blur_{blur}.png"),.path.join(showcase_dir, f"bump_blur_{blur}.png"),
            strength=1.5,
            blur_radius=blurlur
        )
    
    # Generate roughness maps with different scales# Generate roughness maps with different scales
    print("Generating roughness maps with different scales...") scales...")
    for scale in [0.5, 1.0, 2.0]:
        roughness = generate_roughness_map(height_map, scale=scale)hness_map(height_map, scale=scale)
        plt.imsave(os.path.join(showcase_dir, f"roughness_scale_{scale}.png"), roughness, cmap='gray')ale}.png"), roughness, cmap='gray')
    
    # Generate AO maps with different parameters# Generate AO maps with different parameters
    print("Generating ambient occlusion maps with different parameters...")h different parameters...")
    small_map = height_map[::4, ::4]  # Use smaller map for speed
    for intensity in [0.8, 1.2, 1.5]:
        ao_map = convert_heightmap_to_ao_map(_ao_map(
            small_map,
            filename=os.path.join(showcase_dir, f"ao_intensity_{intensity}.png"),s.path.join(showcase_dir, f"ao_intensity_{intensity}.png"),
            intensity=intensity,
            radius=1.5
        )
    
    # Generate multi-channel maps for different material types# Generate multi-channel maps for different material types
    print("Generating multi-channel material maps...")
    for channel_type in ["rgbe", "rg"]:
        convert_heightmap_to_multi_channel_map(nel_map(
            height_map,
            filename=os.path.join(showcase_dir, f"material_{channel_type}.png"),.path.join(showcase_dir, f"material_{channel_type}.png"),
            channel_type=channel_type
        )
    
    # Create a visual summary grid# Create a visual summary grid
    print("Creating visual summary grid of all maps...") grid of all maps...")
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs = axs.ravel()
    
    # Load and display different map types# Load and display different map types
    map_files = {
        0: (os.path.join(showcase_dir, "normal_strength_2.0.png"), "Normal Map (Strength 2.0)"),th.join(showcase_dir, "normal_strength_2.0.png"), "Normal Map (Strength 2.0)"),
        1: (os.path.join(showcase_dir, "bump_blur_1.0.png"), "Bump Map (Blur 1.0)"),
        2: (os.path.join(showcase_dir, "roughness_scale_1.0.png"), "Roughness Map (Scale 1.0)"),cale 1.0)"),
        3: (os.path.join(showcase_dir, "ao_intensity_1.2.png"), "AO Map (Intensity 1.2)"),
        4: (os.path.join(showcase_dir, "material_rgbe.png"), "RGBE Material Map"),
        5: (os.path.join(output_dir, "displacement_map.png"), "Displacement Map"),
        6: (os.path.join(showcase_dir, "material_rg.png"), "RG Material Map"),
        7: (os.path.join(output_dir, "material_rgbe.png"), "Material RGBE Map"),),
        8: (os.path.join(output_dir, "smoothed_height_map.png"), "Smoothed Height Map")t Map")
    }
    
    for i, (file_path, title) in map_files.items():for i, (file_path, title) in map_files.items():
        try:
            img = plt.imread(file_path)img = plt.imread(file_path)
            axs[i].imshow(img)
            axs[i].set_title(title)itle)
            axs[i].axis('off')
        except Exception as e:
            print(f"Error displaying {file_path}: {e}")laying {file_path}: {e}")
    
    plt.tight_layout()plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "image_maps_summary.png"), dpi=300)h.join(output_dir, "image_maps_summary.png"), dpi=300)
    plt.close()
    
    print(f"Advanced image processing showcase saved to {showcase_dir}")print(f"Advanced image processing showcase saved to {showcase_dir}")

# ------------------ MAIN DEMO FUNCTION ------------------# ------------------ MAIN DEMO FUNCTION ------------------

def main():def main():
    """Main demonstration function showcasing TMD library capabilities.""" demonstration function showcasing TMD library capabilities."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)r, exist_ok=True)
    
    print("=" * 80)print("=" * 80)
    print("TMD (TrueMap Data) Library Demonstration")eMap Data) Library Demonstration")
    print("=" * 80)
    
    # Generate and process TMD file# Generate and process TMD file
    data, tmd_path = generate_and_process_tmd(output_dir)rocess_tmd(output_dir)
    height_map = data['height_map']
    
    # Visualize raw height map# Visualize raw height map
    visualize_raw_height_map(height_map, output_dir)eight_map, output_dir)
    
    # Height map manipulation# Height map manipulation
    manipulate_height_map_demo(height_map, output_dir, data)o(height_map, output_dir, data)
    
    # Gaussian filter demonstration# Gaussian filter demonstration
    gaussian_filter_demo(height_map, output_dir), output_dir)
    
    # Cross-section extraction# Cross-section extraction
    x_position = cross_section_demo(height_map, data, output_dir)_demo(height_map, data, output_dir)
    
    # Profile extraction at a specific percentage# Profile extraction at a specific percentage
    profile_demo(height_map, data, output_dir)
    
    # Export to various formats# Export to various formats
    export_formats_demo(height_map, data, output_dir)map, data, output_dir)
    
    # Advanced visualizations (if Plotly is available)# Advanced visualizations (if Plotly is available)
    advanced_visualizations_demo(height_map, data, output_dir, x_position)put_dir, x_position)
    
    # New: Showcase advanced image processing# New: Showcase advanced image processing
    image_processing_showcase(height_map, output_dir)put_dir)
    
    print("\n" + "=" * 80)print("\n" + "=" * 80)
    print("Demonstration complete! Output files are in the 'output' directory.")omplete! Output files are in the 'output' directory.")
    print("=" * 80)
    print("\nHere's a summary of the files created:") a summary of the files created:")
    print("  - Height maps: raw, cropped, rotated, thresholded, smoothed")resholded, smoothed")
    print("  - Cross-sections and profiles")
    print("  - Export formats:")
    print("    - NumPy files, metadata")tadata")
    print("    - Image maps:")
    print("      - Displacement, normal, bump, ambient occlusion (AO)")t, normal, bump, ambient occlusion (AO)")
    print("      - Roughness, multi-channel materials")
    print("    - 3D models: STL (ASCII & Binary), OBJ, PLY")PLY")
    print("    - Visualizations: PNG images and HTML files")
    print("  - Advanced image showcase: comparison of parameters and map types")ters and map types")


if __name__ == "__main__":if __name__ == "__main__":
    main()
