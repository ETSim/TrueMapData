"""
TMD Library Demo - SDF and NVBD Export Demonstration

This script demonstrates:
1. Creating synthetic height maps
2. Exporting to SDF (Signed Distance Field) format
3. Exporting to NVBD (NVIDIA Blast Destruction) format
4. Comparing different export settings
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add parent directory to path to allow imports from tmd package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tmd.utils.utils import create_sample_height_map, generate_synthetic_tmd
from tmd.processor import TMDProcessor
from tmd.exporters.sdf import export_heightmap_to_sdf
from tmd.exporters.nvbd import export_heightmap_to_nvbd


def ensure_output_dir(base_dir="output/sdf_nvbd_demo"):
    """Create output directory if it doesn't exist."""
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def create_test_height_maps(output_dir):
    """Create different test height maps for the demos."""
    
    print("\nGenerating test height maps...")
    height_maps = {}
    
    # Simple patterns
    patterns = {
        "waves": "Sinusoidal wave pattern",
        "peak": "Central peak pattern",
        "dome": "Dome pattern",
        "ramp": "Gradient ramp pattern",
        "combined": "Combined pattern with peaks and waves"
    }
    
    # Generate each pattern and visualize
    fig, axs = plt.subplots(len(patterns), 2, figsize=(12, 4*len(patterns)))
    
    for i, (pattern, description) in enumerate(patterns.items()):
        # Create the height map
        height_map = create_sample_height_map(
            width=200, 
            height=200, 
            pattern=pattern,
            noise_level=0.05
        )
        height_maps[pattern] = height_map
        
        # 2D visualization
        axs[i, 0].imshow(height_map, cmap='viridis')
        axs[i, 0].set_title(f"{pattern.capitalize()} - 2D View")
        axs[i, 0].axis('off')
        
        # 3D visualization - create a downsampled version for speed
        downsampled = height_map[::4, ::4]
        y, x = np.mgrid[0:downsampled.shape[0], 0:downsampled.shape[1]]
        
        # Create a 3D plot
        axs[i, 1] = fig.add_subplot(len(patterns), 2, i*2+2, projection='3d')
        surf = axs[i, 1].plot_surface(x, y, downsampled, cmap='viridis', 
                                     linewidth=0, antialiased=False, alpha=0.7)
        axs[i, 1].set_title(f"{pattern.capitalize()} - 3D View")
        axs[i, 1].set_box_aspect((1, 1, 0.3)) # Set z-axis scale
        axs[i, 1].set_axis_off()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "height_map_patterns.png"), dpi=300)
    plt.close()
    
    print("Created height maps with patterns:", ", ".join(patterns.keys()))
    return height_maps


def export_sdf_demo(height_maps, output_dir):
    """Demonstrate SDF export with different settings."""
    
    print("\nDemonstrating SDF export functionality...")
    results = []
    
    # Select two height maps for the demo
    demo_maps = {
        "peak": height_maps["peak"],
        "waves": height_maps["waves"]
    }
    
    # Try different scale factors
    scales = [0.5, 1.0, 2.0]
    
    for name, height_map in demo_maps.items():
        for scale in scales:
            output_file = os.path.join(output_dir, f"{name}_scale_{scale}.sdf")
            success = export_heightmap_to_sdf(
                height_map,
                output_file,
                scale=scale,
                offset=0.0
            )
            results.append({
                "pattern": name,
                "scale": scale,
                "success": success,
                "file_path": output_file,
                "file_size": os.path.getsize(output_file) if os.path.exists(output_file) else 0
            })
    
    # Create a summary table
    print("\nSDF Export Results:")
    print("-" * 60)
    print(f"{'Pattern':<10} | {'Scale':<6} | {'Success':<8} | {'File Size':<10} | {'Path'}")
    print("-" * 60)
    for result in results:
        print(f"{result['pattern']:<10} | {result['scale']:<6.1f} | {str(result['success']):<8} | "
              f"{result['file_size']:<10} | {os.path.basename(result['file_path'])}")
    
    print("\nSDF files can be used for physics simulations, distance field rendering,")
    print("and other applications that require signed distance fields.")


def export_nvbd_demo(height_maps, output_dir):
    """Demonstrate NVBD export with different settings."""
    
    print("\nDemonstrating NVBD export functionality...")
    results = []
    
    # Select a height map for the demo
    dome_map = height_maps["dome"]
    combined_map = height_maps["combined"]
    
    # Try different chunk sizes
    chunk_sizes = [8, 16, 32, 64]
    
    for chunk_size in chunk_sizes:
        output_file = os.path.join(output_dir, f"dome_chunk_{chunk_size}.nvbd")
        success = export_heightmap_to_nvbd(
            dome_map,
            output_file,
            scale=1.0,
            offset=0.0,
            chunk_size=chunk_size
        )
        results.append({
            "pattern": "dome",
            "chunk_size": chunk_size,
            "success": success,
            "file_path": output_file,
            "file_size": os.path.getsize(output_file) if os.path.exists(output_file) else 0
        })
    
    # Try different scales with fixed chunk size
    scales = [0.5, 1.0, 2.0]
    for scale in scales:
        output_file = os.path.join(output_dir, f"combined_scale_{scale}.nvbd")
        success = export_heightmap_to_nvbd(
            combined_map,
            output_file,
            scale=scale,
            offset=0.0,
            chunk_size=16
        )
        results.append({
            "pattern": "combined",
            "chunk_size": 16,
            "scale": scale,
            "success": success,
            "file_path": output_file,
            "file_size": os.path.getsize(output_file) if os.path.exists(output_file) else 0
        })
    
    # Create a summary table
    print("\nNVBD Export Results:")
    print("-" * 70)
    print(f"{'Pattern':<10} | {'Chunk Size':<10} | {'Scale':<6} | {'Success':<8} | {'File Size':<10} | {'Path'}")
    print("-" * 70)
    for result in results:
        scale = result.get('scale', 1.0)
        print(f"{result['pattern']:<10} | {result['chunk_size']:<10} | {scale:<6.1f} | "
              f"{str(result['success']):<8} | {result['file_size']:<10} | {os.path.basename(result['file_path'])}")
    
    print("\nNVBD files are used for destruction simulations and physics-based")
    print("effects in game engines and visual effects software.")


def analyze_tmd_file(tmd_path, output_dir):
    """Analyze a TMD file and export it to SDF and NVBD formats."""
    
    print(f"\nAnalyzing TMD file: {tmd_path}")
    processor = TMDProcessor(tmd_path)
    processor.set_debug(True)
    data = processor.process()
    
    if not data:
        print("Failed to process TMD file.")
        return None
    
    # Get the height map
    height_map = data['height_map']
    
    # Export to SDF
    sdf_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(tmd_path))[0]}.sdf")
    sdf_success = export_heightmap_to_sdf(height_map, sdf_path)
    
    # Export to NVBD with different chunk sizes
    chunk_sizes = [16, 32]
    nvbd_results = []
    
    for chunk_size in chunk_sizes:
        nvbd_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(tmd_path))[0]}_chunk_{chunk_size}.nvbd")
        nvbd_success = export_heightmap_to_nvbd(height_map, nvbd_path, chunk_size=chunk_size)
        nvbd_results.append((chunk_size, nvbd_success, nvbd_path))
    
    # Print results
    print(f"\nExport results for {os.path.basename(tmd_path)}:")
    print(f"SDF: {'Success' if sdf_success else 'Failed'} - {sdf_path}")
    for chunk_size, success, path in nvbd_results:
        print(f"NVBD (chunk_size={chunk_size}): {'Success' if success else 'Failed'} - {path}")
    
    return data


def synthetic_tmd_export_demo(output_dir):
    """Generate a synthetic TMD file and export it to SDF and NVBD formats."""
    
    print("\nGenerating and exporting a synthetic TMD file...")
    
    # Create a synthetic TMD file
    tmd_path = os.path.join(output_dir, "synthetic_export_demo.tmd")
    tmd_path = generate_synthetic_tmd(
        output_path=tmd_path,
        width=200,
        height=200,
        pattern="combined",
        comment="SDF/NVBD Export Demo"
    )
    
    # Analyze and export the synthetic TMD
    data = analyze_tmd_file(tmd_path, output_dir)
    
    if data:
        print("\nSuccessfully created and exported synthetic TMD file.")
        return True
    else:
        print("\nFailed to create or export synthetic TMD file.")
        return False


def main():
    """Main function to demonstrate SDF and NVBD export capabilities."""
    
    print("=" * 80)
    print("TMD Library - SDF and NVBD Export Demonstration")
    print("=" * 80)
    
    # Create output directory
    output_dir = ensure_output_dir()
    print(f"Output directory: {output_dir}")
    
    # Create test height maps
    height_maps = create_test_height_maps(output_dir)
    
    # Demonstrate SDF export
    export_sdf_demo(height_maps, output_dir)
    
    # Demonstrate NVBD export
    export_nvbd_demo(height_maps, output_dir)
    
    # Demonstrate synthetic TMD export
    synthetic_tmd_export_demo(output_dir)
    
    print("\n" + "=" * 80)
    print("SDF and NVBD Export Demo complete!")
    print("=" * 80)
    print(f"\nOutput files can be found in: {output_dir}")
    print("\nSDF files are Signed Distance Fields that can be used in:")
    print("  - 3D modeling software")
    print("  - Physics simulations")
    print("  - Ray marching renderers")
    print("\nNVBD files can be used in:")
    print("  - NVIDIA Blast destruction simulations")
    print("  - Game engines for real-time physics")
    print("  - Visual effects software")


if __name__ == "__main__":
    main()
