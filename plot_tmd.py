#!/usr/bin/env python3
"""
TMD File Visualization Script

This script reads TMD files and visualizes them using matplotlib.
It supports both 2D heatmaps and 3D surface plots.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, needed for 3d projection
from matplotlib import cm

# Import TMD utility functions
from tmd.utils.utils import process_tmd_file
from tmd.plotters.matplotlib import (
    plot_height_map_matplotlib,
    plot_2d_heatmap_matplotlib,
    plot_x_profile_matplotlib
)


def plot_tmd(file_path, output_dir=None, plot_type='2d', cross_section=None):
    """
    Read a TMD file and create visualizations.
    
    Args:
        file_path: Path to the TMD file
        output_dir: Directory to save output files (default: same as input file)
        plot_type: Type of plot ('2d', '3d', 'profile', or 'all')
        cross_section: Row index for profile (if plot_type includes 'profile')
    
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(file_path))
    
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except:
            print(f"Error: Could not create output directory: {output_dir}")
            return False
    
    # Process the TMD file
    try:
        print(f"Processing file: {file_path}")
        metadata, height_map = process_tmd_file(file_path)
    except Exception as e:
        print(f"Error processing TMD file: {e}")
        return False
    
    # Print some metadata
    print(f"TMD file loaded successfully:")
    print(f"  Dimensions: {metadata['width']} x {metadata['height']}")
    print(f"  X length: {metadata['x_length']}, Y length: {metadata['y_length']}")
    print(f"  X offset: {metadata['x_offset']}, Y offset: {metadata['y_offset']}")
    
    # Get base filename without extension
    basename = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create visualizations based on plot_type
    if plot_type in ['2d', 'all']:
        # Create 2D heatmap
        heatmap_path = os.path.join(output_dir, f"{basename}_heatmap.png")
        print(f"Generating 2D heatmap: {heatmap_path}")
        try:
            plot_2d_heatmap_matplotlib(
                height_map, 
                filename=heatmap_path,
                colorbar_label="Height"
            )
        except Exception as e:
            print(f"Warning: Could not create 2D heatmap: {e}")
    
    if plot_type in ['3d', 'all']:
        # Create 3D surface plot
        surface_path = os.path.join(output_dir, f"{basename}_surface.png")
        print(f"Generating 3D surface plot: {surface_path}")
        try:
            plot_height_map_matplotlib(
                height_map,
                filename=surface_path,
                colorbar_label="Height"
            )
        except Exception as e:
            print(f"Warning: Could not create 3D surface plot: {e}")
    
    if plot_type in ['profile', 'all']:
        # Create profile plot (cross-section)
        profile_row = cross_section
        if profile_row is None:
            # Default to middle row
            profile_row = height_map.shape[0] // 2
            
        if profile_row >= height_map.shape[0]:
            print(f"Warning: Cross section row {profile_row} is out of bounds. Using middle row.")
            profile_row = height_map.shape[0] // 2
        
        profile_path = os.path.join(output_dir, f"{basename}_profile_row{profile_row}.png")
        print(f"Generating profile plot at row {profile_row}: {profile_path}")
        
        try:
            # Create data dictionary for the plotter
            data_dict = {
                'height_map': height_map,
                'width': metadata['width'],
                'height': metadata['height'],
                'x_offset': metadata['x_offset'],
                'y_offset': metadata['y_offset'],
                'x_length': metadata['x_length'],
                'y_length': metadata['y_length']
            }
            
            plot_x_profile_matplotlib(data_dict, profile_row=profile_row, filename=profile_path)
        except Exception as e:
            print(f"Warning: Could not create profile plot: {e}")
    
    print("Visualization complete!")
    return True


def main():
    """Parse command-line arguments and execute main function."""
    parser = argparse.ArgumentParser(description="Visualize TMD files using matplotlib")
    
    parser.add_argument("tmd_file", help="Path to TMD file")
    parser.add_argument(
        "-o", "--output", 
        help="Output directory (default: same as input file)"
    )
    parser.add_argument(
        "-t", "--type", 
        choices=["2d", "3d", "profile", "all"], 
        default="all",
        help="Type of plot to generate (default: all)"
    )
    parser.add_argument(
        "-c", "--cross-section", 
        type=int, 
        help="Row index for profile cross-section (default: middle row)"
    )
    
    args = parser.parse_args()
    
    # Call the main function
    success = plot_tmd(
        args.tmd_file, 
        output_dir=args.output, 
        plot_type=args.type,
        cross_section=args.cross_section
    )
    
    # Set exit code based on success
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
