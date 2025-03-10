"""
Seaborn-based visualization functions for TMD data.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Default settings
COLORBAR_LABEL = "Height (µm)"

def plot_height_map_seaborn(height_map, colorbar_label=None, filename="seaborn_height_map.png", partial_range=None):
    """
    Creates a heatmap visualization of the height map using Seaborn.
    
    Args:
        height_map: 2D numpy array of height values
        colorbar_label: Label for the color bar (default: "Height (µm)")
        filename: Name of the image file to save
        partial_range: Optional tuple (row_start, row_end, col_start, col_end) for partial rendering
        
    Returns:
        Matplotlib figure object
    """
    if colorbar_label is None:
        colorbar_label = COLORBAR_LABEL
        
    if partial_range is not None:
        height_map = height_map[partial_range[0]:partial_range[1], partial_range[2]:partial_range[3]]
        print(f"Partial render applied: rows {partial_range[0]}:{partial_range[1]}, cols {partial_range[2]}:{partial_range[3]}")
    
    # Set the Seaborn style
    sns.set(style="whitegrid")
    
    # Create figure and axis
    plt.figure(figsize=(12, 10))
    
    # Create the heatmap
    ax = sns.heatmap(height_map, cmap="viridis", cbar_kws={'label': colorbar_label})
    
    # Customize the plot
    ax.set_title("Height Map (Seaborn)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Seaborn height map saved to {filename}")
    
    return plt.gcf()

def plot_2d_heatmap_seaborn(height_map, colorbar_label=None, filename="seaborn_2d_heatmap.png"):
    """
    Creates a detailed 2D heatmap of the height map using Seaborn with additional annotations.
    
    Args:
        height_map: 2D numpy array of height values
        colorbar_label: Label for the color bar (default: "Height (µm)")
        filename: Name of the image file to save
        
    Returns:
        Matplotlib figure object
    """
    if colorbar_label is None:
        colorbar_label = COLORBAR_LABEL
    
    # Set the Seaborn style
    sns.set(style="ticks")
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create the heatmap
    sns_heatmap = sns.heatmap(
        height_map, 
        cmap="viridis", 
        cbar_kws={'label': colorbar_label},
        ax=ax
    )
    
    # Add contour lines to show levels
    rows, cols = height_map.shape
    if rows <= 1000 and cols <= 1000:  # Only for smaller maps to avoid excessive computation
        x = np.arange(0, cols, 1)
        y = np.arange(0, rows, 1)
        X, Y = np.meshgrid(x, y)
        levels = np.linspace(height_map.min(), height_map.max(), 10)
        contour = ax.contour(X, Y, height_map, levels=levels, colors='white', alpha=0.5, linewidths=0.5)
        
    # Customize the plot
    ax.set_title("Enhanced Height Map (Seaborn)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    # Save figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Enhanced Seaborn heatmap saved to {filename}")
    
    return fig
