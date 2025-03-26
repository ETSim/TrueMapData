"""
Sequence comparison module for TMD.

This module provides functionality for comparing multiple TMD sequences.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Import local
from .sequence import TMDSequence

# Set up logging
logger = logging.getLogger(__name__)

class TMDSequenceComparator:
    """
    Class for comparing multiple TMD sequences.
    
    This class provides functionality for comparing multiple TMD sequences, calculating
    differences between them, and visualizing the results.
    """
    
    def __init__(self):
        """Initialize a sequence comparator."""
        self.sequences = []
        self.sequence_names = []
        self.frame_differences = {}
        self.statistical_differences = {}
    
    def add_sequence(self, sequence: TMDSequence, name: Optional[str] = None) -> int:
        """
        Add a sequence to the comparator.
        
        Args:
            sequence: TMDSequence object to add
            name: Optional name for the sequence (defaults to sequence.name)
            
        Returns:
            Index of the added sequence
        """
        if not isinstance(sequence, TMDSequence):
            logger.error("Only TMDSequence objects can be added to the comparator")
            return -1
            
        # Use sequence name if no name provided
        if name is None:
            name = sequence.name
            
        self.sequences.append(sequence)
        self.sequence_names.append(name)
        
        # Clear cached results
        self.frame_differences = {}
        self.statistical_differences = {}
        
        return len(self.sequences) - 1
    
    def calculate_frame_wise_differences(self) -> Dict[Tuple[int, int], List[np.ndarray]]:
        """
        Calculate frame-wise differences between sequences.
        
        Returns:
            Dictionary mapping (seq1_idx, seq2_idx) to list of difference arrays
        """
        if len(self.sequences) < 2:
            logger.warning("Need at least 2 sequences to calculate differences")
            return {}
            
        # Use cached results if available
        if self.frame_differences:
            return self.frame_differences
            
        differences = {}
        
        # Compare each pair of sequences
        for i in range(len(self.sequences)):
            for j in range(i + 1, len(self.sequences)):
                seq1 = self.sequences[i]
                seq2 = self.sequences[j]
                
                # Get frames from both sequences
                frames1 = seq1.apply_transformations() if hasattr(seq1, 'apply_transformations') else seq1.get_all_frames()
                frames2 = seq2.apply_transformations() if hasattr(seq2, 'apply_transformations') else seq2.get_all_frames()
                
                # Calculate differences for each frame
                frame_diffs = []
                min_frames = min(len(frames1), len(frames2))
                
                for k in range(min_frames):
                    frame_diffs.append(frames2[k] - frames1[k])
                    
                differences[(i, j)] = frame_diffs
                
        # Store results
        self.frame_differences = differences
        
        return differences
    
    def calculate_statistical_differences(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        """
        Calculate statistical differences between sequences.
        
        Returns:
            Dictionary mapping (seq1_idx, seq2_idx) to dictionary of statistical differences
        """
        if len(self.sequences) < 2:
            logger.warning("Need at least 2 sequences to calculate statistical differences")
            return {}
            
        # Use cached results if available
        if self.statistical_differences:
            return self.statistical_differences
            
        stat_diffs = {}
        
        # Compare each pair of sequences
        for i in range(len(self.sequences)):
            for j in range(i + 1, len(self.sequences)):
                seq1 = self.sequences[i]
                seq2 = self.sequences[j]
                
                # Get statistics for both sequences
                stats1 = seq1.calculate_statistics()
                stats2 = seq2.calculate_statistics()
                
                # Calculate differences for each statistical measure
                diff_stats = {}
                
                for key in ['min', 'max', 'mean', 'median', 'std', 'range', 'sum']:
                    if key in stats1 and key in stats2:
                        # Calculate absolute and relative differences
                        abs_diff = [b - a for a, b in zip(stats1[key][:min(len(stats1[key]), len(stats2[key]))], 
                                                           stats2[key][:min(len(stats1[key]), len(stats2[key]))])]
                        
                        # Avoid division by zero for relative differences
                        rel_diff = []
                        for a, b in zip(stats1[key][:min(len(stats1[key]), len(stats2[key]))], 
                                       stats2[key][:min(len(stats1[key]), len(stats2[key]))]):
                            if a != 0:
                                rel_diff.append((b - a) / abs(a) * 100)  # Percent difference
                            else:
                                rel_diff.append(float('inf') if b != 0 else 0)
                                
                        diff_stats[f'{key}_abs_diff'] = abs_diff
                        diff_stats[f'{key}_rel_diff'] = rel_diff
                        
                # Store timestamps
                diff_stats['timestamps'] = stats1['timestamps'][:min(len(stats1['timestamps']), len(stats2['timestamps']))]
                
                stat_diffs[(i, j)] = diff_stats
                
        # Store results
        self.statistical_differences = stat_diffs
        
        return stat_diffs
    
    def visualize_frame_differences(
        self,
        output_dir: Optional[str] = None,
        frame_indices: Optional[List[int]] = None,
        colormap: str = 'RdBu',
        show: bool = True,
        **kwargs
    ) -> List[Any]:
        """
        Visualize frame-wise differences between sequences.
        
        Args:
            output_dir: Optional directory to save visualizations
            frame_indices: Optional list of frame indices to visualize
            colormap: Colormap to use for visualization
            show: Whether to display the plots
            **kwargs: Additional visualization options
            
        Returns:
            List of created figure objects
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib is not installed. Cannot visualize differences.")
            return []
            
        # Calculate frame-wise differences
        differences = self.calculate_frame_wise_differences()
        
        if not differences:
            logger.warning("No differences to visualize")
            return []
            
        figures = []
        
        # For each pair of sequences
        for (i, j), diff_frames in differences.items():
            seq1_name = self.sequence_names[i]
            seq2_name = self.sequence_names[j]
            
            # Determine which frames to visualize
            if frame_indices is None:
                frames_to_viz = list(range(len(diff_frames)))
            else:
                frames_to_viz = [idx for idx in frame_indices if 0 <= idx < len(diff_frames)]
                
            if not frames_to_viz:
                logger.warning(f"No valid frame indices to visualize for {seq1_name} vs {seq2_name}")
                continue
                
            # Get timestamps from both sequences
            seq1_timestamps = self.sequences[i].get_all_timestamps()
            seq2_timestamps = self.sequences[j].get_all_timestamps()
            
            # Visualize each selected frame
            for frame_idx in frames_to_viz:
                diff = diff_frames[frame_idx]
                
                # Create figure
                fig = plt.figure(figsize=kwargs.get('figsize', (10, 8)))
                ax = fig.add_subplot(111)
                
                # Normalize difference for better visualization
                if kwargs.get('normalize', True):
                    vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
                    vmin = -vmax
                else:
                    vmin = kwargs.get('vmin', np.nanmin(diff))
                    vmax = kwargs.get('vmax', np.nanmax(diff))
                
                # Plot the difference
                im = ax.imshow(diff, cmap=colormap, origin='lower', vmin=vmin, vmax=vmax)
                
                # Add colorbar
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('Difference')
                
                # Add title
                timestamp1 = seq1_timestamps[frame_idx] if frame_idx < len(seq1_timestamps) else f"Frame {frame_idx+1}"
                timestamp2 = seq2_timestamps[frame_idx] if frame_idx < len(seq2_timestamps) else f"Frame {frame_idx+1}"
                
                ax.set_title(f"Difference: {seq2_name} - {seq1_name}\n{timestamp2} vs {timestamp1}")
                
                # Save if output directory provided
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    filename = os.path.join(output_dir, f"diff_{seq1_name}_vs_{seq2_name}_frame_{frame_idx}.png")
                    plt.savefig(filename, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
                    logger.info(f"Saved difference visualization to {filename}")
                
                # Display if requested
                if show:
                    plt.show()
                else:
                    plt.close(fig)
                    
                figures.append(fig)
                
        return figures
    
    def visualize_statistical_comparison(self, *args, **kwargs):
        """
        Visualize the statistical comparison between sequences.
        
        Returns:
            tuple: (figure, axes) matplotlib objects
        """
        # Ensure this method always returns a valid tuple with two elements
        try:
            # Calculate statistical differences
            stat_diffs = self.calculate_statistical_differences()
            
            if not stat_diffs:
                logger.warning("No statistical differences to visualize")
                return []
                
            # Default metrics to visualize
            metrics = kwargs.get('metrics', ['mean', 'std', 'min', 'max'])
            
            figures = []
            
            # For each pair of sequences
            for (i, j), diff_stats in stat_diffs.items():
                seq1_name = self.sequence_names[i]
                seq2_name = self.sequence_names[j]
                
                # Get timestamps
                timestamps = diff_stats.get('timestamps', list(range(len(next(iter(diff_stats.values()))))))
                
                # Create a figure for each metric
                for metric in metrics:
                    abs_key = f'{metric}_abs_diff'
                    rel_key = f'{metric}_rel_diff'
                    
                    if abs_key not in diff_stats or rel_key not in diff_stats:
                        logger.warning(f"Metric '{metric}' not available for {seq1_name} vs {seq2_name}")
                        continue
                        
                    # Get absolute and relative differences
                    abs_diffs = diff_stats[abs_key]
                    rel_diffs = diff_stats[rel_key]
                    
                    # Create figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=kwargs.get('figsize', (12, 10)), sharex=True)
                    
                    # Plot absolute differences
                    ax1.plot(timestamps, abs_diffs, 'b-', marker='o')
                    ax1.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                    ax1.set_ylabel(f'Absolute Difference')
                    ax1.set_title(f"{metric.capitalize()} Difference: {seq2_name} - {seq1_name}")
                    ax1.grid(True, linestyle='--', alpha=0.7)
                    
                    # Plot relative differences
                    ax2.plot(timestamps, rel_diffs, 'g-', marker='o')
                    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.3)
                    ax2.set_xlabel('Time')
                    ax2.set_ylabel('Relative Difference (%)')
                    ax2.grid(True, linestyle='--', alpha=0.7)
                    
                    # Rotate x-axis labels if they're strings
                    if isinstance(timestamps[0], str):
                        plt.xticks(rotation=45)
                        
                    plt.tight_layout()
                    
                    # Save if output directory provided
                    if kwargs.get('output_dir'):
                        os.makedirs(kwargs['output_dir'], exist_ok=True)
                        filename = os.path.join(kwargs['output_dir'], f"{metric}_diff_{seq1_name}_vs_{seq2_name}.png")
                        plt.savefig(filename, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
                        logger.info(f"Saved statistical comparison to {filename}")
                    
                    # Display if requested
                    if kwargs.get('show', True):
                        plt.show()
                    else:
                        plt.close(fig)
                        
                    figures.append(fig)
                    
            return fig, [ax1, ax2]
        except Exception as e:
            # If we encounter an error, create a basic figure to return
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error generating visualization: {str(e)}", 
                    ha='center', va='center')
            return fig, [ax]

    def export_difference_maps(
        self,
        output_dir: str,
        format: str = "png",
        normalize: bool = True,
        colormap: str = "RdBu",
        **kwargs
    ) -> List[str]:
        """
        Export difference maps between sequences.
        
        Args:
            output_dir: Directory to save the difference maps
            format: Output format (e.g., 'png', 'jpg')
            normalize: Whether to normalize the difference values
            colormap: Colormap to use for visualization
            **kwargs: Additional export options
            
        Returns:
            List of exported file paths
        """
        try:
            from tmd.sequence.exporters.image import ImageExporter
        except ImportError:
            logger.error("ImageExporter not available. Cannot export difference maps.")
            return []
            
        # Calculate frame-wise differences
        differences = self.calculate_frame_wise_differences()
        
        if not differences:
            logger.warning("No differences to export")
            return []
            
        # Create exporter
        exporter = ImageExporter()
        
        all_exports = []
        
        # For each pair of sequences
        for (i, j), diff_frames in differences.items():
            seq1_name = self.sequence_names[i]
            seq2_name = self.sequence_names[j]
            
            # Get timestamps from both sequences
            seq1_timestamps = self.sequences[i].get_all_timestamps()
            seq2_timestamps = self.sequences[j].get_all_timestamps()
            
            # Create difference timestamps
            diff_timestamps = []
            for k in range(len(diff_frames)):
                ts1 = seq1_timestamps[k] if k < len(seq1_timestamps) else f"Frame {k+1}"
                ts2 = seq2_timestamps[k] if k < len(seq2_timestamps) else f"Frame {k+1}"
                diff_timestamps.append(f"{seq2_name}:{ts2} - {seq1_name}:{ts1}")
                
            # Export the differences
            exports = exporter.export_sequence_differences(
                frames_data=diff_frames,
                output_dir=output_dir,
                timestamps=diff_timestamps,
                format=format,
                normalize=normalize,
                colormap=colormap,
                **kwargs
            )
            
            all_exports.extend(exports)
            
        return all_exports
    
    def export_difference_report(
        self,
        output_file: str,
        include_stats: bool = True,
        include_frames: bool = True,
        **kwargs
    ) -> Optional[str]:
        """
        Export a comprehensive difference report.
        
        Args:
            output_file: Output file path
            include_stats: Whether to include statistical analysis
            include_frames: Whether to include frame-by-frame analysis
            **kwargs: Additional export options
            
        Returns:
            Path to the exported report or None if export failed
        """
        try:
            import pandas as pd
        except ImportError:
            logger.error("Pandas is required for exporting reports")
            return None
            
        # Calculate differences
        frame_diffs = self.calculate_frame_wise_differences() if include_frames else {}
        stat_diffs = self.calculate_statistical_differences() if include_stats else {}
        
        if not frame_diffs and not stat_diffs:
            logger.warning("No differences to report")
            return None
            
        # Create writer for Excel
        try:
            writer = pd.ExcelWriter(output_file, engine='openpyxl')
        except Exception as e:
            logger.error(f"Could not create Excel writer: {e}")
            return None
            
        # For each pair of sequences
        for (i, j) in set(frame_diffs.keys()) | set(stat_diffs.keys()):
            seq1_name = self.sequence_names[i]
            seq2_name = self.sequence_names[j]
            sheet_name = f"{seq1_name}_vs_{seq2_name}"[:31]  # Excel has 31 char sheet name limit
            
            # Add frame differences if available
            if (i, j) in frame_diffs and include_frames:
                diff_frames = frame_diffs[(i, j)]
                
                # Get timestamps
                seq1_ts = self.sequences[i].get_all_timestamps()
                seq2_ts = self.sequences[j].get_all_timestamps()
                
                # Create a DataFrame with statistics for each frame
                frame_stats = []
                for k, diff in enumerate(diff_frames):
                    ts1 = seq1_ts[k] if k < len(seq1_ts) else f"Frame {k+1}"
                    ts2 = seq2_ts[k] if k < len(seq2_ts) else f"Frame {k+1}"
                    
                    stats = {
                        'Frame': k+1,
                        f'{seq1_name} Timestamp': ts1,
                        f'{seq2_name} Timestamp': ts2,
                        'Min Diff': float(np.nanmin(diff)),
                        'Max Diff': float(np.nanmax(diff)),
                        'Mean Diff': float(np.nanmean(diff)),
                        'Std Dev Diff': float(np.nanstd(diff)),
                        'Absolute Mean Diff': float(np.nanmean(np.abs(diff))),
                    }
                    
                    frame_stats.append(stats)
                
                # Create DataFrame and write to Excel
                if frame_stats:
                    df_frames = pd.DataFrame(frame_stats)
                    df_frames.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Add statistical differences if available
            if (i, j) in stat_diffs and include_stats:
                diff_stats = stat_diffs[(i, j)]
                timestamps = diff_stats.get('timestamps', list(range(len(next(iter(diff_stats.values()))))))
                
                # Create a dictionary of statistical differences
                stats_dict = {'Timestamp': timestamps}
                
                for metric in ['min', 'max', 'mean', 'median', 'std', 'range']:
                    abs_key = f'{metric}_abs_diff'
                    rel_key = f'{metric}_rel_diff'
                    
                    if abs_key in diff_stats and rel_key in diff_stats:
                        stats_dict[f'{metric.capitalize()} Abs Diff'] = diff_stats[abs_key]
                        stats_dict[f'{metric.capitalize()} Rel Diff (%)'] = diff_stats[rel_key]
                
                # Create DataFrame and write to Excel
                if stats_dict:
                    sheet_name_stats = f"{sheet_name}_stats"[:31]
                    df_stats = pd.DataFrame(stats_dict)
                    df_stats.to_excel(writer, sheet_name=sheet_name_stats, index=False)
        
        # Save the workbook
        try:
            writer.close()
            logger.info(f"Saved difference report to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error saving Excel report: {e}")
            return None
    
    def __len__(self) -> int:
        """
        Get the number of sequences in the comparator.
        
        Returns:
            Number of sequences
        """
        return len(self.sequences)

def compare_heightmaps(
    source_map: np.ndarray,
    target_map: np.ndarray,
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Compare two height maps and calculate difference metrics.
    
    Args:
        source_map: Source height map
        target_map: Target height map
        normalize: Whether to normalize differences by height range
        
    Returns:
        Dictionary of difference metrics
    """
    if source_map.shape != target_map.shape:
        logger.warning(f"Height maps have different shapes: {source_map.shape} vs {target_map.shape}")
        # Resize to match smaller dimensions for comparison
        min_rows = min(source_map.shape[0], target_map.shape[0])
        min_cols = min(source_map.shape[1], target_map.shape[1])
        source_cropped = source_map[:min_rows, :min_cols]
        target_cropped = target_map[:min_rows, :min_cols]
    else:
        source_cropped = source_map
        target_cropped = target_map
    
    # Calculate difference
    diff = target_cropped - source_cropped
    
    # Calculate metrics
    metrics = calculate_difference_metrics(diff, source_cropped, target_cropped, normalize)
    
    return {
        "difference": diff,
        "metrics": metrics
    }

def calculate_difference_metrics(
    difference: np.ndarray,
    source_map: Optional[np.ndarray] = None,
    target_map: Optional[np.ndarray] = None,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Calculate metrics to quantify differences between height maps.
    
    Args:
        difference: Difference array (target - source)
        source_map: Source height map (optional)
        target_map: Target height map (optional)
        normalize: Whether to normalize by height range
        
    Returns:
        Dictionary of metrics
    """
    # Calculate basic statistics about the difference
    abs_diff = np.abs(difference)
    
    metrics = {
        "min_diff": float(np.nanmin(difference)),
        "max_diff": float(np.nanmax(difference)),
        "mean_diff": float(np.nanmean(difference)),
        "median_diff": float(np.nanmedian(difference)),
        "std_diff": float(np.nanstd(difference)),
        "mae": float(np.nanmean(abs_diff)),  # Mean Absolute Error
        "rmse": float(np.sqrt(np.nanmean(np.square(difference))))  # Root Mean Square Error
    }
    
    # Calculate normalized metrics if both maps are provided
    if normalize and source_map is not None and target_map is not None:
        # Get overall height range
        combined_min = min(np.nanmin(source_map), np.nanmin(target_map))
        combined_max = max(np.nanmax(source_map), np.nanmax(target_map))
        height_range = combined_max - combined_min
        
        if height_range > 0:
            metrics["normalized_mae"] = metrics["mae"] / height_range
            metrics["normalized_rmse"] = metrics["rmse"] / height_range
            
        # Calculate correlation coefficient
        valid_mask = ~(np.isnan(source_map) | np.isnan(target_map))
        if np.count_nonzero(valid_mask) > 1:
            correlation = np.corrcoef(
                source_map[valid_mask].flatten(),
                target_map[valid_mask].flatten()
            )[0, 1]
            metrics["correlation"] = float(correlation)
    
    return metrics

def create_comparison_visualizations(
    maps: List[np.ndarray],
    labels: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
    colormap: str = 'viridis',
    show: bool = True,
    **kwargs
) -> List[Any]:
    """
    Create visualizations to compare multiple height maps.
    
    Args:
        maps: List of height maps to compare
        labels: Optional list of labels for each map
        output_dir: Optional directory to save visualizations
        colormap: Colormap for visualization
        show: Whether to show the plots
        **kwargs: Additional visualization options
        
    Returns:
        List of created figure objects
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("Matplotlib is not installed. Cannot create visualizations.")
        return []
    
    if not maps:
        logger.warning("No maps to visualize")
        return []
    
    # Default labels if not provided
    if labels is None:
        labels = [f"Map {i+1}" for i in range(len(maps))]
    
    figures = []
    
    # 1. Create individual visualizations
    for i, (height_map, label) in enumerate(zip(maps, labels)):
        fig = plt.figure(figsize=kwargs.get('figsize', (10, 8)))
        ax = fig.add_subplot(111)
        
        im = ax.imshow(height_map, cmap=colormap, origin='lower')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Height')
        
        ax.set_title(label)
        
        # Save if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{label.replace(' ', '_')}.png")
            plt.savefig(filename, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
            logger.info(f"Saved visualization to {filename}")
        
        # Show if requested
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        figures.append(fig)
    
    # 2. Create difference visualizations for consecutive pairs
    if len(maps) >= 2:
        for i in range(len(maps) - 1):
            diff = maps[i+1] - maps[i]
            
            fig = plt.figure(figsize=kwargs.get('figsize', (10, 8)))
            ax = fig.add_subplot(111)
            
            # Normalize difference for better visualization
            vmax = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
            vmin = -vmax
            
            im = ax.imshow(diff, cmap='RdBu', origin='lower', vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Difference')
            
            ax.set_title(f"Difference: {labels[i+1]} - {labels[i]}")
            
            # Save if output directory provided
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, f"diff_{labels[i]}_vs_{labels[i+1]}.png")
                plt.savefig(filename, dpi=kwargs.get('dpi', 300), bbox_inches='tight')
                logger.info(f"Saved difference visualization to {filename}")
            
            # Show if requested
            if show:
                plt.show()
            else:
                plt.close(fig)
            
            figures.append(fig)
    
    return figures

"""
Comparison module for height map sequences.

This module provides functionality to compare multiple height map sequences
using various metrics and visualization methods.
"""

import os
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

# Set up logger
logger = logging.getLogger(__name__)


def calculate_sequence_differences(
    sequence_a: List[np.ndarray],
    sequence_b: List[np.ndarray],
    metric: str = "mse",
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Calculate differences between two height map sequences.
    
    Args:
        sequence_a: First sequence of height maps
        sequence_b: Second sequence of height maps
        metric: Difference metric to use ("mse", "mae", "rmse", "correlation")
        normalize: Whether to normalize height maps before comparison
        
    Returns:
        Dictionary containing difference metrics
    """
    # Check input sequences
    if not sequence_a or not sequence_b:
        raise ValueError("Empty sequence provided")
        
    # Check sequence lengths
    if len(sequence_a) != len(sequence_b):
        raise ValueError(f"Sequences have different lengths: {len(sequence_a)} vs {len(sequence_b)}")
        
    # Check that frames have the same shape
    if sequence_a[0].shape != sequence_b[0].shape:
        raise ValueError(f"Frames have different shapes: {sequence_a[0].shape} vs {sequence_b[0].shape}")
    
    # Normalize sequences if requested
    if normalize:
        # Function to normalize a single heightmap to 0-1 range
        def normalize_heightmap(heightmap):
            min_val = np.min(heightmap)
            max_val = np.max(heightmap)
            if max_val > min_val:
                return (heightmap - min_val) / (max_val - min_val)
            return np.zeros_like(heightmap)
            
        sequence_a = [normalize_heightmap(frame) for frame in sequence_a]
        sequence_b = [normalize_heightmap(frame) for frame in sequence_b]
    
    # Calculate frame-by-frame differences
    frame_differences = []
    for i in range(len(sequence_a)):
        frame_a = sequence_a[i]
        frame_b = sequence_b[i]
        
        # Calculate difference based on specified metric
        if metric == "mse":
            # Mean Squared Error
            diff = np.mean((frame_a - frame_b) ** 2)
        elif metric == "mae":
            # Mean Absolute Error
            diff = np.mean(np.abs(frame_a - frame_b))
        elif metric == "rmse":
            # Root Mean Squared Error
            diff = np.sqrt(np.mean((frame_a - frame_b) ** 2))
        elif metric == "correlation":
            # Correlation coefficient
            # Flatten arrays
            flat_a = frame_a.flatten()
            flat_b = frame_b.flatten()
            # Calculate correlation
            corr = np.corrcoef(flat_a, flat_b)[0, 1]
            diff = 1.0 - corr  # Convert to difference (0 = identical)
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        frame_differences.append(diff)
    
    # Calculate aggregate statistics
    results = {
        "metric": metric,
        "frame_differences": frame_differences,
        "mean_difference": np.mean(frame_differences),
        "max_difference": np.max(frame_differences),
        "min_difference": np.min(frame_differences),
        "std_difference": np.std(frame_differences),
        "sequence_length": len(sequence_a)
    }
    
    return results


def visualize_sequence_differences(
    sequence_a: List[np.ndarray],
    sequence_b: List[np.ndarray],
    output_file: Optional[str] = None,
    title: str = "Sequence Comparison",
    show_progress: bool = True,
    colormap: str = "RdBu_r",
    normalize: bool = True,
    display: bool = False
) -> Optional[str]:
    """
    Create visualization of differences between height map sequences.
    
    Args:
        sequence_a: First sequence of height maps
        sequence_b: Second sequence of height maps
        output_file: Path to save visualization (image or video)
        title: Title for the visualization
        show_progress: Whether to show a progress bar
        colormap: Colormap for difference visualization
        normalize: Whether to normalize height maps before comparison
        display: Whether to display the visualization (for interactive use)
        
    Returns:
        Path to the output file or None if no file was saved
    """
    try:
        # Check for necessary packages
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from tqdm import tqdm
        except ImportError as e:
            logger.error(f"Required package not found: {e}")
            return None
            
        # Check input sequences
        if not sequence_a or not sequence_b:
            logger.error("Empty sequence provided")
            return None
            
        # Check sequence lengths
        if len(sequence_a) != len(sequence_b):
            logger.error(f"Sequences have different lengths: {len(sequence_a)} vs {len(sequence_b)}")
            return None
            
        # Check that frames have the same shape
        if sequence_a[0].shape != sequence_b[0].shape:
            logger.error(f"Frames have different shapes: {sequence_a[0].shape} vs {sequence_b[0].shape}")
            return None
        
        # Normalize sequences if requested
        if normalize:
            # Function to normalize a single heightmap to 0-1 range
            def normalize_heightmap(heightmap):
                min_val = np.min(heightmap)
                max_val = np.max(heightmap)
                if max_val > min_val:
                    return (heightmap - min_val) / (max_val - min_val)
                return np.zeros_like(heightmap)
                
            sequence_a = [normalize_heightmap(frame) for frame in sequence_a]
            sequence_b = [normalize_heightmap(frame) for frame in sequence_b]
        
        # Create figure with subplots
        n_frames = len(sequence_a)
        
        # Single frame comparison for image output, or multi-frame for video
        if output_file and output_file.lower().endswith(('.mp4', '.avi', '.mov', '.gif')):
            # Create video comparison
            from .exporters.video import export_sequence_to_video
            
            # Calculate differences
            diff_frames = []
            iterator = tqdm(range(n_frames), desc="Calculating differences") if show_progress else range(n_frames)
            
            for i in iterator:
                diff = sequence_a[i] - sequence_b[i]
                diff_frames.append(diff)
            
            # Export video
            return export_sequence_to_video(
                frames=diff_frames,
                output_file=output_file,
                title=title,
                colormap=colormap,
                show_progress=show_progress,
                fps=10.0
            )
        else:
            # Static image showing multiple frames
            # Determine grid layout
            max_frames = min(n_frames, 6)  # Show up to 6 frames
            cols = min(3, max_frames)
            rows = (max_frames + cols - 1) // cols
            
            # Create figure
            fig = plt.figure(figsize=(5 * cols, 4 * rows))
            fig.suptitle(title, fontsize=16)
            
            # Sample frames evenly
            indices = np.linspace(0, n_frames-1, max_frames, dtype=int)
            
            for i, idx in enumerate(indices):
                # Add subplot
                ax = fig.add_subplot(rows, cols, i + 1)
                
                # Calculate difference
                diff = sequence_a[idx] - sequence_b[idx]
                
                # Determine symmetric colormap range
                vmax = max(abs(np.max(diff)), abs(np.min(diff)))
                vmin = -vmax
                
                # Plot difference
                im = ax.imshow(diff, cmap=colormap, vmin=vmin, vmax=vmax)
                ax.set_title(f"Frame {idx}")
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save or display
            if output_file:
                # Create directory if necessary
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                
                # Save figure
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                logger.info(f"Comparison visualization saved to {output_file}")
                
            if display:
                plt.show()
            else:
                plt.close()
                
            return output_file
            
    except Exception as e:
        logger.error(f"Error visualizing sequence differences: {e}")
        import traceback
        traceback.print_exc()
        return None
