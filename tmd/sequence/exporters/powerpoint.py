""".

PowerPoint exporter implementation for TMD sequence data.

This module provides functionality to export TMD sequence data as PowerPoint presentations.
"""

import os
import logging
import tempfile
import numpy as np
from typing import List, Dict, Any, Optional, Union

from .base import BaseExporter
from .image import ImageExporter

logger = logging.getLogger(__name__)

# Add these standalone functions to match imports in __init__.py
def export_sequence_to_pptx(
    sequence: 'TMDSequence',
    output_file: str,
    include_original: bool = True,
    include_transformed: bool = True,
    include_difference: bool = True,
    colormap: str = "viridis",
    title: str = "TMD Sequence Analysis",
    **kwargs
) -> Optional[str]:
    """.

    Export a TMD sequence to a PowerPoint presentation.
    
    Args:
        sequence: TMDSequence object with frames to export
        output_file: Output PowerPoint file path
        include_original: Whether to include original frames
        include_transformed: Whether to include transformed frames
        include_difference: Whether to include difference frames
        colormap: Colormap to use for visualizations
        title: Presentation title
        **kwargs: Additional options to pass to the exporter
        
    Returns:
        Path to the created file or None if failed
    """
    exporter = PowerPointExporter()
    
    # Get frames to export
    frames = []
    timestamps = []
    
    if include_transformed:
        # Get transformed frames
        transformed_frames = sequence.apply_transformations()
        if transformed_frames:
            frames.extend(transformed_frames)
            timestamps.extend(sequence.frame_timestamps)
    
    if include_original and not include_transformed:
        # Only include original if transformed not included
        original_frames = [frame['height_map'] for frame in sequence.frames]
        frames.extend(original_frames)
        timestamps.extend(sequence.frame_timestamps)
    
    if not frames:
        logger.warning("No frames to export to PowerPoint")
        return None
    
    # Create directory for the output file if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(output_file))
    os.makedirs(output_dir, exist_ok=True)
    
    # Use the export_normal_maps method which can export both heightmaps and normal maps
    result = exporter.export_normal_maps(
        frames_data=frames,
        output_dir=output_dir,
        timestamps=timestamps,
        format='pptx',
        title=title,
        output_file=output_file,
        colormap=colormap,
        heightmaps=True,  # Export heightmaps instead of normal maps
        **kwargs
    )
    
    if result and len(result) > 0:
        return result[0]
    return None

def export_sequence_by_parameter(
    sequences: Dict[Any, 'TMDSequence'],
    output_dir: str,
    output_template: str = "Sequence_{}.pptx",
    **kwargs
) -> List[str]:
    """.

    Export multiple sequences organized by a parameter to PowerPoint files.
    
    Args:
        sequences: Dictionary mapping parameter values to TMDSequence objects
        output_dir: Directory to save the output files
        output_template: Template for output filenames
        **kwargs: Additional arguments to pass to export_sequence_to_pptx
        
    Returns:
        List of paths to created PowerPoint files
    """
    output_files = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for param_value, sequence in sequences.items():
        # Create output filename
        output_file = os.path.join(output_dir, output_template.format(param_value))
        
        # Update title if not provided
        if 'title' not in kwargs:
            kwargs['title'] = f"Sequence Analysis - Parameter {param_value}"
        
        # Export this sequence
        result = export_sequence_to_pptx(
            sequence=sequence, 
            output_file=output_file,
            **kwargs
        )
        
        if result:
            output_files.append(result)
    
    return output_files

class PowerPointExporter(BaseExporter):
    """PowerPoint exporter for TMD sequence data.."""
    
    def __init__(self):
        """Initialize the PowerPoint exporter.."""
        self._has_pptx = self._check_pptx()
        self._image_exporter = ImageExporter()
    
    def _check_pptx(self) -> bool:
        """Check if python-pptx is available.."""
        try:
            import pptx
            return True
        except ImportError:
            logger.warning("python-pptx not installed. Install with: pip install python-pptx")
            return False
    
    def export_sequence_differences(
        self,
        frames_data: List[np.ndarray],
        output_dir: str,
        timestamps: List[Any] = None,
        format: str = 'pptx',
        normalize: bool = True,
        colormap: str = 'RdBu',
        **kwargs
    ) -> List[str]:
        """.

        Export differences between frames as a PowerPoint presentation.
        
        Args:
            frames_data: List of difference arrays
            output_dir: Directory to save the output files
            timestamps: Optional list of timestamps for each difference
            format: Output format (unused, always 'pptx')
            normalize: Whether to normalize difference values
            colormap: Color map for visualization
            **kwargs: Additional options
            
        Returns:
            List of paths to saved files
        """
        # Check dependencies
        if not self._has_pptx:
            logger.error("python-pptx is required for PowerPoint export.")
            return []
            
        # Ensure output directory exists
        if not self.ensure_output_dir(output_dir):
            return []
            
        # Use frame indices if no timestamps provided
        if timestamps is None:
            timestamps = [f"diff_{i}" for i in range(len(frames_data))]
        
        # Get presentation-specific options
        title = kwargs.get('title', "TMD Sequence Differences")
        width = kwargs.get('width', 10)  # inches
        height = kwargs.get('height', 7.5)  # inches
        
        output_files = []
        
        try:
            import pptx
            from pptx.util import Inches, Pt
            from pptx.enum.text import PP_ALIGN
            
            # Create a temporary directory for images
            with tempfile.TemporaryDirectory() as temp_dir:
                # First generate individual difference images
                image_files = self._image_exporter.export_sequence_differences(
                    frames_data=frames_data,
                    output_dir=temp_dir,
                    timestamps=timestamps,
                    format='png',
                    normalize=normalize,
                    colormap=colormap
                )
                
                # Create a new presentation
                prs = pptx.Presentation()
                
                # Set slide dimensions
                prs.slide_width = Inches(width)
                prs.slide_height = Inches(height)
                
                # Create title slide
                title_slide_layout = prs.slide_layouts[0]
                title_slide = prs.slides.add_slide(title_slide_layout)
                title_slide.shapes.title.text = title
                title_slide.placeholders[1].text = f"Sequence with {len(frames_data)} frames"
                
                # Add a slide for each difference image
                blank_slide_layout = prs.slide_layouts[6]  # Blank layout
                
                for i, image_path in enumerate(image_files):
                    # Create a new slide
                    slide = prs.slides.add_slide(blank_slide_layout)
                    
                    # Add a title
                    title_shape = slide.shapes.add_textbox(
                        Inches(0.5), Inches(0.5), Inches(9), Inches(0.75)
                    )
                    title_frame = title_shape.text_frame
                    title_frame.text = f"Difference: {timestamps[i]}"
                    title_frame.paragraphs[0].font.size = Pt(24)
                    title_frame.paragraphs[0].font.bold = True
                    title_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
                    
                    # Add the image
                    slide.shapes.add_picture(
                        image_path,
                        Inches(1.0),   # Left
                        Inches(1.5),   # Top
                        width=Inches(8.0),  # Width
                        height=Inches(5.0)  # Height
                    )
                
                # Create output filename
                output_path = os.path.join(output_dir, "difference_sequence.pptx")
                
                # Save the presentation
                prs.save(output_path)
                output_files.append(output_path)
                logger.info(f"Saved difference presentation to {output_path}")
                
        except Exception as e:
            logger.error(f"Error creating PowerPoint presentation: {e}")
        
        return output_files
    
    def export_normal_maps(
        self,
        frames_data: List[np.ndarray],
        output_dir: str,
        timestamps: List[Any] = None,
        format: str = 'pptx',
        z_scale: float = 10.0,
        output_file: Optional[str] = None,
        title: str = "TMD Sequence Normal Maps",
        heightmaps: bool = False,
        colormap: str = "viridis",
        **kwargs
    ) -> List[str]:
        """.

        Export normal maps of the frames as a PowerPoint presentation.
        
        Args:
            frames_data: List of heightmap arrays
            output_dir: Directory to save the output files
            timestamps: Optional list of timestamp strings for each frame
            format: Output format (unused, always 'pptx')
            z_scale: Z-scale factor for normal map generation
            output_file: Optional specific output file path
            title: Presentation title
            heightmaps: Whether to export heightmaps instead of normal maps
            colormap: Colormap to use for heightmap visualization
            **kwargs: Additional options
            
        Returns:
            List of paths to saved files
        """
        # Check dependencies
        if not self._has_pptx:
            logger.error("python-pptx is required for PowerPoint export.")
            return []
            
        # Ensure output directory exists
        if not self.ensure_output_dir(output_dir):
            return []
        
        # Override output_file if provided
        if output_file:
            output_path = output_file
        else:
            output_path = os.path.join(output_dir, "normal_map_sequence.pptx")
            
        # Use provided timestamps or generate default ones
        if timestamps is None or len(timestamps) != len(frames_data):
            timestamps = [f"frame_{i}" for i in range(len(frames_data))]
            
        # Get presentation-specific options
        width = kwargs.get('width', 10)  # inches
        height = kwargs.get('height', 7.5)  # inches
        
        output_files = []
        
        try:
            import pptx
            from pptx.util import Inches, Pt
            from pptx.enum.text import PP_ALIGN
            
            # Create a temporary directory for images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Generate images based on what we're exporting
                if heightmaps:
                    # Export heightmaps instead of normal maps
                    image_files = self._image_exporter.export_sequence(
                        frames_data=frames_data,
                        output_dir=temp_dir,
                        timestamps=timestamps,
                        format='png',
                        colormap=colormap
                    )
                    slide_title_prefix = "Heightmap"
                    slide_desc = "Height values visualized with colormap"
                else:
                    # First generate individual normal map images
                    image_files = self._image_exporter.export_normal_maps(
                        frames_data=frames_data,
                        output_dir=temp_dir,
                        timestamps=timestamps,
                        format='png',
                        z_scale=z_scale
                    )
                    slide_title_prefix = "Normal Map"
                    slide_desc = "Normal map RGB channels: R=X, G=Y, B=Z"
                
                # Create a new presentation
                prs = pptx.Presentation()
                
                # Set slide dimensions
                prs.slide_width = Inches(width)
                prs.slide_height = Inches(height)
                
                # Create title slide
                title_slide_layout = prs.slide_layouts[0]
                title_slide = prs.slides.add_slide(title_slide_layout)
                title_slide.shapes.title.text = title
                title_slide.placeholders[1].text = f"Sequence with {len(frames_data)} frames"
                
                # Add a slide for each image
                blank_slide_layout = prs.slide_layouts[6]  # Blank layout
                
                for i, image_path in enumerate(image_files):
                    # Create a new slide
                    slide = prs.slides.add_slide(blank_slide_layout)
                    
                    # Add a title
                    title_shape = slide.shapes.add_textbox(
                        Inches(0.5), Inches(0.5), Inches(9), Inches(0.75)
                    )
                    title_frame = title_shape.text_frame
                    title_frame.text = f"{slide_title_prefix}: {timestamps[i]}"
                    title_frame.paragraphs[0].font.size = Pt(24)
                    title_frame.paragraphs[0].font.bold = True
                    title_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
                    
                    # Add the image
                    slide.shapes.add_picture(
                        image_path,
                        Inches(1.0),   # Left
                        Inches(1.5),   # Top
                        width=Inches(8.0),  # Width
                        height=Inches(5.0)  # Height
                    )
                    
                    # Add a description
                    desc_shape = slide.shapes.add_textbox(
                        Inches(1.0), Inches(6.5), Inches(8.0), Inches(0.5)
                    )
                    desc_frame = desc_shape.text_frame
                    desc_frame.text = slide_desc
                    desc_frame.paragraphs[0].font.size = Pt(12)
                    desc_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
                
                # Save the presentation
                prs.save(output_path)
                output_files.append(output_path)
                logger.info(f"Saved presentation to {output_path}")
                
        except Exception as e:
            logger.error(f"Error creating PowerPoint presentation: {e}")
        
        return output_files
