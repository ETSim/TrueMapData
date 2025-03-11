"""
TrueMap & GelSight TMD Processor package.

A Python-based TMD file processor with visualization and export capabilities for height maps.
"""

# Import commonly used classes and functions for easy access
from .processor import TMDProcessor

def load_tmd(file_path: str):
    """
    Convenience function to load a TMD file.
    
    Args:
        file_path: Path to the TMD file
        
    Returns:
        Tuple of (metadata, height_map)
    """
    processor = TMDProcessor(file_path)
    processor.process()
    return processor.get_metadata(), processor.get_height_map()