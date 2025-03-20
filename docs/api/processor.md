# TMD Processor

The TMD Processor module provides the central class for loading, parsing, and processing TMD files. It serves as the entry point for working with TMD data.

## Overview

The `TMDProcessor` class handles:

- Reading and parsing TMD files (v1 and v2 formats)
- Extracting metadata and height maps
- Computing statistics on the height data
- Providing access to the processed data for further operations

## Core Class

::: tmd.processor.TMDProcessor

## Workflow

1. **Initialize a processor** with a TMD file path
2. **Process the file** to extract data
3. **Access the height map and metadata** for further operations
4. **Compute statistics** or export metadata if needed

## Examples

### Basic Usage

```python
from tmd.processor import TMDProcessor

# Initialize the processor with a TMD file
processor = TMDProcessor("examples/v2/Dime.tmd")

# Process the file
data = processor.process()

# Access the height map
height_map = data['height_map']
# or
height_map = processor.get_height_map()

# Access metadata
metadata = processor.get_metadata()

# Get statistics
stats = processor.get_stats()
print(f"Min height: {stats['min']}")
print(f"Max height: {stats['max']}")
print(f"Mean height: {stats['mean']}")
```

### Debugging Mode

When processing problematic files, you can enable debug mode:

```python
from tmd.processor import TMDProcessor

processor = TMDProcessor("problematic_file.tmd")
processor.set_debug(True)
data = processor.process()

# Processor will print detailed information during parsing
```

### Exporting Metadata

```python
from tmd.processor import TMDProcessor

processor = TMDProcessor("sample.tmd")
processor.process()

# Export metadata to a text file
metadata_file = processor.export_metadata("sample_metadata.txt")
print(f"Metadata exported to {metadata_file}")
```

### Error Handling

```python
from tmd.processor import TMDProcessor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

try:
    processor = TMDProcessor("sample.tmd")
    data = processor.process()
    
    if data is None:
        print("Processing failed, check logs for details")
    else:
        print("Processing successful")
        height_map = data['height_map']
        # Continue with analysis...
        
except FileNotFoundError:
    print("TMD file not found")
except Exception as e:
    print(f"Unexpected error: {str(e)}")
```

## Complete Processing Pipeline

```python
from tmd.processor import TMDProcessor
from tmd.utils.filter import apply_gaussian_filter, calculate_rms_roughness
from tmd.utils.processing import threshold_height_map
from tmd.exporters.image import generate_all_maps
from tmd.exporters.model import convert_heightmap_to_stl
import os

def process_tmd_file(file_path, output_dir="."):
    """Complete processing pipeline for a TMD file."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process TMD file
    processor = TMDProcessor(file_path)
    data = processor.process()
    
    if data is None:
        print(f"Failed to process {file_path}")
        return None
    
    # Get height map
    height_map = data['height_map']
    
    # Export metadata
    metadata_file = processor.export_metadata(
        os.path.join(output_dir, "metadata.txt")
    )
    
    # Filter and threshold
    smoothed_map = apply_gaussian_filter(height_map, sigma=1.0)
    cleaned_map = threshold_height_map(
        smoothed_map,
        min_height=smoothed_map.min() * 1.05,
        max_height=smoothed_map.max() * 0.95
    )
    
    # Calculate roughness
    roughness = calculate_rms_roughness(height_map)
    print(f"RMS Roughness: {roughness:.4f}")
    
    # Generate material maps
    material_dir = os.path.join(output_dir, "materials")
    maps = generate_all_maps(cleaned_map, output_dir=material_dir)
    
    # Generate 3D model
    model_file = os.path.join(output_dir, "model.stl")
    convert_heightmap_to_stl(
        cleaned_map,
        filename=model_file,
        x_length=data.get('x_length', 10.0),
        y_length=data.get('y_length', 10.0),
        z_scale=5.0
    )
    
    print(f"Processing complete. Results saved to {output_dir}")
    return data

# Example usage
process_tmd_file("sample.tmd", "output/sample_results")
```

## File Format Support

The TMDProcessor supports both TMD v1 and v2 file formats:

- **v1 format**: Earlier version with simpler header structure
- **v2 format**: Current version used by TrueMap v6 and GelSight

The processor automatically detects the format version from the file header.
