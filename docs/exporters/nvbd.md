# NVBD Exporter

The NVBD (NVIDIA Blast Destruction) exporter converts height maps to a format that can be used for realistic destruction simulations in NVIDIA Blast technology and compatible game engines.

## What is NVBD?

NVIDIA Blast is a destruction physics technology that allows for efficient, highly detailed destruction of 3D objects. The NVBD format used in this library is a simplified version designed to represent height map data in a chunked format suitable for destruction simulations.

Key features of the NVBD format:

- Divides the height map into smaller chunks for efficient destruction
- Stores height values in a structured format for physics simulations
- Can be used with NVIDIA Blast or similar destruction frameworks

## File Format

The TMD library implements an NVBD file format with the following structure:

- **Magic number**: `NVBD` (4 bytes)
- **Version**: 1 (int, 4 bytes)
- **Width**: Total width in samples (int, 4 bytes)
- **Height**: Total height in samples (int, 4 bytes)
- **Chunk size**: Size of each chunk (int, 4 bytes)
- **Chunks**: Series of chunk data:
  - **Chunk X, Y indices**: Position of chunk (2 ints, 8 bytes)
  - **Chunk width, height**: Dimensions of chunk (2 ints, 8 bytes)
  - **Chunk data**: Height values as 32-bit floats

## Usage

```python
from tmd.exporters.nvbd import export_heightmap_to_nvbd

# Export a height map to NVBD format
result = export_heightmap_to_nvbd(
    heightmap,        # 2D numpy array of height values
    'output.nvbd',    # Output file path
    scale=1.0,        # Optional scaling factor (default: 1.0)
    offset=0.0,       # Optional height offset (default: 0.0)
    chunk_size=16     # Size of destruction chunks (default: 16)
)

# Check if export was successful
if result:
    print("Export successful")
else:
    print("Export failed")
```

## Parameters

- **heightmap**: 2D numpy array containing height values
- **output_file**: Path where the NVBD file will be saved
- **scale**: Optional scaling factor for height values (default: 1.0)
- **offset**: Optional offset added to all height values (default: 0.0)
- **chunk_size**: Size of each destruction chunk (default: 16)

## Chunk Size

The `chunk_size` parameter is critical for destruction simulation performance:

- **Smaller chunks** (e.g., 8 or 16): More detailed destruction, but higher computational cost
- **Larger chunks** (e.g., 32 or 64): Less detailed destruction, but better performance

Choose a chunk size appropriate for your application's needs, balancing physics detail against performance requirements.

## Example Applications

### Game Engine Integration

The NVBD files can be loaded into game engines that support NVIDIA Blast or similar technologies:

1. Load the NVBD file into your physics engine
2. Generate mesh geometry for each chunk
3. Set up destruction simulation parameters
4. When an impact occurs, the relevant chunks break apart realistically

### Visual Effects

For visual effects applications:

1. Load the NVBD file to get chunk information
2. Pre-generate fracture patterns based on the chunks
3. Trigger destruction based on simulated forces or impacts
4. Apply physics simulation to the separated chunks

## Limitations

- The current implementation is a simplified version of the NVBD format
- For full compatibility with NVIDIA Blast, additional processing steps may be required
- Performance will depend on the complexity of the height map and the chosen chunk size
