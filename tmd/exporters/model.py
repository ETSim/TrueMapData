import numpy as np
import struct
import meshio
import os

# ---------------------------
# Core Mesh Generation Function
# ---------------------------
def _create_mesh_from_heightmap(
    height_map, x_offset=0, y_offset=0, x_length=1, y_length=1, z_scale=1, base_height=0
):
    """
    Core function to create a 3D mesh from a height map.
    
    Args:
        height_map: 2D numpy array of height values.
        x_offset: X-axis offset.
        y_offset: Y-axis offset.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis.
        base_height: Height of solid base (0 = no base).
    
    Returns:
        tuple: (vertices, faces) or None if height map is too small
    """
    rows, cols = height_map.shape
    if cols < 2 or rows < 2:
        return None
    
    # Generate basic mesh
    vertices, faces = _generate_mesh(height_map, x_offset, y_offset, x_length, y_length, z_scale)
    
    # Add base if requested
    if base_height > 0:
        try:
            vertices, faces = _add_base_to_mesh(vertices, faces, base_height)
        except Exception as e:
            print(f"Error adding base to mesh: {e}. Proceeding without base.")
    
    return vertices, faces

# ---------------------------
# Helper: Mesh Generation from Height Map
# ---------------------------
def _generate_mesh(height_map, x_offset, y_offset, x_length, y_length, z_scale):
    """
    Generates vertices and faces (triangles) from a height map.

    Args:
        height_map: 2D numpy array of height values.
        x_offset: X-axis offset.
        y_offset: Y-axis offset.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis.

    Returns:
        vertices: List of vertices as [x, y, z].
        faces: List of faces (triangles) as [i, j, k] indices into the vertices list.
    """
    rows, cols = height_map.shape
    if rows < 2 or cols < 2:
        raise ValueError("Height map too small to generate mesh.")
    
    x_scale = x_length / max(1, cols - 1)
    y_scale = y_length / max(1, rows - 1)
    
    # Generate vertices in row-major order
    vertices = []
    for i in range(rows):
        for j in range(cols):
            vertices.append([
                x_offset + j * x_scale,
                y_offset + i * y_scale,
                height_map[i, j] * z_scale
            ])
    
    # Generate faces (triangles). Two triangles per grid cell.
    faces = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = i * cols + j
            v1 = i * cols + (j + 1)
            v2 = (i + 1) * cols + (j + 1)
            v3 = (i + 1) * cols + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    
    return vertices, faces

# ---------------------------
# Helper: Add Base to Mesh
# ---------------------------
def _add_base_to_mesh(vertices, faces, base_height=0.0):
    """
    Adds a solid base to a mesh for 3D printing stability.
    
    Args:
        vertices: List of vertices as [x, y, z]
        faces: List of triangular faces as [v1, v2, v3] indices
        base_height: Height of the base below the lowest point of the mesh
        
    Returns:
        Tuple of (new_vertices, new_faces)
    """
    if base_height <= 0:
        return vertices, faces
        
    # Find mesh dimensions
    vertices_array = np.array(vertices)
    x_coords = vertices_array[:, 0]
    y_coords = vertices_array[:, 1]
    z_coords = vertices_array[:, 2]
    
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min = np.min(z_coords)
    
    # Create base coordinates (below the mesh)
    base_z = z_min - base_height
    
    # Copy all existing vertices and faces
    new_vertices = vertices.copy()
    new_faces = faces.copy()
    
    # Get the current vertex count for indexing
    vertex_count = len(vertices)
    
    # Add base vertices (bottom face corners)
    base_indices = []
    for corner in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]:
        new_vertices.append([corner[0], corner[1], base_z])
        base_indices.append(vertex_count)
        vertex_count += 1
    
    # Find boundary edges to create side walls
    # Create a set to store edges (vertex pairs) that appear only once
    edge_count = {}
    for face in faces:
        edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
        for edge in edges:
            # Sort the vertex indices to ensure we count the edge properly
            sorted_edge = tuple(sorted(edge))
            edge_count[sorted_edge] = edge_count.get(sorted_edge, 0) + 1
    
    # Edges that appear only once are on the boundary
    boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
    
    # Create dictionary to map original vertex coordinates to base vertices
    bottom_corners = {
        (x_min, y_min): base_indices[0],
        (x_max, y_min): base_indices[1],
        (x_max, y_max): base_indices[2],
        (x_min, y_max): base_indices[3]
    }
    
    # Create side triangles
    for edge in boundary_edges:
        v1, v2 = edge
        v1_x, v1_y = vertices[v1][0], vertices[v1][1]
        v2_x, v2_y = vertices[v2][0], vertices[v2][1]
        
        # Find closest bottom corners for these edge vertices
        bottom_v1 = min(bottom_corners.items(), key=lambda x: 
                      ((x[0][0]-v1_x)**2 + (x[0][1]-v1_y)**2))[1]
        bottom_v2 = min(bottom_corners.items(), key=lambda x: 
                      ((x[0][0]-v2_x)**2 + (x[0][1]-v2_y)**2))[1]
        
        # Add two triangles to create a quad for this side face
        new_faces.append([v1, v2, bottom_v2])
        new_faces.append([v1, bottom_v2, bottom_v1])
    
    # Add bottom face (triangulated)
    new_faces.append([base_indices[0], base_indices[1], base_indices[2]])
    new_faces.append([base_indices[0], base_indices[2], base_indices[3]])
    
    return new_vertices, new_faces

# ---------------------------
# STL Export Function (Unified)
# ---------------------------
def convert_heightmap_to_stl(
    height_map,
    filename="output.stl",
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    ascii=True,
    base_height=0.0
):
    """
    Converts a height map into an STL file for 3D printing.

    Args:
        height_map: 2D numpy array of height values.
        filename: Name of the output STL file.
        x_offset: X-axis offset for the model.
        y_offset: Y-axis offset for the model.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis values.
        ascii: If True, creates ASCII STL; if False, creates binary STL.
        base_height: Height of solid base to add below the model (0 = no base).

    Returns:
        str: Path to the created file or None if failed.
    """
    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    except (PermissionError, OSError) as e:
        print(f"Error creating directory for {filename}: {e}")
        return None
    
    # Generate the mesh
    mesh_result = _create_mesh_from_heightmap(
        height_map, x_offset, y_offset, x_length, y_length, z_scale, base_height
    )
    
    if not mesh_result:
        print("Height map too small to generate STL.")
        return None
    
    vertices, faces = mesh_result
    vertices_array = np.array(vertices)
    
    # Write the STL file (ASCII or binary)
    try:
        if ascii:
            # Write ASCII STL
            with open(filename, "w") as f:
                f.write("solid displacement\n")
                
                for face in faces:
                    v0 = vertices_array[face[0]]
                    v1 = vertices_array[face[1]]
                    v2 = vertices_array[face[2]]
                    
                    # Calculate normal
                    n = np.cross(v1 - v0, v2 - v0)
                    norm_val = np.linalg.norm(n)
                    if norm_val < 1e-10:
                        n = np.array([0, 0, 1.0])
                    else:
                        n = n / norm_val
                    
                    # Write facet
                    f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
                    f.write("    outer loop\n")
                    f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                    f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                    f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                    f.write("    endloop\n")
                    f.write("  endfacet\n")
                
                f.write("endsolid displacement\n")
                
            print(f"ASCII STL file{' with base' if base_height > 0 else ''} saved to {filename}")
        else:
            # Write binary STL
            with open(filename, "wb") as f:
                # Write header (80 bytes)
                header = b"TMD Processor Generated Binary STL"
                header = header.ljust(80, b" ")
                f.write(header)
                
                # Write number of triangles (4 bytes)
                f.write(struct.pack("<I", len(faces)))
                
                # Write each triangle
                for face in faces:
                    v0 = vertices_array[face[0]]
                    v1 = vertices_array[face[1]]
                    v2 = vertices_array[face[2]]
                    
                    # Calculate normal
                    n = np.cross(v1 - v0, v2 - v0)
                    norm_val = np.linalg.norm(n)
                    if norm_val > 0:
                        n = n / norm_val
                    
                    # Write triangle data
                    f.write(struct.pack("<fff", *n))      # normal
                    f.write(struct.pack("<fff", *v0))     # vertex 1
                    f.write(struct.pack("<fff", *v1))     # vertex 2
                    f.write(struct.pack("<fff", *v2))     # vertex 3
                    f.write(struct.pack("<H", 0))         # attribute byte count
                
            print(f"Binary STL file{' with base' if base_height > 0 else ''} saved to {filename}")
        
        return filename
    except Exception as e:
        print(f"Error writing STL file: {e}")
        return None

# ---------------------------
# OBJ Export Function
# ---------------------------
def convert_heightmap_to_obj(
    height_map,
    filename="output.obj",
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    base_height=0.0
):
    """
    Converts a height map into an OBJ file.

    Args:
        height_map: 2D numpy array of height values.
        filename: Name of the output OBJ file.
        x_offset: X-axis offset for the model.
        y_offset: Y-axis offset for the model.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis values.
        base_height: Height of solid base to add below the model (0 = no base).

    Returns:
        str: Path to the created file or None if failed.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Generate the mesh
    mesh_result = _create_mesh_from_heightmap(
        height_map, x_offset, y_offset, x_length, y_length, z_scale, base_height
    )
    
    if not mesh_result:
        print("Height map too small to generate OBJ.")
        return None
    
    vertices, faces = mesh_result
    
    # Write the OBJ file
    try:
        with open(filename, "w") as f:
            # Write vertices (OBJ indices start at 1)
            for v in vertices:
                f.write(f"v {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            # Write faces (OBJ format uses 1-indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"OBJ file{' with base' if base_height > 0 else ''} saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error writing OBJ file: {e}")
        return None

# ---------------------------
# PLY Export Function
# ---------------------------
def convert_heightmap_to_ply(
    height_map,
    filename="output.ply",
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    base_height=0.0
):
    """
    Converts a height map into an ASCII PLY file.

    Args:
        height_map: 2D numpy array of height values.
        filename: Name of the output PLY file.
        x_offset: X-axis offset for the model.
        y_offset: Y-axis offset for the model.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis values.
        base_height: Height of solid base to add below the model (0 = no base).

    Returns:
        str: Path to the created file or None if failed.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    
    # Generate the mesh
    mesh_result = _create_mesh_from_heightmap(
        height_map, x_offset, y_offset, x_length, y_length, z_scale, base_height
    )
    
    if not mesh_result:
        print("Height map too small to generate PLY.")
        return None
    
    vertices, faces = mesh_result
    
    # Write the PLY file
    try:
        with open(filename, "w") as f:
            # Write the PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            # Write vertices
            for v in vertices:
                f.write(f"{v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            
            # Write faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        print(f"PLY file{' with base' if base_height > 0 else ''} saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error writing PLY file: {e}")
        return None

# ---------------------------
# Meshio-based Export Functions
# ---------------------------
def _export_with_meshio(
    height_map,
    filename,
    file_format,
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    base_height=0.0,
    **kwargs
):
    """
    Common function for meshio-based exports.
    
    Args:
        height_map: 2D numpy array of height values.
        filename: Output filename.
        file_format: Format to export ("stl", "obj", "ply", etc.)
        x_offset, y_offset: Offset values.
        x_length, y_length: Physical dimensions.
        z_scale: Z-axis scaling.
        base_height: Height of solid base.
        **kwargs: Additional format-specific arguments.
        
    Returns:
        str: Path to the created file or None if failed.
    """
    # Ensure directory exists
    try:
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
    except (PermissionError, OSError) as e:
        print(f"Error creating directory for {filename}: {e}")
        return None
    
    # Generate the mesh
    mesh_result = _create_mesh_from_heightmap(
        height_map, x_offset, y_offset, x_length, y_length, z_scale, base_height
    )
    
    if not mesh_result:
        print(f"Height map too small to generate {file_format.upper()}.")
        return None
    
    vertices, faces = mesh_result
    
    try:
        # Convert to meshio format
        points = np.array(vertices)
        cells = [("triangle", np.array(faces, dtype=np.int32))]
        mesh = meshio.Mesh(points=points, cells=cells)
        
        # Write the file
        meshio.write(filename, mesh, file_format=file_format, **kwargs)
        
        # Special handling for OBJ format (fix for meshio format inconsistencies)
        if file_format == "obj":
            with open(filename, 'r') as f:
                content = f.read()
            
            # If the file doesn't have the expected format, rewrite it
            if not content.startswith("v "):
                with open(filename, 'w') as f:
                    # Add vertices first
                    for v in points:
                        f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
                    # Then add faces (OBJ uses 1-indexed vertices)
                    for face in faces:
                        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        print(f"Meshio {file_format.upper()} file{' with base' if base_height > 0 else ''} saved to {filename}")
        return filename
    except Exception as e:
        print(f"Error creating {file_format.upper()} file with meshio: {e}")
        return None

def convert_heightmap_to_stl_meshio(
    height_map,
    filename="meshio_output.stl",
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    ascii=True,
    base_height=0.0
):
    """
    Converts a height map into an STL file using Meshio.

    Args:
        height_map: 2D numpy array of height values.
        filename: Output STL filename.
        x_offset: X-axis offset.
        y_offset: Y-axis offset.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis.
        ascii: If True, write ASCII STL; otherwise, binary STL.
        base_height: Height of solid base to add below the model (0 = no base).
    
    Returns:
        str: Path to the created file or None if failed.
    """
    return _export_with_meshio(
        height_map, filename, "stl",
        x_offset, y_offset, x_length, y_length, z_scale, base_height,
        binary=not ascii
    )

def convert_heightmap_to_obj_meshio(
    height_map,
    filename="meshio_output.obj",
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    base_height=0.0
):
    """
    Converts a height map into an OBJ file using Meshio.

    Args:
        height_map: 2D numpy array of height values.
        filename: Output OBJ filename.
        x_offset: X-axis offset.
        y_offset: Y-axis offset.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis.
        base_height: Height of solid base to add below the model (0 = no base).
    
    Returns:
        str: Path to the created file or None if failed.
    """
    return _export_with_meshio(
        height_map, filename, "obj",
        x_offset, y_offset, x_length, y_length, z_scale, base_height
    )

def convert_heightmap_to_ply_meshio(
    height_map,
    filename="meshio_output.ply",
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    base_height=0.0
):
    """
    Converts a height map into a PLY file using Meshio.

    Args:
        height_map: 2D numpy array of height values.
        filename: Output PLY filename.
        x_offset: X-axis offset.
        y_offset: Y-axis offset.
        x_length: Physical length in the X direction.
        y_length: Physical length in the Y direction.
        z_scale: Scale factor for Z-axis.
        base_height: Height of solid base to add below the model (0 = no base).
    
    Returns:
        str: Path to the created file or None if failed.
    """
    return _export_with_meshio(
        height_map, filename, "ply",
        x_offset, y_offset, x_length, y_length, z_scale, base_height,
        binary=False
    )
