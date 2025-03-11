"""
Functions for exporting TMD height maps to STL files.
"""

import numpy as np
import struct


def convert_heightmap_to_stl(
    height_map,
    filename="output.stl",
    x_offset=0,
    y_offset=0,
    x_length=1,
    y_length=1,
    z_scale=1,
    ascii=True,
):
    """
    Converts a height map into an STL file for 3D printing.

    Args:
        height_map: 2D numpy array of height values
        filename: Name of the output STL file
        x_offset: X-axis offset for the model
        y_offset: Y-axis offset for the model
        x_length: Physical length in the X direction
        y_length: Physical length in the Y direction
        z_scale: Scale factor for Z-axis values
        ascii: If True, creates ASCII STL; if False, creates binary STL

    Returns:
        None
    """
    rows, cols = height_map.shape
    if cols < 2 or rows < 2:
        print("Height map too small to generate STL.")
        return

    x_scale = x_length / (cols - 1)
    y_scale = y_length / (rows - 1)
    vertices = np.zeros((rows, cols, 3))

    # Generate vertices
    for i in range(rows):
        for j in range(cols):
            vertices[i, j] = [
                x_offset + j * x_scale,
                y_offset + i * y_scale,
                height_map[i, j] * z_scale,
            ]

    if ascii:
        _write_ascii_stl(vertices, filename)
    else:
        _write_binary_stl(vertices, filename)


def _write_ascii_stl(vertices, filename):
    """
    Writes an ASCII STL file using the given vertices.

    Args:
        vertices: 3D numpy array of vertex coordinates
        filename: Output STL filename
    """
    rows, cols, _ = vertices.shape
    triangles = []

    # Generate triangles (two per grid cell)
    for i in range(rows - 1):
        for j in range(cols - 1):
            v0 = vertices[i, j]
            v1 = vertices[i, j + 1]
            v2 = vertices[i + 1, j + 1]
            v3 = vertices[i + 1, j]
            triangles.append((v0, v1, v2))
            triangles.append((v0, v2, v3))

    # Write ASCII STL
    with open(filename, "w") as f:
        f.write("solid displacement\n")
        for tri in triangles:
            v0, v1, v2 = tri
            n = np.cross(v1 - v0, v2 - v0)
            norm_val = np.linalg.norm(n)
            if norm_val == 0:
                n = np.array([0, 0, 0])
            else:
                n = n / norm_val
            f.write(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n")
            f.write("    outer loop\n")
            for vertex in tri:
                f.write(
                    f"      vertex {vertex[0]:.6e} {vertex[1]:.6e} {vertex[2]:.6e}\n"
                )
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write("endsolid displacement\n")

    print(f"ASCII STL file saved to {filename}")


def _write_binary_stl(vertices, filename):
    """
    Writes a binary STL file using the given vertices.

    Args:
        vertices: 3D numpy array of vertex coordinates
        filename: Output STL filename
    """
    rows, cols, _ = vertices.shape

    # Count the number of triangles
    num_triangles = 2 * (rows - 1) * (cols - 1)

    # Open file for binary writing
    with open(filename, "wb") as f:
        # Write STL header (80 bytes)
        header = b"TMD Processor Generated Binary STL"
        header = header.ljust(80, b" ")
        f.write(header)

        # Write number of triangles (4 bytes)
        f.write(struct.pack("<I", num_triangles))

        # Write each triangle
        for i in range(rows - 1):
            for j in range(cols - 1):
                v0 = vertices[i, j]
                v1 = vertices[i, j + 1]
                v2 = vertices[i + 1, j + 1]
                v3 = vertices[i + 1, j]

                # Triangle 1
                n1 = np.cross(v1 - v0, v2 - v0)
                norm_val = np.linalg.norm(n1)
                if norm_val > 0:
                    n1 = n1 / norm_val
                f.write(struct.pack("<fff", *n1))  # normal
                f.write(struct.pack("<fff", *v0))  # vertex 1
                f.write(struct.pack("<fff", *v1))  # vertex 2
                f.write(struct.pack("<fff", *v2))  # vertex 3
                f.write(struct.pack("<H", 0))  # attribute byte count

                # Triangle 2
                n2 = np.cross(v2 - v0, v3 - v0)
                norm_val = np.linalg.norm(n2)
                if norm_val > 0:
                    n2 = n2 / norm_val
                f.write(struct.pack("<fff", *n2))  # normal
                f.write(struct.pack("<fff", *v0))  # vertex 1
                f.write(struct.pack("<fff", *v2))  # vertex 2
                f.write(struct.pack("<fff", *v3))  # vertex 3
                f.write(struct.pack("<H", 0))  # attribute byte count

    print(f"Binary STL file saved to {filename}")
