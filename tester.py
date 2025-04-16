import os
import sys
import numpy as np
import cv2 as cv
import trimesh as tm

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

########################################
# CUDA Helper Functions (using cuda.bindings)
########################################

from cuda.bindings import driver, nvrtc

def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError("Unknown error type: {}".format(error))

def checkCudaErrors(result):
    if result[0].value != 0:
        raise RuntimeError("CUDA error code={} ({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]

########################################
# CPU Detail Magnitude Function (OpenCV-based)
########################################

def detail_magnitude_cpu(heightmap):
    blurred = cv.GaussianBlur(heightmap, (11, 11), 0) * 255
    gX = cv.Sobel(blurred, ddepth=cv.CV_64F, dx=1, dy=0, ksize=5)
    gY = cv.Sobel(blurred, ddepth=cv.CV_64F, dx=0, dy=1, ksize=5)
    gX = cv.convertScaleAbs(gX)
    gY = cv.convertScaleAbs(gY)
    return cv.addWeighted(gX, 0.5, gY, 0.5, 0)

########################################
# GPU Detail Magnitude Function (CUDA Sobel)
########################################

def detail_magnitude_cuda(heightmap):
    height, width = heightmap.shape
    input_data = np.ascontiguousarray(heightmap.astype(np.float32))
    output_data = np.zeros_like(input_data)

    sobel_kernel_code = r"""
    extern "C" __global__
    void sobel_filter(const float* input, float* output, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x > 0 && x < width-1 && y > 0 && y < height-1) {
            int idx = y * width + x;
            float gx = - input[(y-1)*width + (x-1)] - 2.0f * input[y*width + (x-1)] - input[(y+1)*width + (x-1)]
                       + input[(y-1)*width + (x+1)] + 2.0f * input[y*width + (x+1)] + input[(y+1)*width + (x+1)];
            float gy = - input[(y-1)*width + (x-1)] - 2.0f * input[(y-1)*width + x] - input[(y-1)*width + (x+1)]
                       + input[(y+1)*width + (x-1)] + 2.0f * input[(y+1)*width + x] + input[(y+1)*width + (x+1)];
            output[idx] = 0.5f * fabsf(gx) + 0.5f * fabsf(gy);
        } else if (x < width && y < height) {
            int idx = y * width + x;
            output[idx] = 0.0f;
        }
    }
    """

    checkCudaErrors(driver.cuInit(0))
    device = checkCudaErrors(driver.cuDeviceGet(0))
    context = checkCudaErrors(driver.cuCtxCreate(0, device))

    prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(sobel_kernel_code.encode(), b"sobel_filter.cu", 0, [], []))
    checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, 0, None))
    ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
    ptx = bytearray(ptxSize)
    checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))

    module = checkCudaErrors(driver.cuModuleLoadData(ptx))
    sobel_kernel = checkCudaErrors(driver.cuModuleGetFunction(module, b"sobel_filter"))

    d_input = checkCudaErrors(driver.cuMemAlloc(input_data.nbytes))
    d_output = checkCudaErrors(driver.cuMemAlloc(output_data.nbytes))

    checkCudaErrors(driver.cuMemcpyHtoD(d_input, input_data.ctypes.data, input_data.nbytes))

    block_dim = (16, 16, 1)
    grid_dim = ((width + block_dim[0] - 1) // block_dim[0],
                (height + block_dim[1] - 1) // block_dim[1],
                1)
    width_np = np.array([width], dtype=np.int32)
    height_np = np.array([height], dtype=np.int32)
    arg_list = [
         np.array([int(d_input)], dtype=np.uint64),
         np.array([int(d_output)], dtype=np.uint64),
         width_np,
         height_np
    ]
    args = np.array([arg.ctypes.data for arg in arg_list], dtype=np.uint64)

    checkCudaErrors(driver.cuLaunchKernel(
         sobel_kernel,
         grid_dim[0], grid_dim[1], grid_dim[2],
         block_dim[0], block_dim[1], block_dim[2],
         0, 0, args.ctypes.data, 0))
    checkCudaErrors(driver.cuCtxSynchronize())

    checkCudaErrors(driver.cuMemcpyDtoH(output_data.ctypes.data, d_output, output_data.nbytes))

    checkCudaErrors(driver.cuMemFree(d_input))
    checkCudaErrors(driver.cuMemFree(d_output))
    checkCudaErrors(driver.cuModuleUnload(module))
    checkCudaErrors(driver.cuCtxDestroy(context))

    return output_data

########################################
# BVH-Style Subdivision for Terrain Mesh Generation
########################################

class BVHNode:
    def __init__(self, x, y, width, height, depth):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.depth = depth

    def corners(self):
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x, self.y + self.height),
            (self.x + self.width, self.y + self.height),
        ]

def subdivideBVH(detail_map, max_depth, threshold):
    n = detail_map.shape[0]
    acc_sum = np.cumsum(detail_map, axis=0)
    acc_sum = np.cumsum(acc_sum, axis=1)
    acc_sum = np.swapaxes(acc_sum, 0, 1)
    plane = [BVHNode(0, 0, n, n, 0)]
    leaves = []
    print("Subdividing plane using BVH-style binary splits up to depth {}...".format(max_depth))
    for d in range(max_depth):
        new_plane = []
        for node in plane:
            x1, y1 = node.x, node.y
            x2, y2 = node.x + node.width - 1, node.y + node.height - 1
            region_sum = acc_sum[x2, y2] + acc_sum[x1, y1] - acc_sum[x1, y2] - acc_sum[x2, y1]
            if region_sum > threshold:
                if node.width >= node.height:
                    split = node.width // 2
                    left_child = BVHNode(node.x, node.y, split, node.height, node.depth + 1)
                    right_child = BVHNode(node.x + split, node.y, node.width - split, node.height, node.depth + 1)
                else:
                    split = node.height // 2
                    left_child = BVHNode(node.x, node.y, node.width, split, node.depth + 1)
                    right_child = BVHNode(node.x, node.y + split, node.width, node.height - split, node.depth + 1)
                new_plane.extend([left_child, right_child])
            else:
                leaves.append(node)
        if not new_plane:
            break
        plane = new_plane
    leaves.extend(plane)
    return leaves

def triangulate(vertices, poly):
    points = [vertices[i] for i in poly]
    center = (sum([p[0] for p in points]) // len(points),
              sum([p[1] for p in points]) // len(points))
    center_index = len(vertices)
    vertices.append(center)
    triangles = []
    for i in range(len(poly)):
        triangles.append([poly[i], center_index, poly[(i+1)%len(poly)]])
    return triangles

def createFaces(vertices, polys):
    triangles = []
    for poly in polys:
        triangles += triangulate(vertices, poly)
    return triangles

def getSourroundingVertices(vert_set, x, y, w, h):
    north, east, south, west = [], [], [], []
    for i in range(x, x + w):
        point = (i, y)
        if point in vert_set:
            north.append(vert_set[point])
    for i in range(y, y + h):
        point = (x + w, i)
        if point in vert_set:
            east.append(vert_set[point])
    for i in range(x + w, x, -1):
        point = (i, y + h)
        if point in vert_set:
            south.append(vert_set[point])
    for i in range(y + h, y, -1):
        point = (x, i)
        if point in vert_set:
            west.append(vert_set[point])
    return north, east, south, west

def createPolyFaces(vert_set, leaves):
    polys = []
    for leaf in leaves:
        n, e, s, w = getSourroundingVertices(vert_set, leaf.x, leaf.y, leaf.width, leaf.height)
        polys.append(n + e + s + w)
    return polys

########################################
# Create Terrain Mesh from Heightmap Using BVH Subdivision
########################################

def createTerrain(heightmap, max_depth, threshold, z_scale, ground_height):
    n = 2 ** max_depth
    aspect = heightmap.shape[1] / heightmap.shape[0]
    heightmap = cv.resize(heightmap, (n + 1, n + 1), interpolation=cv.INTER_CUBIC)
    print("Computing detail magnitude on CPU for mesh generation...")
    detail = detail_magnitude_cpu(heightmap)
    leaves = subdivideBVH(detail, max_depth, threshold * n * n)
    vert_set = {}
    for leaf in leaves:
        for corner in leaf.corners():
            if corner not in vert_set:
                vert_set[corner] = len(vert_set)
    print("Creating polygonal faces from {} vertices...".format(len(vert_set)))
    polys = createPolyFaces(vert_set, leaves)
    vertices = [None] * len(vert_set)
    for key in vert_set:
        vertices[vert_set[key]] = key
    print("Triangulating {} polygons...".format(len(polys)))
    triangles = createFaces(vertices, polys)
    print("Adjusting vertices with height values...")
    for i in range(len(vertices)):
        x, y = vertices[i]
        # Clamp indices so they do not exceed n (since heightmap shape is (n+1, n+1))
        xi = min(int(x), n)
        yi = min(int(y), n)
        vertices[i] = [y / n, aspect * x / n, heightmap[yi, xi] * z_scale]
    # Create sides of the terrain.
    north, east, south, west = getSourroundingVertices(vert_set, 0, 0, n, n)
    sides = [north + east[:1], east + south[:1], south + west[:1], west + north[:1]]
    edge = []
    for side in sides:
        for i in range(len(side)):
            vertex = vertices[side[i]].copy()
            vertex[2] = ground_height
            vertices.append(vertex)
            edge.append(len(vertices) - 1)
            if i > 0:
                triangles.append([side[i], len(vertices)-1, side[i-1]])
                triangles.append([len(vertices)-2, side[i-1], len(vertices)-1])
    vertex = [0.5, 0.5 * aspect, ground_height]
    vertices.append(vertex)
    for i in range(len(edge)):
        triangles.append([edge[i], edge[(i+1) % len(edge)], len(vertices) - 1])
    print("Terrain mesh created. Exporting STL...")
    return tm.Trimesh(vertices=np.array(vertices), faces=np.array(triangles))

########################################
# Main: Load image, compare detail magnitude (CPU vs GPU), and export STL
########################################

if __name__ == '__main__':
    image_file = "circle_0mm_100g_heightmap_linear_detrend_displacement.png"
    heightmap = cv.imread(image_file, cv.IMREAD_ANYDEPTH | cv.IMREAD_GRAYSCALE)
    if heightmap is None:
        print("Error: Could not load image:", image_file)
        sys.exit(1)
    heightmap = np.array(heightmap, dtype=np.float32)
    heightmap /= np.max(heightmap.flatten())

    # Compute CPU detail magnitude.
    cpu_start = time.time()
    detail_cpu = detail_magnitude_cpu(heightmap)
    cpu_end = time.time()
    cpu_time = cpu_end - cpu_start
    print(f"CPU detail magnitude took {cpu_time:.6f} seconds")

    # Compute GPU detail magnitude.
    gpu_start = time.time()
    detail_gpu = detail_magnitude_cuda(heightmap)
    gpu_end = time.time()
    gpu_time = gpu_end - gpu_start
    print(f"GPU detail magnitude took {gpu_time:.6f} seconds")

    diff = np.abs(detail_cpu.astype(np.float32) - detail_gpu.astype(np.float32))
    mean_diff = np.mean(diff)
    print(f"Mean absolute difference between CPU and GPU results: {mean_diff:.6f}")

    # Plot detail magnitude images and CPU vs GPU times.
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(heightmap, cmap='gray')
    plt.title("Heightmap")
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(detail_cpu, cmap='viridis')
    plt.title("Detail Magnitude (CPU)")
    plt.colorbar()
    plt.subplot(1, 3, 3)
    plt.imshow(detail_gpu, cmap='viridis')
    plt.title("Detail Magnitude (GPU)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 4))
    methods = ['CPU', 'GPU']
    times = [cpu_time, gpu_time]
    bars = plt.bar(methods, times, color=['blue', 'green'])
    plt.ylabel("Time (seconds)")
    plt.title("Detail Magnitude Computation: CPU vs GPU")
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, t + 0.0005, f"{t:.6f}s", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Create terrain mesh using BVH-style subdivision (using CPU detail magnitude for mesh generation).
    max_depth = 10
    threshold = 0.01
    z_scale = 1
    ground_height = 0
    terrain_mesh = createTerrain(heightmap, max_depth, threshold, z_scale, ground_height)

    output_file = "terrain_output.stl"
    terrain_mesh.export(output_file)
    print("Terrain STL exported to", output_file)
