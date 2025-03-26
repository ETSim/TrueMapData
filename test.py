import cv2
import numpy as np

# Load grayscale images
img1 = cv2.imread("circle_0mm_100g_heightmap_linear_detrend.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("circle_150mm_100g_heightmap_linear_detrend.png", cv2.IMREAD_GRAYSCALE)

# Resize to same shape if needed
if img1.shape != img2.shape:
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Preprocess: blur + histogram equalization
img1_blur = cv2.GaussianBlur(img1, (5, 5), 0)
img2_blur = cv2.GaussianBlur(img2, (5, 5), 0)
img1_eq = cv2.equalizeHist(img1_blur)
img2_eq = cv2.equalizeHist(img2_blur)

# Convert to float32 for ECC
img1_float = img1_eq.astype(np.float32) / 255.0
img2_float = img2_eq.astype(np.float32) / 255.0

# Initialize warp matrix (Affine = 2x3)
warp_matrix = np.eye(2, 3, dtype=np.float32)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-7)

# ECC alignment
try:
    cc, warp_matrix = cv2.findTransformECC(img1_float, img2_float, warp_matrix, cv2.MOTION_AFFINE, criteria)

    # Apply transformation
    aligned = cv2.warpAffine(img2, warp_matrix, (img1.shape[1], img1.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # Save the aligned result
    cv2.imwrite("aligned_ecc_image.png", aligned)
    print("✅ Aligned image saved as 'aligned_ecc_image.png'")

except cv2.error as e:
    print("❌ ECC alignment failed:", e)
