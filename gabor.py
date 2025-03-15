import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Load grayscale image
image_path = "/home/fahadabul/mask_rcnn_skyhub/waviness_final/8Fbhf2iE.jpeg"
gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

### 1. Gabor Filter for Texture Analysis ###
def apply_gabor_filter(image, ksize=9, sigma=3, theta=0, lambd=10, gamma=0.5, psi=0):
    """Applies Gabor filter with specified parameters."""
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    return cv2.filter2D(image, cv2.CV_8UC3, kernel)

# Apply Gabor filters at multiple orientations
gabor_0 = apply_gabor_filter(gray, theta=0)
gabor_45 = apply_gabor_filter(gray, theta=np.pi/4)
gabor_90 = apply_gabor_filter(gray, theta=np.pi/2)
gabor_135 = apply_gabor_filter(gray, theta=3*np.pi/4)

# Combine Gabor responses
gabor_combined = cv2.addWeighted(gabor_0, 0.25, gabor_45, 0.25, 0)
gabor_combined = cv2.addWeighted(gabor_combined, 0.5, gabor_90, 0.25, 0)
gabor_combined = cv2.addWeighted(gabor_combined, 0.75, gabor_135, 0.25, 0)

### 2. Local Binary Pattern (LBP) ###
radius = 1
n_points = 8 * radius
lbp = local_binary_pattern(gray, n_points, radius, method="uniform")

### 3. Sobel Edge Detection ###
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))

### 4. Laplacian Edge Detection ###
laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
laplacian = np.uint8(255 * laplacian / np.max(laplacian))

### Save Images ###
output_dir = "/home/fahadabul/mask_rcnn_skyhub/waviness_final/"
cv2.imwrite(output_dir + "gabor_combined.jpg", gabor_combined)
cv2.imwrite(output_dir + "lbp.jpg", np.uint8(255 * lbp / np.max(lbp)))  # Normalize LBP
cv2.imwrite(output_dir + "sobel.jpg", sobel_combined)
cv2.imwrite(output_dir + "laplacian.jpg", laplacian)

print("Saved processed images in:", output_dir)

### Display Results ###
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(gabor_combined, cmap='gray')
plt.title('Gabor Filter (Texture Analysis)')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(lbp, cmap='gray')
plt.title('Local Binary Pattern (LBP)')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(sobel_combined, cmap='gray')
plt.title('Sobel Edge Detection')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.savefig(output_dir + "comparison.png")  # Save full figure
plt.close()  # Prevent interactive display issues

print("Saved comparison figure as 'comparison.png'.")
