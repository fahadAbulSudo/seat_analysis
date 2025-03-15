import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def moravec_corner_detection(image, window_size=3, threshold=100):
    rows, cols = image.shape
    offset = window_size // 2

    # Convert to int32 to prevent overflow during subtraction
    image = np.array(image, dtype=np.int32)

    corner_response = np.zeros(image.shape, dtype=np.float32)

    for y in range(offset, rows - offset):
        for x in range(offset, cols - offset):
            min_ssd = float('inf')

            for shift_x, shift_y in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                ssd = 0.0

                for dy in range(-offset, offset + 1):
                    for dx in range(-offset, offset + 1):
                        if (0 <= y + dy < rows) and (0 <= x + dx < cols) and \
                           (0 <= y + dy + shift_y < rows) and (0 <= x + dx + shift_x < cols):
                            diff = image[y + dy, x + dx] - image[y + dy + shift_y, x + dx + shift_x]
                            ssd += diff ** 2

                min_ssd = min(min_ssd, ssd)

            corner_response[y, x] = min_ssd

    # Normalize response for better visualization
    corner_response = cv2.normalize(corner_response, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Thresholding
    corner_response[corner_response < threshold] = 0

    # Extract corner locations
    corners = np.argwhere(corner_response > 0)
    corners = [(x, y) for y, x in corners]

    return corners, corner_response

# Load the local image
input_image_path = "/home/fahadabul/mask_rcnn_skyhub/waviness_final/8Fbhf2iE.jpeg"
image = Image.open(input_image_path).convert("L")  # Convert to grayscale
image = np.array(image, dtype=np.int32)  # Convert to int32 to prevent overflow

# Apply Moravec Corner Detection
corners, corner_response = moravec_corner_detection(image, window_size=3, threshold=50)

# Convert grayscale to color for visualization
image_with_corners = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Draw detected corners
for corner in corners:
    cv2.circle(image_with_corners, corner, 3, (255, 0, 0), -1)

# Convert to RGB for matplotlib display
image_with_corners_rgb = cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB)

# Display results (Save instead of showing if in a non-interactive environment)
plt.figure(figsize=(15, 7))
plt.imshow(image_with_corners_rgb)
plt.title("Moravec Corner Detection")
plt.axis("off")

plt.savefig("moravec_output.png")  # Save output image
print("Processing complete. Image saved as 'moravec_output.png'.")
