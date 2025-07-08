import cv2
import numpy as np
import os

# Input and Output paths
input_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/MSN_12763_new/seat_11-06/PXL_20250611_092550578.jpg"
output_dir = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/MSN_12763_new/seat_11-06/output"
output_path = os.path.join(output_dir, "PXL_20250611_092550578_processed.jpg")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read in grayscale
img = cv2.imread(input_path, 0)

# Step 1: Dilate to suppress text/barcodes and emphasize background
dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))

# Step 2: Median blur to smooth background
bg_img = cv2.medianBlur(dilated_img, 21)

# Step 3: Subtract and invert
diff_img = 255 - cv2.absdiff(img, bg_img)

# Step 4: Normalize
norm_img = diff_img.copy()
cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# Step 5: Truncate and re-normalize
_, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

# Save the result
cv2.imwrite(output_path, thr_img)
print(f"Processed image saved to: {output_path}")
