import numpy as np

# Load the NumPy file (assuming it contains a segmentation mask)
mask_data = np.load("/home/fahadabul/fcc-intro-to-llms/latest_image_mask_rcnn/brush_numpy/task-853-annotation-498-by-1-tag-torn-0.npy")

# Print shape and data type to inspect
print(f"Shape: {mask_data.shape}, Data Type: {mask_data.dtype}")

# Check unique values to confirm binary mask (0 for background, 1 for object)
print("Unique values in the mask:", np.unique(mask_data))
