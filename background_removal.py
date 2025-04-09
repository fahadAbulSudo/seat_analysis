import cv2
import numpy as np
import os

# Input folder containing test images
input_folder = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/extracted_images"

# Output folders
segmented_output_folder = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/segmented_output_images"
superimposed_output_folder = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/segmented_output_images"

# Create output directories if they don't exist
os.makedirs(segmented_output_folder, exist_ok=True)
os.makedirs(superimposed_output_folder, exist_ok=True)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(input_folder, filename)

        # Load the image
        img = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply GaussianBlur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use Canny edge detection to detect boundaries
        edges = cv2.Canny(blurred, 50, 150)

        # Use GrabCut for segmentation
        mask = np.zeros(img.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        # Define a rectangle enclosing the foreground
        rect = (10, 10, img.shape[1] - 10, img.shape[0] - 10)

        # Apply GrabCut algorithm
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)

        # Convert the GrabCut mask to binary
        mask_final = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # Extract the foreground (segmented seats)
        segmented_seats = img * mask_final[:, :, np.newaxis]

        # Blend extracted seats back into the original image
        alpha = 0.7  # Adjust transparency
        blended_img = cv2.addWeighted(segmented_seats, alpha, img, 1 - alpha, 0)

        # Define save paths
        segmented_save_path = os.path.join(segmented_output_folder, f"segmented_{filename}")
        superimposed_save_path = os.path.join(superimposed_output_folder, f"superimposed_{filename}")

        # Save the images
        cv2.imwrite(segmented_save_path, segmented_seats)
        cv2.imwrite(superimposed_save_path, blended_img)

        print(f"Processed: {filename}")

print("✅ All images processed and saved!")
