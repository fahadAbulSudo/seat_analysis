import cv2
import numpy as np
from rembg import remove

input_path = "/home/fahadabul/mask_rcnn_skyhub/background/PXL_20250218_040258573.jpg"

def fore_back_extraction(input_path):
    # Read input image
    input_image = cv2.imread(input_path)

    if input_image is None:
        raise ValueError(f"Error: Unable to load image from {input_path}")

    # Convert image to RGBA (rembg requires images with alpha channel)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGBA)

    # Foreground extraction
    foreground_img = remove(input_image)

    # Save foreground image
    cv2.imwrite("fore.jpg", foreground_img)

    # Convert to grayscale
    gray = cv2.cvtColor(foreground_img, cv2.COLOR_RGBA2GRAY)

    kernel = np.ones((7, 7), np.uint8)

    # Apply thresholding
    foreground_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

    # Invert the mask to get background
    background_mask = cv2.bitwise_not(foreground_mask)
    background_mask = cv2.morphologyEx(background_mask, cv2.MORPH_OPEN, kernel)

    # Apply mask to original image
    color_mask = np.repeat(foreground_mask[:, :, np.newaxis], 3, axis=2)
    color_mask_b = np.repeat(background_mask[:, :, np.newaxis], 3, axis=2)
    foreground_mask = np.expand_dims(foreground_mask, axis=2)
    background_mask = np.expand_dims(background_mask, axis=2)

    # Create images using the mask
    background_image = np.where(foreground_mask == 0, input_image[:, :, :3], color_mask)
    foreground_image = np.where(background_mask == 0, input_image[:, :, :3], color_mask_b)

    return foreground_image, background_image

# Run the function
f, b = fore_back_extraction(input_path)
