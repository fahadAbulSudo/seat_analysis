import json
import os
import cv2
import numpy as np
from pycocotools import mask as mask_utils

# Function to decode RLE mask to binary mask
# def rle_to_mask(rle, height, width):
#     rle_dict = {"size": [height, width], "counts": rle}
#     binary_mask = mask_utils.decode(rle_dict)
#     return binary_mask

def rle_to_mask(rle, height, width):
    """
    Converts an RLE mask (list format) to a binary mask.
    Fixes the issue where 'counts' should be a properly formatted object.
    """
    if isinstance(rle, list):  
        # Convert uncompressed RLE list to compressed RLE format
        rle = mask_utils.frPyObjects({"size": [height, width], "counts": rle}, height, width)
        print(rle)
    else:
        rle = {"size": [height, width], "counts": rle}  # If already formatted, keep unchanged

    binary_mask = mask_utils.decode(rle)
    return binary_mask

# Function to overlay mask on an image
def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlay the binary mask on an image.
    """
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 1] = mask * 255  # Green mask
    blended = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return blended

# Function to annotate and save images
def annotate_images(json_file, image_folder, output_folder):
    with open(json_file, "r") as file:
        data = json.load(file)

    os.makedirs(output_folder, exist_ok=True)

    for item in data:
        image_path = item["image"]  # Extract image path
        image_filename = os.path.basename(image_path)  # Extract filename
        full_image_path = os.path.join(image_folder, image_filename)  # Construct path

        if not os.path.exists(full_image_path):
            print(f"Warning: Image {full_image_path} not found. Skipping...")
            continue
        
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"Error: Could not read {full_image_path}")
            continue

        for tag in item.get("tag", []):  # Extract segmentation data
            if tag.get("format") == "rle":
                rle_data = tag["rle"]
                height = tag["original_height"]
                width = tag["original_width"]

                # Convert RLE to binary mask
                binary_mask = rle_to_mask(rle_data, height, width)

                # Resize mask if necessary
                if binary_mask.shape[:2] != image.shape[:2]:
                    binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Overlay mask on image
                annotated_image = overlay_mask(image, binary_mask)

                # Save the annotated image
                save_path = os.path.join(output_folder, image_filename)
                cv2.imwrite(save_path, annotated_image)
                print(f"Annotated image saved: {save_path}")

# Usage
json_file = "/home/fahadabul/fcc-intro-to-llms/latest_image_mask_rcnn/modified_annotations.json"  # JSON with updated image paths
image_folder = "/home/fahadabul/fcc-intro-to-llms/latest_image_mask_rcnn/torn"  # Folder where images are stored
output_folder = "/home/fahadabul/fcc-intro-to-llms/latest_image_mask_rcnn/annotated_images"  # Folder to save annotated images

annotate_images(json_file, image_folder, output_folder)
