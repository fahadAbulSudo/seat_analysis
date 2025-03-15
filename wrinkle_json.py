import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO

# Define paths
image_dir = "/home/fahadabul/mask_rcnn_skyhub/segmentation_torn_wrinkle_2.v2i.coco-segmentation/train"
annotation_file = "/home/fahadabul/mask_rcnn_skyhub/segmentation_torn_wrinkle_2.v2i.coco-segmentation/train/_annotations.coco.json"
filtered_annotation_file = "/home/fahadabul/mask_rcnn_skyhub/segmentation_torn_wrinkle_2.v2i.coco-segmentation/train/_filtered_annotations.json"
output_dir = "/home/fahadabul/mask_rcnn_skyhub/segmentation_torn_wrinkle_2.v2i.coco-segmentation/annotated"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load COCO annotations
with open(annotation_file, 'r') as f:
    coco_data = json.load(f)

# Keep only the 'wrinkle' category
wrinkle_category_id = None
filtered_categories = []
for category in coco_data["categories"]:
    if category["name"] == "wrinkle":
        wrinkle_category_id = category["id"]
        filtered_categories.append(category)
        break

if wrinkle_category_id is None:
    raise ValueError("Category 'wrinkle' not found in annotations.")

# Filter annotations to include only those belonging to 'wrinkle'
filtered_annotations = [ann for ann in coco_data["annotations"] if ann["category_id"] == wrinkle_category_id]

# Get image IDs that have 'wrinkle' annotations
valid_image_ids = {ann["image_id"] for ann in filtered_annotations}
filtered_images = [img for img in coco_data["images"] if img["id"] in valid_image_ids]

# Save the filtered annotations
filtered_coco_data = {
    "images": filtered_images,
    "annotations": filtered_annotations,
    "categories": filtered_categories
}

with open(filtered_annotation_file, 'w') as f:
    json.dump(filtered_coco_data, f, indent=4)

print(f"Filtered annotations saved to {filtered_annotation_file}")