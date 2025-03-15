import os
import json
import cv2
import numpy as np
from pycocotools.coco import COCO

# Define paths
image_dir = "/home/fahadabul/mask_rcnn_skyhub/final_images/val"
annotation_file = "/home/fahadabul/mask_rcnn_skyhub/final_images/val_annotations.json"
output_dir = "/home/fahadabul/mask_rcnn_skyhub/final_images/annotated"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load COCO annotations
coco = COCO(annotation_file)

# Get all image IDs
image_ids = coco.getImgIds()

for image_id in image_ids:
    # Load image metadata
    img_info = coco.loadImgs(image_id)[0]
    img_path = os.path.join(image_dir, img_info['file_name'])
    
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
    
    # Read the image
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to read image: {img_path}")
        continue
    
    # Get annotations for the image
    ann_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(ann_ids)
    
    for ann in annotations:
        # Draw bounding box
        x, y, w, h = map(int, ann['bbox'])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw segmentation mask (if available)
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                poly = np.array(seg, np.int32).reshape((-1, 2))
                cv2.polylines(image, [poly], isClosed=True, color=(0, 0, 255), thickness=2)
    
    # Save the annotated image
    output_path = os.path.join(output_dir, img_info['file_name'])
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved: {output_path}")

print("Processing complete.")
