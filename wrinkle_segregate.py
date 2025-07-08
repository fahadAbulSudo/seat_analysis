import os
import cv2
import json
import numpy as np
from pycocotools import mask as maskUtils
from collections import defaultdict
import shutil

# Paths
images_dir = "/home/fahadabul/mask_rcnn_skyhub/desired_dataset/images"
annotations_path = "/home/fahadabul/mask_rcnn_skyhub/desired_dataset/coco_annotations.json"
output_dir = "/home/fahadabul/mask_rcnn_skyhub/dataset_separated_1"
output_images_dir = os.path.join(output_dir, "images")
output_annotations_path = os.path.join(output_dir, "seat_wrinkle_annotations.json")

os.makedirs(output_images_dir, exist_ok=True)

# Load original COCO
with open(annotations_path, 'r') as f:
    coco_data = json.load(f)

# Find category IDs
seat_category_id = None
wrinkle_category_id = None
for cat in coco_data['categories']:
    if cat['name'].lower() == 'seat':
        seat_category_id = cat['id']
    elif cat['name'].lower() == 'wrinkle':
        wrinkle_category_id = cat['id']

assert seat_category_id is not None, "Seat category not found!"
assert wrinkle_category_id is not None, "Wrinkle category not found!"

# Group annotations by image
annotations_by_image = defaultdict(list)
for ann in coco_data['annotations']:
    annotations_by_image[ann['image_id']].append(ann)

# New COCO structure
new_images = []
new_annotations = []
new_categories = [
    {'id': 1, 'name': 'seat'},
    {'id': 2, 'name': 'wrinkle'}
]

new_image_id = 1
new_annotation_id = 1
count = 0
# Process each image
for img in coco_data['images']:
    img_id = img['id']
    file_name = img['file_name']  #.lstrip('/')
    img_path = os.path.join(images_dir, file_name)
    print(img_path, file_name, images_dir)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Warning: Cannot read {img_path}")
        continue
    count += 1
    print(count)
    anns = annotations_by_image[img_id]
    seat_anns = [a for a in anns if a['category_id'] == seat_category_id]
    wrinkle_anns = [a for a in anns if a['category_id'] == wrinkle_category_id]
    height, width = image.shape[:2]
    for idx, seat_ann in enumerate(seat_anns):
        # Decode seat mask
        if 'segmentation' not in seat_ann:
            continue
        
        # Handle RLE or polygon segmentation
        if isinstance(seat_ann['segmentation'], list):
            # polygon -> to mask
            rles = maskUtils.frPyObjects(seat_ann['segmentation'], height, width)
            rle = maskUtils.merge(rles)
        else:
            rle = seat_ann['segmentation']

        seat_mask = maskUtils.decode(rle)

        # Mask the seat
        masked_image = np.zeros_like(image)
        for c in range(3):  # For each channel
            masked_image[:,:,c] = image[:,:,c] * seat_mask

        # Save new image
        new_file_name = f"{os.path.splitext(file_name)[0]}_seat{idx}.jpg"
        new_image_path = os.path.join(output_images_dir, new_file_name)
        print("error")
        cv2.imwrite(new_image_path, masked_image)
        print("error1")
        # Add to new images list
        new_images.append({
            "id": new_image_id,
            "file_name": new_file_name,
            "height": img['height'],
            "width": img['width']
        })

        # Add the seat mask annotation
        new_seat_ann = {
            "id": new_annotation_id,
            "image_id": new_image_id,
            "category_id": 1,  # new seat category id
            "segmentation": seat_ann['segmentation'],
            "bbox": seat_ann['bbox'],
            "area": seat_ann['area'],
            "iscrowd": 0
        }
        new_annotations.append(new_seat_ann)
        new_annotation_id += 1

        # Now check for wrinkles inside this seat
        for wr_ann in wrinkle_anns:
            # Decode wrinkle mask
            if isinstance(wr_ann['segmentation'], list):
                wr_rles = maskUtils.frPyObjects(wr_ann['segmentation'], height, width)
                wr_rle = maskUtils.merge(wr_rles)
            else:
                wr_rle = wr_ann['segmentation']

            wrinkle_mask = maskUtils.decode(wr_rle)

            # Check overlap
            overlap = np.logical_and(seat_mask, wrinkle_mask)
            if np.sum(overlap) > 0:
                # Copy wrinkle annotation
                new_wrinkle_ann = {
                    "id": new_annotation_id,
                    "image_id": new_image_id,
                    "category_id": 2,  # new wrinkle category id
                    "segmentation": wr_ann['segmentation'],
                    "bbox": wr_ann['bbox'],
                    "area": wr_ann['area'],
                    "iscrowd": 0
                }
                new_annotations.append(new_wrinkle_ann)
                new_annotation_id += 1

        new_image_id += 1

# Save new COCO JSON
new_coco = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": new_categories
}

with open(output_annotations_path, 'w') as f:
    json.dump(new_coco, f, indent=4)

print("Done splitting images and saving new annotations!")
