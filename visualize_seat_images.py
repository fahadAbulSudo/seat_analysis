import os
import cv2
import json
import numpy as np
import random
from pycocotools import mask as maskUtils

# === Paths ===
output_dir = "/home/fahadabul/mask_rcnn_skyhub/dataset_separated"
images_dir = os.path.join(output_dir, "images")
annotations_path = os.path.join(output_dir, "merged_annotations.json")
visualization_dir = os.path.join(output_dir, "visualizations")
os.makedirs(visualization_dir, exist_ok=True)

# === Load COCO Annotations ===
with open(annotations_path, "r") as f:
    coco_data = json.load(f)

# === Group annotations by image ID ===
annotations_by_image = {}
for ann in coco_data["annotations"]:
    image_id = ann["image_id"]
    annotations_by_image.setdefault(image_id, []).append(ann)

# === Visualization ===
for img in coco_data["images"]:
    img_id = img["id"]
    file_name = img["file_name"]
    width = img["width"]
    height = img["height"]

    img_path = os.path.join(images_dir, file_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️ Failed to read image: {img_path}")
        continue

    ann_list = annotations_by_image.get(img_id, [])

    # Visualize wrinkle masks (category_id == 2)
    for ann in ann_list:
        if ann["category_id"] != 1:
            continue  # Only show wrinkles

        # Decode mask
        if isinstance(ann["segmentation"], list):
            rles = maskUtils.frPyObjects(ann["segmentation"], height, width)
            rle = maskUtils.merge(rles)
        else:
            rle = ann["segmentation"]

        mask = maskUtils.decode(rle)
        # Resize mask to match image if needed
        # if mask.shape != image.shape[:2]:
        #     mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Overlay colored mask
        color = [random.randint(100, 255) for _ in range(3)]
        colored_mask = np.stack([mask * c for c in color], axis=-1).astype(np.uint8)
        image = cv2.addWeighted(image, 1.0, colored_mask, 0.5, 0)

        # Draw bounding box
        x, y, w, h = map(int, ann["bbox"])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # === Save visualization ===
    vis_path = os.path.join(visualization_dir, f"vis_{file_name}")
    cv2.imwrite(vis_path, image)
    print(f"✅ Saved: {vis_path}")
