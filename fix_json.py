import json
import os
from PIL import Image, ImageDraw, ImageOps
from pycocotools import mask as maskUtils
import numpy as np

# === Paths ===
ann_path = "/home/fahadabul/mask_rcnn_skyhub/new_annotation/result.json"
img_dir = "/home/fahadabul/mask_rcnn_skyhub/new_annotation/images"
output_vis_dir = "/home/fahadabul/mask_rcnn_skyhub/new_annotation/visualized_bboxes_2025"

os.makedirs(output_vis_dir, exist_ok=True)

# === Load COCO Annotations ===
with open(ann_path) as f:
    coco = json.load(f)

# === Filter images starting with "2025" and fix dimensions ===
valid_image_ids = set()
for img in coco["images"]:
    if not img["file_name"].startswith("2025"):
        continue

    path = os.path.join(img_dir, img["file_name"])
    if not os.path.exists(path):
        print(f"⚠️ Skipping missing file: {img['file_name']}")
        continue

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)  # ✅ Correct orientation
        width, height = im.size

    img["width"] = width
    img["height"] = height
    valid_image_ids.add(img["id"])

# === Group annotations by image ===
annotations_by_image = {}
for ann in coco["annotations"]:
    if ann["image_id"] in valid_image_ids:
        annotations_by_image.setdefault(ann["image_id"], []).append(ann)

# === Fix annotations and draw bounding boxes ===
for image_id in valid_image_ids:
    img_info = next(img for img in coco["images"] if img["id"] == image_id)
    file_name = img_info["file_name"]
    width, height = img_info["width"], img_info["height"]

    img_path = os.path.join(img_dir, file_name)
    output_path = os.path.join(output_vis_dir, f"bbox_{file_name}")

    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image).convert("RGB")  # ✅ Handle rotation
    draw = ImageDraw.Draw(image)

    for ann in annotations_by_image.get(image_id, []):
        seg = ann["segmentation"][0]
        coords = np.array(seg).reshape(-1, 2)

        if np.any(coords[:, 0] > width) or np.any(coords[:, 1] > height):
            print(f"❌ Segmentation out of bounds for annotation {ann['id']} in {file_name}")

        # Fix bbox
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
        ann["bbox"] = bbox

        # Fix area
        rles = maskUtils.frPyObjects([seg], height, width)
        rle = maskUtils.merge(rles)
        ann["area"] = float(maskUtils.area(rle))

        # Draw updated bbox
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

    # Save visualization
    image.save(output_path)
    print(f"✅ Saved visualization: {output_path}")

# === Save updated annotation file ===
with open("fixed_annotations_2025_only.json", "w") as f:
    json.dump(coco, f)

print("✅ Annotation file updated and visualizations saved.")
