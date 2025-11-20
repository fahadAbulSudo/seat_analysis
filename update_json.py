import os
import json
from PIL import Image

# === CONFIG ===
json_path = "/home/swapnil/AIR_BUS/May/28th_data_preparation_for_mask_retraining/dataset/cropped_wrinkle_annotations.json"
images_dir = "/home/swapnil/AIR_BUS/May/28th_data_preparation_for_mask_retraining/dataset/cropped_images"
output_json_path = "/home/swapnil/AIR_BUS/May/28th_data_preparation_for_mask_retraining/dataset/fixed_coco_annotations.json"

# === LOAD JSON ===
with open(json_path, "r") as f:
    coco = json.load(f)

# === Validate and Fix Images ===
valid_images = []
filename_to_id = {}
fixed_image_ids = set()

for img in coco.get("images", []):
    img_path = os.path.join(images_dir, img["file_name"])
    try:
        with Image.open(img_path) as im:
            w, h = im.size
            if img["width"] != w or img["height"] != h:
                print(f"⚠️ Fixing image size for {img['file_name']}: ({img['width']}, {img['height']}) → ({w}, {h})")
                img["width"], img["height"] = w, h
            valid_images.append(img)
            filename_to_id[img["file_name"]] = img["id"]
            fixed_image_ids.add(img["id"])
    except Exception as e:
        print(f"❌ Skipping image {img['file_name']}: {e}")

# === Validate and Keep Matching Annotations ===
valid_annotations = []
for ann in coco.get("annotations", []):
    if ann["image_id"] not in fixed_image_ids:
        print(f"❌ Annotation image_id {ann['image_id']} not found. Skipping annotation {ann['id']}.")
        continue

    seg = ann.get("segmentation", [])
    if not isinstance(seg, list) or len(seg) == 0 or len(seg[0]) < 6:
        print(f"❌ Invalid segmentation in annotation {ann['id']}. Skipping.")
        continue

    if ann.get("area", 0) <= 0:
        print(f"❌ Annotation {ann['id']} has non-positive area. Skipping.")
        continue

    valid_annotations.append(ann)

# === Write Fixed JSON ===
fixed_coco = {
    "images": valid_images,
    "annotations": valid_annotations,
    "categories": coco.get("categories", [])
}

with open(output_json_path, "w") as f:
    json.dump(fixed_coco, f, indent=2)

print(f"✅ Fixed JSON saved to: {output_json_path}")
