import os
import json
import shutil

# === Paths ===
# First dataset (close)
images_dir_1 = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/aug_8/merged/images"
json1_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/aug_8/merged/merged_annotations.json"

# Second dataset (far)
images_dir_2 = "/home/fahadabul/mask_rcnn_skyhub/dataset_separated/images"
json2_path = "/home/fahadabul/mask_rcnn_skyhub/dataset_separated/merged_annotations.json"

# Output merged dataset
merged_images_dir = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/aug_8/merged/new/images"
merged_json_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/aug_8/merged/new/merged_annotations.json"
os.makedirs(merged_images_dir, exist_ok=True)

# === Load both JSON files ===
with open(json1_path, "r") as f1, open(json2_path, "r") as f2:
    coco1 = json.load(f1)
    coco2 = json.load(f2)

# === Starting indexes ===
max_img_id = max(img["id"] for img in coco1["images"]) + 1
max_ann_id = max(ann["id"] for ann in coco1["annotations"]) + 1

# === Prepare merged data ===
merged_images = coco1["images"].copy()
merged_annotations = coco1["annotations"].copy()

# Copy images from dataset 1 to merged folder
for img in coco1["images"]:
    src = os.path.join(images_dir_1, img["file_name"])
    dst = os.path.join(merged_images_dir, img["file_name"])
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)

# === Update image & annotation IDs for dataset 2 ===
id_mapping = {}
for img in coco2["images"]:
    old_id = img["id"]
    img["id"] = max_img_id
    id_mapping[old_id] = max_img_id

    # Copy image from dataset 2 to merged folder
    src = os.path.join(images_dir_2, img["file_name"])
    dst = os.path.join(merged_images_dir, img["file_name"])
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)

    merged_images.append(img)
    max_img_id += 1

for ann in coco2["annotations"]:
    ann["id"] = max_ann_id
    ann["image_id"] = id_mapping[ann["image_id"]]
    merged_annotations.append(ann)
    max_ann_id += 1

# === Merge categories (ensure no duplicates by name) ===
category_map = {cat["name"]: cat["id"] for cat in coco1["categories"]}
new_categories = coco1["categories"].copy()

for cat in coco2["categories"]:
    if cat["name"] not in category_map:
        cat["id"] = max(category_map.values()) + 1
        category_map[cat["name"]] = cat["id"]
        new_categories.append(cat)

# === Build final merged COCO structure ===
merged_coco = {
    "images": merged_images,
    "annotations": merged_annotations,
    "categories": new_categories
}

# === Save merged JSON ===
with open(merged_json_path, "w") as f:
    json.dump(merged_coco, f, indent=4)

print("✅ Combined annotations saved to:", merged_json_path)
print("✅ All merged images stored in:", merged_images_dir)
