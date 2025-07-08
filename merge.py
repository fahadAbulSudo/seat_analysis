import os
import json
import shutil

# Paths
base_dir = "/home/fahadabul/mask_rcnn_skyhub/dataset_separated"
json1_path = os.path.join(base_dir, "seat_wrinkle_only_annotations.json")
json2_path = os.path.join(base_dir, "seat_wrinkle_only_annotations_result.json")
merged_json_path = os.path.join(base_dir, "merged_annotations.json")

images_dir = os.path.join(base_dir, "images")
images_dir_A = os.path.join(base_dir, "images_A")
os.makedirs(images_dir, exist_ok=True)

# Load both JSON files
with open(json1_path, "r") as f1, open(json2_path, "r") as f2:
    coco1 = json.load(f1)
    coco2 = json.load(f2)

# Starting indexes
max_img_id = max(img["id"] for img in coco1["images"]) + 1
max_ann_id = max(ann["id"] for ann in coco1["annotations"]) + 1

# Prepare merged data
merged_images = coco1["images"]
merged_annotations = coco1["annotations"]

# Update image and annotation IDs in second JSON
id_mapping = {}
for img in coco2["images"]:
    old_id = img["id"]
    img["id"] = max_img_id
    id_mapping[old_id] = max_img_id

    # Move image from images_A to images (if not already moved)
    src = os.path.join(images_dir_A, img["file_name"])
    dst = os.path.join(images_dir, img["file_name"])
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)

    merged_images.append(img)
    max_img_id += 1

for ann in coco2["annotations"]:
    ann["id"] = max_ann_id
    ann["image_id"] = id_mapping[ann["image_id"]]
    merged_annotations.append(ann)
    max_ann_id += 1

# Merge categories (ensure uniqueness by name)
category_map = {cat["name"]: cat["id"] for cat in coco1["categories"]}
new_categories = coco1["categories"]

for cat in coco2["categories"]:
    if cat["name"] not in category_map:
        cat["id"] = max(category_map.values()) + 1
        category_map[cat["name"]] = cat["id"]
        new_categories.append(cat)

# Build final merged COCO structure
merged_coco = {
    "images": merged_images,
    "annotations": merged_annotations,
    "categories": new_categories
}

# Save merged JSON
with open(merged_json_path, "w") as f:
    json.dump(merged_coco, f, indent=4)

print("✅ Combined annotations saved to:", merged_json_path)
