import json
import os
import shutil

# Paths
torn_json_path = "/home/fahadabul/mask_rcnn_skyhub/final_images/merged_annotations_torn.json"
wrinkle_json_path = "/home/fahadabul/mask_rcnn_skyhub/final_images/merged_annotations_wrinkle.json"
torn_images_dir = "/home/fahadabul/mask_rcnn_skyhub/final_images/merged_images_torn"
wrinkle_images_dir = "/home/fahadabul/mask_rcnn_skyhub/final_images/merged_images_wrinkle"
output_json_path = "/home/fahadabul/mask_rcnn_skyhub/final_images/final_merged_annotations.json"
output_images_dir = "/home/fahadabul/mask_rcnn_skyhub/final_images/final_merged_images"

# Create output directory if not exists
os.makedirs(output_images_dir, exist_ok=True)

# Load JSON files
with open(torn_json_path, "r") as f:
    torn_data = json.load(f)

with open(wrinkle_json_path, "r") as f:
    wrinkle_data = json.load(f)

# Ensure "wrinkle" category (id=2) is in the category list of torn_data
categories = torn_data["categories"]
category_ids = {cat["id"] for cat in categories}
if 2 not in category_ids:
    categories.append({"id": 2, "name": "wrinkle", "supercategory": "none"})

# Initialize merged dataset
merged_data = {
    "images": [],
    "annotations": [],
    "categories": categories  # Keep updated categories
}

# Track new IDs
image_id_map = {}
new_image_id = 1
new_annotation_id = 1

def process_data(data, image_dir):
    global new_image_id, new_annotation_id
    for image in data["images"]:
        old_image_id = image["id"]
        new_filename = f"{new_image_id:06d}_" + os.path.basename(image["file_name"])

        # Copy image to merged folder
        old_image_path = os.path.join(image_dir, image["file_name"])
        new_image_path = os.path.join(output_images_dir, new_filename)
        if os.path.exists(old_image_path):
            shutil.copy(old_image_path, new_image_path)

        # Update image metadata
        image["id"] = new_image_id
        image["file_name"] = new_filename
        image_id_map[old_image_id] = new_image_id
        merged_data["images"].append(image)

        new_image_id += 1

    for annotation in data["annotations"]:
        annotation["id"] = new_annotation_id
        annotation["image_id"] = image_id_map[annotation["image_id"]]  # Update with new image ID
        merged_data["annotations"].append(annotation)
        new_annotation_id += 1

# Process torn and wrinkle datasets
process_data(torn_data, torn_images_dir)
process_data(wrinkle_data, wrinkle_images_dir)

# Save merged JSON
with open(output_json_path, "w") as f:
    json.dump(merged_data, f, indent=4)

print(f"Final merge complete! JSON saved to {output_json_path}")
