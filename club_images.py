import os
import shutil
import json
from tqdm import tqdm

# Input folders
base_folder = "/home/fahadabul/mask_rcnn_skyhub/dataset_remain"
folders = ["1", "2", "3", "4", "5", "6", "7"]

# Output folders
output_images_folder = "/home/fahadabul/mask_rcnn_skyhub/combined_dataset/images"
output_annotation_file = "/home/fahadabul/mask_rcnn_skyhub/combined_dataset/annotations.json"

os.makedirs(output_images_folder, exist_ok=True)

# Prepare the COCO merged structure
merged_coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Track ID mappings
image_id_counter = 1
annotation_id_counter = 1
image_old_to_new = {}

# Assume categories are same across all datasets
categories_set = False

for folder in folders:
    print(f"Processing folder {folder}...")
    images_folder = os.path.join(base_folder, folder, "images")
    annotation_file = os.path.join(base_folder, folder, "result.json")

    # Load the annotation file
    with open(annotation_file, "r") as f:
        coco = json.load(f)

    if not categories_set:
        merged_coco["categories"] = coco["categories"]
        categories_set = True

    # Copy images and fix image ids
    for image_info in tqdm(coco["images"], desc=f"Copying images from folder {folder}"):
        old_image_id = image_info["id"]
        old_file_name = image_info["file_name"].lstrip('/')
        # file_name = img['file_name']
        src_image_path = os.path.join(images_folder, old_file_name)

        # Create a new unique file name if needed (avoid clashes)
        new_file_name = f"{folder}_{old_file_name}"
        dst_image_path = os.path.join(output_images_folder, new_file_name)

        shutil.copy(src_image_path, dst_image_path)

        # Update image info
        new_image_info = {
            "id": image_id_counter,
            "width": image_info["width"],
            "height": image_info["height"],
            "file_name": new_file_name
        }
        merged_coco["images"].append(new_image_info)

        # Map old image id to new image id
        image_old_to_new[(folder, old_image_id)] = image_id_counter
        image_id_counter += 1

    # Update annotations
    for anno in coco["annotations"]:
        new_anno = anno.copy()
        new_anno["id"] = annotation_id_counter
        new_anno["image_id"] = image_old_to_new[(folder, anno["image_id"])]
        merged_coco["annotations"].append(new_anno)
        annotation_id_counter += 1

# Save the merged annotation file
with open(output_annotation_file, "w") as f:
    json.dump(merged_coco, f)

print("Merging completed!")
print(f"All images copied to {output_images_folder}")
print(f"Combined annotation JSON saved at {output_annotation_file}")
