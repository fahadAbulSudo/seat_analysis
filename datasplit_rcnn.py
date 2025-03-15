import json
import os
import shutil
import random

# Paths
json_path = "/home/fahadabul/mask_rcnn_skyhub/final_images/final_merged_annotations.json"
images_dir = "/home/fahadabul/mask_rcnn_skyhub/final_images/final_merged_images"
output_dir = "/home/fahadabul/mask_rcnn_skyhub/final_images"

train_json_path = os.path.join(output_dir, "train_annotations.json")
val_json_path = os.path.join(output_dir, "val_annotations.json")
train_images_dir = os.path.join(output_dir, "train")
val_images_dir = os.path.join(output_dir, "val")

# Create directories if they don't exist
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Shuffle images randomly
random.seed(42)  # For reproducibility
random.shuffle(data["images"])

# Define split ratio (e.g., 80% train, 20% val)
split_ratio = 0.8
split_index = int(len(data["images"]) * split_ratio)

train_images = data["images"][:split_index]
val_images = data["images"][split_index:]

# Create image_id lookup sets
train_image_ids = {img["id"] for img in train_images}
val_image_ids = {img["id"] for img in val_images}

# Split annotations
train_annotations = [ann for ann in data["annotations"] if ann["image_id"] in train_image_ids]
val_annotations = [ann for ann in data["annotations"] if ann["image_id"] in val_image_ids]

# Move images to respective folders
def move_images(images, target_dir):
    for img in images:
        src_path = os.path.join(images_dir, img["file_name"])
        dst_path = os.path.join(target_dir, img["file_name"])
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)

move_images(train_images, train_images_dir)
move_images(val_images, val_images_dir)

# Create train and val JSON files
train_data = {"images": train_images, "annotations": train_annotations, "categories": data["categories"]}
val_data = {"images": val_images, "annotations": val_annotations, "categories": data["categories"]}

with open(train_json_path, "w") as f:
    json.dump(train_data, f, indent=4)

with open(val_json_path, "w") as f:
    json.dump(val_data, f, indent=4)

print(f"Dataset split complete!\nTrain: {len(train_images)} images\nVal: {len(val_images)} images")
print(f"Train JSON saved to {train_json_path}")
print(f"Val JSON saved to {val_json_path}")
