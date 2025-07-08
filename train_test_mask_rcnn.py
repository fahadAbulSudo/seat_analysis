import os
import json
import random
import shutil

# Paths
dataset_root = "/home/fahadabul/mask_rcnn_skyhub/dataset_separated"
images_folder = os.path.join(dataset_root, "images")
annotation_file = os.path.join(dataset_root, "merged_annotations.json")

train_images_folder = os.path.join(dataset_root, "train", "images")
test_images_folder = os.path.join(dataset_root, "test", "images")
train_annotation_file = os.path.join(dataset_root, "train", "annotations.json")
test_annotation_file = os.path.join(dataset_root, "test", "annotations.json")

# Create directories
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(test_images_folder, exist_ok=True)

# Load COCO annotations
with open(annotation_file, "r") as f:
    coco = json.load(f)

images = coco["images"]
annotations = coco["annotations"]
categories = coco["categories"]

# Shuffle images
random.seed(42)  # for reproducibility
random.shuffle(images)

# 80-20 split
split_idx = int(0.8 * len(images))
train_images = images[:split_idx]
test_images = images[split_idx:]

# Create mapping from image_id to image_info
image_id_to_image = {img['id']: img for img in images}

# Collect annotations for train and test
train_image_ids = set(img['id'] for img in train_images)
test_image_ids = set(img['id'] for img in test_images)

train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
test_annotations = [ann for ann in annotations if ann['image_id'] in test_image_ids]

# Move/copy images
for img in train_images:
    src_path = os.path.join(images_folder, img['file_name'])
    dst_path = os.path.join(train_images_folder, img['file_name'])
    shutil.copy(src_path, dst_path)

for img in test_images:
    src_path = os.path.join(images_folder, img['file_name'])
    dst_path = os.path.join(test_images_folder, img['file_name'])
    shutil.copy(src_path, dst_path)

# Save new annotation files
train_coco = {
    "images": train_images,
    "annotations": train_annotations,
    "categories": categories
}
test_coco = {
    "images": test_images,
    "annotations": test_annotations,
    "categories": categories
}

with open(train_annotation_file, "w") as f:
    json.dump(train_coco, f)

with open(test_annotation_file, "w") as f:
    json.dump(test_coco, f)

print("✅ Dataset split into train and test successfully!")
