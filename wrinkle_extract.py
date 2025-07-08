import os
import json
import shutil

# Paths
image_folder = '/home/fahadabul/mask_rcnn_skyhub/dataset_separated/images_A'
coco_json_path = '/home/fahadabul/mask_rcnn_skyhub/dataset_separated/new_annotations_cleaned_result.json'
output_coco_json_path = '/home/fahadabul/mask_rcnn_skyhub/dataset_separated/seat_wrinkle_only_annotations_result.json'

# Load COCO JSON
with open(coco_json_path, 'r') as f:
    coco_data = json.load(f)

# Step 1: Collect all image_ids that have wrinkle annotations
wrinkle_image_ids = set()
new_annotations = []

for ann in coco_data['annotations']:
    if ann['category_id'] == 2:  # Wrinkle
        wrinkle_image_ids.add(ann['image_id'])
        new_annotations.append(ann)  # Keep wrinkle annotation
    else:
        # Skipping seat or any other annotations
        continue

# Step 2: Filter images to only those with wrinkles
new_images = []
image_id_to_filename = {}
for img in coco_data['images']:
    if img['id'] in wrinkle_image_ids:
        new_images.append(img)
        image_id_to_filename[img['id']] = img['file_name']
    else:
        # Image does not have wrinkle -> remove the file
        img_path = os.path.join(image_folder, img['file_name'])
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Deleted image without wrinkles: {img['file_name']}")

# Step 3: Prepare new coco_data
new_coco_data = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": [cat for cat in coco_data['categories'] if cat['id'] == 2]  # Keep only wrinkle category
}

# Step 4: Save the updated COCO JSON
with open(output_coco_json_path, 'w') as f:
    json.dump(new_coco_data, f, indent=4)

print(f"\n✅ Updated JSON saved to: {output_coco_json_path}")
print(f"✅ Only images with wrinkles are kept inside: {image_folder}")
