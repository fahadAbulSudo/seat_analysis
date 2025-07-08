import os
import json
import cv2
import numpy as np

# Paths
output_images_folder = "/home/fahadabul/mask_rcnn_skyhub/dataset_remain/3/images"
output_annotation_file = "/home/fahadabul/mask_rcnn_skyhub/dataset_remain/3/result.json"
output_visualization_folder = "/home/fahadabul/mask_rcnn_skyhub/dataset_remain/3/visualization"
out_of_bounds_report = "/home/fahadabul/mask_rcnn_skyhub/dataset_remain/3/out_of_bounds_annotations.txt"

# Create visualization output folder
os.makedirs(output_visualization_folder, exist_ok=True)

# Open the report file
report_f = open(out_of_bounds_report, "w")

# Load COCO annotations
with open(output_annotation_file, "r") as f:
    coco = json.load(f)

images_info = {img["id"]: img for img in coco["images"]}
annotations_info = coco["annotations"]

print(f"Loaded {len(images_info)} images and {len(annotations_info)} annotations.")

# Map image_id -> list of its annotations
img_id_to_annotations = {}
for ann in annotations_info:
    img_id_to_annotations.setdefault(ann["image_id"], []).append(ann)

x_offset = 0
y_offset = 0#-700

# Start validating
for img_id, img_info in images_info.items():
    file_name = img_info["file_name"].lstrip('/')
    image_path = os.path.join(output_images_folder, file_name)

    if not os.path.exists(image_path):
        print(f"Image file missing on disk: {file_name}")
        continue

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {file_name}")
        continue

    height, width = img.shape[:2]
    ann_list = img_id_to_annotations.get(img_id, [])

    # Draw annotations
    for ann in ann_list:
        if "segmentation" not in ann:
            continue
        segmentations = ann["segmentation"]

        for segmentation in segmentations:
            if len(segmentation) < 6:
                print(f"⚠️ Invalid segmentation in {file_name}, skipping...")
                continue
            # Convert flat list [x1, y1, x2, y2, ...] -> array of (x, y) points
            pts = np.array(segmentation, dtype=np.int32).reshape(-1, 2)
            # Correct the offset
            pts[:, 0] -= x_offset
            pts[:, 1] -= y_offset

            # Check if points are inside the image
            # if np.any(pts[:, 0] < 0) or np.any(pts[:, 0] >= width) or np.any(pts[:, 1] < 0) or np.any(pts[:, 1] >= height):
            #     warning_text = f"Out of bounds annotation in {file_name} (Annotation ID: {ann['id']})\n"
            #     print(f"⚠️ {warning_text.strip()}")
            #     report_f.write(warning_text)
            #     continue

            # Draw polygon
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Save the visualized image
    output_vis_path = os.path.join(output_visualization_folder, file_name)
    cv2.imwrite(output_vis_path, img)

# Close the report file
report_f.close()

print(f"\n✅ Validation complete! Visualized images saved to: {output_visualization_folder}")
print(f"📄 Out-of-bounds annotations report saved to: {out_of_bounds_report}")
