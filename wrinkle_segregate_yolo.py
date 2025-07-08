import os
import cv2
import json
import numpy as np
from collections import defaultdict
from pycocotools import mask as maskUtils
from ultralytics import YOLO

# Paths
images_dir = "/home/fahadabul/mask_rcnn_skyhub/new_annotation/images"
annotations_path = "/home/fahadabul/mask_rcnn_skyhub/new_annotation/new_annotations_cleaned.json"
output_dir = "/home/fahadabul/mask_rcnn_skyhub/dataset_separated"
os.makedirs(os.path.join(output_dir, "images_A"), exist_ok=True)
output_ann_path = os.path.join(output_dir, "new_annotations_cleaned_result.json")

# Load original annotations
with open(annotations_path, "r") as f:
    coco_data = json.load(f)

# Load YOLO model (seat segmentation)
yolo_model = YOLO("/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/model_3rd/best.pt")

# Get wrinkle category ID
wrinkle_category_id = next((cat["id"] for cat in coco_data["categories"] if cat["name"].lower() == "wrinkle"), None)
assert wrinkle_category_id is not None, "Wrinkle category not found!"

# Group annotations by image
annotations_by_image = defaultdict(list)
for ann in coco_data["annotations"]:
    annotations_by_image[ann["image_id"]].append(ann)

# Output COCO structure
new_images = []
new_annotations = []
new_categories = [
    {"id": 1, "name": "seat"},
    {"id": 2, "name": "wrinkle"}
]
new_image_id = 1
new_annotation_id = 1

# Process each image
for img in coco_data["images"]:
    image_path = os.path.join(images_dir, img["file_name"])
    image = cv2.imread(image_path)
    if image is None:
        continue

    height, width = image.shape[:2]
    anns = annotations_by_image[img["id"]]
    wrinkle_anns = [a for a in anns if a["category_id"] == wrinkle_category_id]

    # Run YOLO segmentation to detect seats
    yolo_results = yolo_model(image)[0]
    if not hasattr(yolo_results, "masks") or yolo_results.masks is None:
        continue

    seat_masks = yolo_results.masks.data.cpu().numpy()

    for seat_idx, seat_mask in enumerate(seat_masks):
        binary_seat = (seat_mask > 0.3).astype(np.uint8)
        resized_mask = cv2.resize(binary_seat, (width, height), interpolation=cv2.INTER_NEAREST)

        # Bounding box of seat mask
        x, y, w, h = cv2.boundingRect(resized_mask)

        # Crop image and mask
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = resized_mask[y:y+h, x:x+w]

        # Apply mask to image
        masked_image = np.zeros_like(cropped_image)
        for c in range(3):
            masked_image[:, :, c] = cropped_image[:, :, c] * cropped_mask

        # Save cropped masked seat image
        new_file_name = f"{os.path.splitext(img['file_name'])[0]}_seat{seat_idx}.jpg"
        save_path = os.path.join(output_dir, "images_A", new_file_name)
        cv2.imwrite(save_path, masked_image)

        # Add new image entry
        new_images.append({
            "id": new_image_id,
            "file_name": new_file_name,
            "height": h,
            "width": w
        })

        # Encode cropped mask
        rle = maskUtils.encode(np.asfortranarray(cropped_mask))
        rle["counts"] = rle["counts"].decode("utf-8")
        area = int(np.sum(cropped_mask))

        new_annotations.append({
            "id": new_annotation_id,
            "image_id": new_image_id,
            "category_id": 1,
            "segmentation": rle,
            "bbox": [0, 0, w, h],
            "area": area,
            "iscrowd": 0
        })
        new_annotation_id += 1

        # Check wrinkle overlap using mask logic
        for wr_ann in wrinkle_anns:
            if isinstance(wr_ann["segmentation"], list):
                wr_rles = maskUtils.frPyObjects(wr_ann["segmentation"], height, width)
                wr_rle = maskUtils.merge(wr_rles)
            else:
                wr_rle = wr_ann["segmentation"]

            wrinkle_mask = maskUtils.decode(wr_rle).astype(np.uint8)
            overlap = np.logical_and(resized_mask, wrinkle_mask).astype(np.uint8)

            if not np.any(overlap):
                continue

            # Crop wrinkle mask to seat region
            cropped_wrinkle = overlap[y:y+h, x:x+w]
            if not np.any(cropped_wrinkle):
                continue

            # Convert to polygon
            contours, _ = cv2.findContours(cropped_wrinkle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            for contour in contours:
                if contour.shape[0] >= 3:
                    polygons.append(contour.flatten().astype(float).tolist())

            if not polygons:
                continue

            wr_area = int(np.sum(cropped_wrinkle))
            x0, y0, w0, h0 = cv2.boundingRect(cropped_wrinkle)

            print(f"Wrinkle in {new_file_name} => segmentation: {polygons}")

            new_annotations.append({
                "id": new_annotation_id,
                "image_id": new_image_id,
                "category_id": 2,
                "segmentation": polygons,
                "bbox": [x0, y0, w0, h0],
                "area": wr_area,
                "iscrowd": 0
            })
            new_annotation_id += 1

        new_image_id += 1

# Save final COCO JSON
with open(output_ann_path, "w") as f:
    json.dump({
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories
    }, f, indent=4)

print("✅ Preprocessing complete. YOLO-seat + wrinkle mapped dataset created.")
