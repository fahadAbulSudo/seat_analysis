import os
import cv2
import json
import numpy as np
from collections import defaultdict
from pycocotools import mask as maskUtils
from ultralytics import YOLO

# Paths
images_dir = "/home/fahadabul/mask_rcnn_skyhub/desired_dataset/images"
annotations_path = "/home/fahadabul/mask_rcnn_skyhub/desired_dataset/coco_annotations.json"
output_dir = "/home/fahadabul/mask_rcnn_skyhub/dataset_separated"
os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
output_ann_path = os.path.join(output_dir, "new_annotations.json")

# Load original annotations
with open(annotations_path, "r") as f:
    coco_data = json.load(f)

# Load YOLO model (seat segmentation)
yolo_model = YOLO("/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/model_3rd/best.pt")

# Get wrinkle category ID
wrinkle_category_id = None
for cat in coco_data["categories"]:
    if cat["name"].lower() == "wrinkle":
        wrinkle_category_id = cat["id"]
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
        # if w == 0 or h == 0:
        #     continue  # Skip empty/invalid masks

        # Crop image and mask
        cropped_image = image[y:y+h, x:x+w]
        cropped_mask = resized_mask[y:y+h, x:x+w]

        # Apply mask to image
        masked_image = np.zeros_like(cropped_image)
        for c in range(3):
            masked_image[:, :, c] = cropped_image[:, :, c] * cropped_mask

        # Save cropped masked seat image
        new_file_name = f"{os.path.splitext(img['file_name'])[0]}_seat{seat_idx}.jpg"
        save_path = os.path.join(output_dir, "images", new_file_name)
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
            "category_id": 1,  # seat
            "segmentation": rle,
            "bbox": [0, 0, w, h],  # since it’s cropped, bbox always starts at (0,0)
            "area": area,
            "iscrowd": 0
        })
        new_annotation_id += 1

        # Cropped seat bounding box in original image coordinates
        seat_x1, seat_y1, seat_x2, seat_y2 = x, y, x + w, y + h

        # Add wrinkle annotations that overlap the seat bounding box
        for wr_ann in wrinkle_anns:
            if isinstance(wr_ann["segmentation"], list):
                wr_rles = maskUtils.frPyObjects(wr_ann["segmentation"], height, width)
                wr_rle = maskUtils.merge(wr_rles)
            else:
                wr_rle = wr_ann["segmentation"]

            wrinkle_mask = maskUtils.decode(wr_rle)

            # Get wrinkle bbox
            wr_x, wr_y, wr_w, wr_h = cv2.boundingRect(wrinkle_mask)
            wr_x1, wr_y1, wr_x2, wr_y2 = wr_x, wr_y, wr_x + wr_w, wr_y + wr_h

            # Check for intersection with seat bbox
            inter_x1 = max(seat_x1, wr_x1)
            inter_y1 = max(seat_y1, wr_y1)
            inter_x2 = min(seat_x2, wr_x2)
            inter_y2 = min(seat_y2, wr_y2)

            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            intersection_area = inter_w * inter_h

        #     if intersection_area > 0:
        #         # Crop the wrinkle mask to seat bounding box
        #         cropped_wrinkle_mask = wrinkle_mask[y:y+h, x:x+w]

        #         # Skip empty masks
        #         if not np.any(cropped_wrinkle_mask):
        #             continue

        #         # Encode new cropped wrinkle mask
        #         wr_rle_cropped = maskUtils.encode(np.asfortranarray(cropped_wrinkle_mask))
        #         wr_rle_cropped["counts"] = wr_rle_cropped["counts"].decode("utf-8")

        #         # Compute new bbox within cropped image
        #         x0, y0, w0, h0 = cv2.boundingRect(cropped_wrinkle_mask)
        #         wr_area = int(np.sum(cropped_wrinkle_mask))

        #         new_annotations.append({
        #             "id": new_annotation_id,
        #             "image_id": new_image_id,
        #             "category_id": 2,  # wrinkle
        #             "segmentation": wr_rle_cropped,
        #             "bbox": [x0, y0, w0, h0],
        #             "area": wr_area,
        #             "iscrowd": 0
        #         })
        #         new_annotation_id += 1

        # new_image_id += 1
            if inter_w * inter_h == 0:
                continue

            cropped_wrinkle_mask = wrinkle_mask[y:y+h, x:x+w]
            if not np.any(cropped_wrinkle_mask):
                continue

            contours, _ = cv2.findContours(cropped_wrinkle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            for contour in contours:
                if contour.shape[0] >= 3:
                    polygon = contour.flatten().astype(float).tolist()
                    polygons.append(polygon)

            if not polygons:
                continue

            wr_area = int(np.sum(cropped_wrinkle_mask))
            x0, y0, w0, h0 = cv2.boundingRect(cropped_wrinkle_mask)

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
