import os
import cv2
import json
import glob
import numpy as np
from PIL import Image

# === Config ===
image_dir = "/home/swapnil/AIR_BUS/May/28th_data_preparation_for_mask_retraining/dataset/yolo/images"
label_dir = "/home/swapnil/AIR_BUS/May/28th_data_preparation_for_mask_retraining/dataset/yolo/labels/"
output_dir = "/home/swapnil/AIR_BUS/May/28th_data_preparation_for_mask_retraining/dataset/yolo/cropped_images/"
output_json = "/home/swapnil/AIR_BUS/May/28th_data_preparation_for_mask_retraining/dataset/yolo/cropped_wrinkle_annotations.json"
os.makedirs(output_dir, exist_ok=True)

# === COCO JSON structure ===
coco = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "wrinkle"}]
}
annotation_id = 1
image_id = 1

def yolo_to_pixel_polygon(pts, img_w, img_h):
    return [(float(pts[i]) * img_w, float(pts[i+1]) * img_h) for i in range(0, len(pts), 2)]

def get_segmentation_mask(polygon, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    polygon_np = np.array([polygon], dtype=np.int32)
    cv2.fillPoly(mask, polygon_np, 255)
    return mask

def safe_polygon_within_mask(polygon, mask):
    h, w = mask.shape
    for x, y in polygon:
        xi = min(max(int(x), 0), w - 1)
        yi = min(max(int(y), 0), h - 1)
        if mask[yi, xi] == 0:
            return False
    return True

for label_path in glob.glob(os.path.join(label_dir, "*.txt")):
    file_stem = os.path.splitext(os.path.basename(label_path))[0]
    image_path_jpg = os.path.join(image_dir, file_stem + ".jpg")
    image_path_png = os.path.join(image_dir, file_stem + ".png")
    
    image_path = image_path_jpg if os.path.exists(image_path_jpg) else image_path_png
    if not os.path.exists(image_path):
        print(f"Image not found for {file_stem}")
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    h, w = image.shape[:2]

    with open(label_path, "r") as f:
        lines = f.read().splitlines()

    seat_polygons = []
    wrinkle_polygons = []

    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        coords = parts[1:]

        if class_id == 0 and len(coords) >= 6:
            seat_polygons.append(yolo_to_pixel_polygon(coords, w, h))
        elif class_id == 1 and len(coords) >= 6:
            wrinkle_polygons.append(yolo_to_pixel_polygon(coords, w, h))

    for idx, seat_polygon in enumerate(seat_polygons):
        seat_mask = get_segmentation_mask(seat_polygon, image.shape)

        # Get bounding box from polygon
        seat_np = np.array(seat_polygon, dtype=np.int32)
        x, y, bw, bh = cv2.boundingRect(seat_np)
        if bw == 0 or bh == 0:
            print(f"Empty crop for {file_stem}, seat {idx}")
            continue

        # Blackout outside mask
        masked_image = np.zeros_like(image)
        masked_image[seat_mask == 255] = image[seat_mask == 255]

        # Crop to bounding box
        crop = masked_image[y:y+bh, x:x+bw]
        if crop.size == 0:
            print(f"Empty crop for {file_stem}, seat {idx}")
            continue

        # Save cropped image
        crop_filename = f"{file_stem}_seat{idx}.jpg"
        crop_path = os.path.join(output_dir, crop_filename)
        cv2.imwrite(crop_path, crop)

        # Add to COCO images
        coco["images"].append({
            "id": image_id,
            "file_name": crop_filename,
            "width": bw,
            "height": bh
        })

        # Check wrinkle polygons
        for poly in wrinkle_polygons:
            if not safe_polygon_within_mask(poly, seat_mask):
                continue

            # Remap to crop coordinates
            new_poly = [(x_ - x, y_ - y) for (x_ , y_) in poly]
            flat_poly = [float(f"{pt:.2f}") for pair in new_poly for pt in pair]

            # Calculate area and bounding box
            np_poly = np.array(new_poly, np.int32)
            area = cv2.contourArea(np_poly)
            bx, by, bw_, bh_ = cv2.boundingRect(np_poly)

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [flat_poly],
                "area": float(f"{area:.2f}"),
                "bbox": [bx, by, bw_, bh_],
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

# Save COCO JSON
with open(output_json, "w") as f:
    json.dump(coco, f, indent=2)

print("Done! Cropped images and COCO JSON saved.")
