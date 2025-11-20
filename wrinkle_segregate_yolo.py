import os
import cv2
import json
import numpy as np
from collections import defaultdict
from pycocotools import mask as maskUtils
from ultralytics import YOLO

# Load and return COCO annotation JSON
def load_coco_annotations(path):
    with open(path, "r") as f:
        return json.load(f)

# Find the category ID for "wrinkle" class in COCO categories
def get_wrinkle_category_id(categories):
    for cat in categories:
        if cat["name"].lower() == "wrinkle":
            return cat["id"]
    raise ValueError("Wrinkle category not found!")

# Group all annotations by image ID
def group_annotations_by_image(annotations):
    grouped = defaultdict(list)
    for ann in annotations:
        grouped[ann["image_id"]].append(ann)
    return grouped

# Apply binary mask to an RGB image channel-wise
def apply_mask_to_image(image, mask):
    masked = np.zeros_like(image)
    for c in range(3):
        masked[:, :, c] = image[:, :, c] * mask
    return masked

 # Extract external contours (polygons) from a binary mask
def extract_polygons_from_mask(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if contour.shape[0] >= 3:
            polygons.append(contour.flatten().astype(float).tolist())
    return polygons

# Draw polygons on image for visual debugging
def save_debug_image(image, polygons, path):
    for poly in polygons:
        try:
            poly_arr = np.array(poly, dtype=np.float32)
            if poly_arr.ndim != 1 or poly_arr.size % 2 != 0:
                continue  # not valid
            pts = poly_arr.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        except Exception as e:
            print(f"⚠️ Skipping malformed polygon: {poly}, error: {e}")
    cv2.imwrite(path, image)

def process_dataset(images_dir, annotations_path, output_dir, yolo_model_path, save_debug=True):
    # Load data and YOLO model
    coco_data = load_coco_annotations(annotations_path)
    wrinkle_cat_id = get_wrinkle_category_id(coco_data["categories"])
    annotations_by_image = group_annotations_by_image(coco_data["annotations"])
    yolo_model = YOLO(yolo_model_path)

    # Create output folders
    images_out_dir = os.path.join(output_dir, "images_A")
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(images_out_dir, exist_ok=True)
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    # Prepare new COCO output format
    new_images = []
    new_annotations = []
    new_categories = [
        {"id": 1, "name": "seat"},
        {"id": 2, "name": "wrinkle"}
    ]
    new_image_id = 1
    new_annotation_id = 1

    for img in coco_data["images"]:
        image_path = os.path.join(images_dir, img["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue

        height, width = image.shape[:2]
        anns = annotations_by_image[img["id"]]
        wrinkle_anns = [a for a in anns if a["category_id"] == wrinkle_cat_id]

        # Run YOLO model to detect seat masks
        yolo_results = yolo_model(image)[0]
        if not hasattr(yolo_results, "masks") or yolo_results.masks is None:
            continue

        seat_masks = yolo_results.masks.data.cpu().numpy()

        for seat_idx, seat_mask in enumerate(seat_masks):
            binary_seat = (seat_mask > 0.3).astype(np.uint8)
            resized_mask = cv2.resize(binary_seat, (width, height), interpolation=cv2.INTER_NEAREST)

            # Get bounding box around seat
            x, y, w, h = cv2.boundingRect(resized_mask)
            if w == 0 or h == 0:
                continue

            # Crop image and mask to seat region
            cropped_image = image[y:y+h, x:x+w]
            cropped_mask = resized_mask[y:y+h, x:x+w]
            masked_image = apply_mask_to_image(cropped_image, cropped_mask)

            # Save seat image
            file_stem = os.path.splitext(img["file_name"])[0]
            new_file_name = f"{file_stem}_seat{seat_idx}.jpg"
            save_path = os.path.join(images_out_dir, new_file_name)
            cv2.imwrite(save_path, masked_image)

            # Add seat annotation
            new_images.append({
                "id": new_image_id,
                "file_name": new_file_name,
                "height": h,
                "width": w
            })

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

            debug_polys = []

            for wr_ann in wrinkle_anns:
                # Decode wrinkle annotation
                if isinstance(wr_ann["segmentation"], list):
                    wr_rles = maskUtils.frPyObjects(wr_ann["segmentation"], height, width)
                    wr_rle = maskUtils.merge(wr_rles)
                else:
                    wr_rle = wr_ann["segmentation"]

                wrinkle_mask = maskUtils.decode(wr_rle).astype(np.uint8)

                # Find intersection between seat and wrinkle
                overlap = np.logical_and(resized_mask, wrinkle_mask).astype(np.uint8)

                if not np.any(overlap):
                    continue

                # Crop wrinkle mask to seat region
                cropped_wrinkle = overlap[y:y+h, x:x+w]
                if not np.any(cropped_wrinkle):
                    continue

                # Convert mask to polygons
                polygons = extract_polygons_from_mask(cropped_wrinkle)
                if not polygons:
                    continue

                # Save wrinkle annotation
                wr_area = int(np.sum(cropped_wrinkle))
                x0, y0, w0, h0 = cv2.boundingRect(cropped_wrinkle)
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
                debug_polys.extend(polygons)
                print(debug_polys)

            if save_debug and debug_polys:
                debug_path = os.path.join(debug_dir, f"{file_stem}_seat{seat_idx}_debug.jpg")
                save_debug_image(masked_image.copy(), debug_polys, debug_path)

            new_image_id += 1

    return {
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories
    }

# === Run This Section ===
if __name__ == "__main__":
    images_dir = "/home/fahadabul/mask_rcnn_skyhub/desired_dataset/images"
    annotations_path = "/home/fahadabul/mask_rcnn_skyhub/desired_dataset/coco_annotations.json"
    output_dir = "/home/fahadabul/mask_rcnn_skyhub/dataset_separated/test_mask"
    yolo_model_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/model_3rd/best.pt"
    output_ann_path = os.path.join(output_dir, "new_annotations_cleaned_result.json")

    coco_result = process_dataset(
        images_dir=images_dir,
        annotations_path=annotations_path,
        output_dir=output_dir,
        yolo_model_path=yolo_model_path,
        save_debug=True
    )

    with open(output_ann_path, "w") as f:
        json.dump(coco_result, f, indent=4)

    print("✅ Preprocessing complete. YOLO-seat + wrinkle mapped dataset created.")
