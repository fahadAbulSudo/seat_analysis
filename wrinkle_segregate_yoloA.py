import os
import cv2
import json
import numpy as np
from pycocotools import mask as maskUtils
from collections import defaultdict
from ultralytics import YOLO

# Load COCO-style JSON annotation file
def load_coco_annotations(annotations_path):
    with open(annotations_path, "r") as f:
        return json.load(f)

# Get category ID for the "wrinkle" class from the category list
def get_wrinkle_category_id(categories):
    for cat in categories:
        if cat["name"].lower() == "wrinkle":
            return cat["id"]
    raise ValueError("Wrinkle category not found!")

# Group annotations by image_id (used to quickly find relevant annotations per image)
def get_annotations_by_image(annotations):
    grouped = defaultdict(list)
    for ann in annotations:
        grouped[ann["image_id"]].append(ann)
    return grouped

# Translate polygon points by an offset (typically to align with cropped image)
def translate_polygons(polygons, offset):
    translated = []
    for poly in polygons:
        points = np.array(poly).reshape(-1, 2)
        translated.append((points - offset).flatten().tolist())
    return translated

# Draw polygons on the image for debugging and save it
def save_debug_image(image, polys, path):
    for poly in polys:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(path, image)

def process_dataset(images_dir, annotations_path, output_dir, yolo_model_path, pad=10, save_debug=True):
    # Load annotations and YOLO model
    coco_data = load_coco_annotations(annotations_path)
    wrinkle_cat_id = get_wrinkle_category_id(coco_data["categories"])
    anns_by_image = get_annotations_by_image(coco_data["annotations"])
    yolo_model = YOLO(yolo_model_path)

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    debug_dir = os.path.join(output_dir, "output", "test")
    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)

    # Initialize new COCO-style lists for the output
    new_images, new_annotations = [], []
    new_categories = [
        {"id": 1, "name": "seat"},
        {"id": 2, "name": "wrinkle"}
    ]
    new_image_id = 1
    new_annotation_id = 1

    # Iterate over each image in the dataset
    for img in coco_data["images"]:
        image_path = os.path.join(images_dir, img["file_name"])
        image = cv2.imread(image_path)
        if image is None:
            continue

        height, width = image.shape[:2]
        anns = anns_by_image[img["id"]]

        # Filter only wrinkle annotations
        wrinkle_anns = [a for a in anns if a["category_id"] == wrinkle_cat_id]

        # Run YOLO model to get seat masks
        yolo_results = yolo_model(image)[0]
        if not hasattr(yolo_results, "masks") or yolo_results.masks is None:
            continue

        seat_masks = yolo_results.masks.data.cpu().numpy()

        # Process each seat mask individually
        for seat_idx, seat_mask in enumerate(seat_masks):
            binary_seat = (seat_mask > 0.3).astype(np.uint8)
            resized_mask = cv2.resize(binary_seat, (width, height), interpolation=cv2.INTER_NEAREST)

            # Get bounding box around the mask
            x, y, w, h = cv2.boundingRect(resized_mask)
            if w == 0 or h == 0:
                continue

            # Pad the bounding box for better cropping
            x = max(0, x - pad)
            y = max(0, y - pad)
            w = min(width - x, w + 2 * pad)
            h = min(height - y, h + 2 * pad)

            # Crop image and mask
            cropped_image = image[y:y+h, x:x+w]
            cropped_mask = resized_mask[y:y+h, x:x+w]

            # Apply mask to crop (i.e., remove non-seat background)
            masked_image = np.zeros_like(cropped_image)
            for c in range(3):
                masked_image[:, :, c] = cropped_image[:, :, c] * cropped_mask

            # Save cropped and masked seat image
            new_file_name = f"{os.path.splitext(img['file_name'])[0]}_seat{seat_idx}.jpg"
            print(new_file_name)
            save_path = os.path.join(output_dir, "images", new_file_name)
            print(save_path)
            cv2.imwrite(save_path, masked_image)

            new_images.append({
                "id": new_image_id,
                "file_name": save_path,
                "height": h,
                "width": w
            })

            # Save seat annotation as RLE (Run-Length Encoding) segmentation
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

            # For visual debugging
            debug_img = masked_image.copy()
            all_translated_polys = []

            # Process all wrinkles in this image
            for wr_ann in wrinkle_anns:
                # Convert segmentation to RLE
                if isinstance(wr_ann["segmentation"], list):
                    wr_rles = maskUtils.frPyObjects(wr_ann["segmentation"], height, width)
                    wr_rle = maskUtils.merge(wr_rles)
                else:
                    wr_rle = wr_ann["segmentation"]

                # Decode wrinkle mask
                wrinkle_mask = maskUtils.decode(wr_rle)

                # Check overlap between seat and wrinkle
                overlap = np.logical_and(resized_mask, wrinkle_mask).astype(np.uint8)
                if not np.any(overlap):
                    continue

                # Crop wrinkle mask to seat region
                cropped_wrinkle = overlap[y:y+h, x:x+w]
                if not np.any(cropped_wrinkle):
                    continue

                # Ignore RLE-only annotations (polygon needed)
                if not isinstance(wr_ann["segmentation"], list):
                    continue

                # Translate wrinkle polygons to cropped seat coordinates
                translated_polys = translate_polygons(wr_ann["segmentation"], np.array([[x, y]]))
                if not translated_polys:
                    continue

                # Append for saving debug visualization
                all_translated_polys.extend(translated_polys)

                # Compute bounding box and area from polygons
                all_points = np.array(translated_polys[0]).reshape(-1, 2)
                for additional_poly in translated_polys[1:]:
                    all_points = np.vstack((all_points, np.array(additional_poly).reshape(-1, 2)))
                x0b, y0b, w0b, h0b = cv2.boundingRect(all_points.astype(np.int32))
                area = int(cv2.contourArea(all_points.astype(np.int32)))

                # Save translated wrinkle annotation
                new_annotations.append({
                    "id": new_annotation_id,
                    "image_id": new_image_id,
                    "category_id": 2,
                    "segmentation": translated_polys,
                    "bbox": [x0b, y0b, w0b, h0b],
                    "area": area,
                    "iscrowd": 0
                })
                new_annotation_id += 1

            if save_debug:
                debug_path = os.path.join(debug_dir, f"{os.path.splitext(img['file_name'])[0]}_seat{seat_idx}_debug.jpg")
                save_debug_image(debug_img, all_translated_polys, debug_path)

            new_image_id += 1

    # Save final annotations
    with open(os.path.join(output_dir, "new_annotations.json"), "w") as f:
        json.dump({
            "images": new_images,
            "annotations": new_annotations,
            "categories": new_categories
        }, f, indent=4)

    print("\n✅ Preprocessing complete. YOLO-seat + wrinkle mapped dataset created.")

if __name__ == "__main__":
    images_dir = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/MSN_12741_right-20250805T030028Z-1-001/test/coco/seat1_paste/images"
    annotations_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/MSN_12741_right-20250805T030028Z-1-001/test/coco/seat1_paste/result.json"
    output_dir = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/MSN_12741_right-20250805T030028Z-1-001/test/coco/seat1_paste/extract"
    yolo_model_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/model_3rd/yolo_seat_back_best_model/best_only_seat_n_backseat_yolo.pt"  # <-- replace with actual model path

    process_dataset(
        images_dir=images_dir,
        annotations_path=annotations_path,
        output_dir=output_dir,
        yolo_model_path=yolo_model_path,
        pad=10,
        save_debug=True
    )
