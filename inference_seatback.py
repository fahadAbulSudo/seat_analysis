import os
import cv2
import numpy as np
from ultralytics import YOLO
import random

# ---------------- Paths ----------------
YOLO_MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/model_3rd/yolo_seat_back_best_model/best_only_seat_n_backseat_yolo.pt"

INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/back_test"
OUTPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/back_mask_test"

# ---------------- Model ----------------
yolo_model = YOLO(YOLO_MODEL_PATH)

# ---------------- Utils ----------------

def add_random_anomalies(image, mask, num_anomalies=3, max_size=0.05):
    """
    Insert random irregular anomalies into the masked ROI region.
    - image: Original image
    - mask: Binary mask (1 where seat_back is)
    - num_anomalies: number of synthetic defects to add
    - max_size: relative max anomaly size (fraction of ROI smaller side)
    Returns: image with anomalies
    """
    out_img = image.copy()
    coords = np.argwhere(mask > 0)  # seat_back pixels

    if coords.size == 0:
        return out_img

    h, w = mask.shape
    min_side = min(h, w)
    max_radius = int(min_side * max_size)  # much smaller than before

    for _ in range(num_anomalies):
        # Pick random center inside mask
        y, x = coords[np.random.choice(len(coords))]
        radius = random.randint(3, max(5, max_radius))  # small blobs

        # Random color (simulate anomaly)
        color = (
            random.randint(0, 60),   # darker R
            random.randint(0, 60),   # darker G
            random.randint(0, 60)    # darker B
        )

        # Create irregular polygon around (x, y)
        pts = []
        num_points = random.randint(5, 8)  # irregular polygon sides
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points + random.uniform(-0.2, 0.2)
            r = radius + random.randint(-radius//3, radius//3)
            px = int(x + r * np.cos(angle))
            py = int(y + r * np.sin(angle))
            pts.append([px, py])
        pts = np.array(pts, np.int32).reshape((-1, 1, 2))

        # Draw filled irregular anomaly
        cv2.fillPoly(out_img, [pts], color)

    return out_img

def add_scratches(image, mask, num_scratches=5, length_range=(20, 60), thickness_range=(1, 3)):
    out_img = image.copy()
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return out_img

    h, w = mask.shape

    for _ in range(num_scratches):
        # pick a random pixel inside the seat mask
        y, x = coords[np.random.choice(len(coords))]

        # random line properties
        length = random.randint(*length_range)
        thickness = random.randint(*thickness_range)
        angle = random.uniform(0, np.pi)

        # line end point
        x2 = int(x + length * np.cos(angle))
        y2 = int(y + length * np.sin(angle))

        # keep inside image bounds
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)

        # grayish scratch color
        color = (random.randint(50, 120),) * 3

        cv2.line(out_img, (x, y), (x2, y2), color, thickness)

    return out_img

# ---------------- Main ----------------
def process_images(input_dir, output_dir, simulate_anomalies=False):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_dir)
            save_dir = os.path.join(output_dir, relative_path)
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, file)

            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠ Skipping {image_path} (unreadable)")
                continue

            # Run YOLO
            results = yolo_model(image)[0]

            if results.masks is None:
                print(f"⚠ No masks found in {image_path}")
                continue

            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes
            class_ids = boxes.cls.cpu().numpy().astype(int)
            id_to_name = yolo_model.names

            found = False
            for idx, mask in enumerate(masks):
                cls_name = id_to_name[class_ids[idx]].lower()
                if cls_name == "seat_back":
                    # Binary mask
                    binary_mask = (mask > 0.3).astype(np.uint8)
                    binary_mask = cv2.resize(binary_mask, (image.shape[1], image.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)

                    # Apply mask to extract seat_back region
                    seat_back = cv2.bitwise_and(image, image, mask=binary_mask)

                    # Crop ROI
                    x, y, w, h = cv2.boundingRect(binary_mask)
                    seat_back_cropped = seat_back[y:y+h, x:x+w]
                    seat_back_mask = binary_mask[y:y+h, x:x+w]

                    # Optionally add anomalies
                    if simulate_anomalies:
                        seat_back_cropped = add_scratches(seat_back_cropped, seat_back_mask)

                    # Save seat_back image (with or without anomalies)
                    cv2.imwrite(output_path, seat_back_cropped)
                    print(f"✅ Saved seat_back ROI: {output_path}")
                    found = True
                    break  # only first seat_back mask

            if not found:
                print(f"⚠ No seat_back detected in {image_path}")

if __name__ == "__main__":
    # Set simulate_anomalies=True to inject synthetic defects
    process_images(INPUT_DIR, OUTPUT_DIR, simulate_anomalies=True)
    print("🎯 Done! Seat_back ROI images saved.")
