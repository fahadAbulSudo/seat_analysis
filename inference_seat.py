import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO  # YOLOv8

# Paths
INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/MSN_12763_new/seat_11-06"
OUTPUT_DIR = "./output_predictions/yolo_segmented_seats"
YOLO_MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/model_3rd/best.pt"

# Load YOLO segmentation model
yolo_model = YOLO(YOLO_MODEL_PATH)

def save_yolo_segmented_seats(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_dir)
            save_dir = os.path.join(output_dir, relative_path)
            os.makedirs(save_dir, exist_ok=True)

            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping {image_path}: Unable to read image.")
                continue

            results = yolo_model(image)[0]

            if results.masks is None:
                print(f"No seat mask found for {image_path}")
                continue

            masks = results.masks.data.cpu().numpy()
            height, width = image.shape[:2]

            for i, mask in enumerate(masks):
                binary_mask = cv2.resize((mask > 0.3).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

                # Get bounding box of the mask
                x, y, w, h = cv2.boundingRect(binary_mask)

                # Crop image and mask to the bounding box
                cropped_image = image[y:y+h, x:x+w]
                cropped_mask = binary_mask[y:y+h, x:x+w]

                # Apply mask
                masked_seat = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

                # Save result
                output_filename = f"{os.path.splitext(file)[0]}_seat_{i+1}.png"
                output_path = os.path.join(save_dir, output_filename)
                cv2.imwrite(output_path, masked_seat)

                print(f"Saved seat segment: {output_path}")

# Run YOLO-based seat segmentation
save_yolo_segmented_seats(INPUT_DIR, OUTPUT_DIR)
print("YOLO seat segmentation complete.")
