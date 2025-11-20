import os
import cv2
from ultralytics import YOLO
import torch

# Path to model and image directory
MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/model_3rd/yolo_seat_back_best_model/best_only_seat_n_backseat.pt"
IMAGE_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/model_3rd/yolo_seat_back_best_model/"

# Load YOLO model
model = YOLO(MODEL_PATH)

# Get all image paths (you can customize the extensions as needed)
image_paths = [
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"Found {len(image_paths)} images.\n")

for image_path in image_paths:
    print(f"\n--- Running on: {os.path.basename(image_path)} ---")
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ Unable to read {image_path}")
        continue

    # Run inference
    results = model(image)[0]

    # Print overall results object
    print("Result keys:", results.__dict__.keys())

    # Print classes detected
    if results.boxes is not None and results.boxes.cls is not None:
        print("Detected classes (boxes):", results.boxes.cls.cpu().numpy())
    else:
        print("No boxes detected.")

    # Print masks and associated classes
    if results.masks is not None and results.masks.data is not None:
        print("Detected masks shape:", results.masks.data.shape)  # (N, H, W)
        if results.masks.data.shape[0] == results.boxes.cls.shape[0]:
            print("Classes for masks:", results.boxes.cls.cpu().numpy())
        else:
            print("Mismatch between masks and class labels")
    else:
        print("No masks detected.")
