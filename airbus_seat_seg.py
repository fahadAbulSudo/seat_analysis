import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO  # YOLOv8

# Paths
INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/AIRBUS - EXCHANGE FOLDER"
OUTPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/AIRBUS_Output_SEG"
YOLO_MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/segment/output/yolov8_trained.pt"

# Load YOLOv8 segmentation model
yolo_model = YOLO(YOLO_MODEL_PATH)

# Define class label for segmentation
TARGET_CLASS = "seat"

# Function to process images recursively
def process_images(input_dir, output_dir):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                image_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_dir)
                save_dir = os.path.join(output_dir, relative_path)
                os.makedirs(save_dir, exist_ok=True)
                output_path = os.path.join(save_dir, file)

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Skipping {image_path}: Unable to read image.")
                    continue
                
                # Perform YOLO segmentation
                results = yolo_model(image)
                for result in results:
                    for i, mask in enumerate(result.masks.xy if result.masks else []):
                        class_id = int(result.boxes.cls[i].item())  # Convert tensor to int
                        if result.names[class_id] == TARGET_CLASS:
                            # Draw polygon mask
                            points = np.array(mask, dtype=np.int32)
                            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                    
                            print(result.names)    
                # Save segmented image
                cv2.imwrite(output_path, image)
                print(f"Processed {image_path} -> Saved to {output_path}")

# Run processing
process_images(INPUT_DIR, OUTPUT_DIR)
print("Inference complete. Segmented images saved.")
