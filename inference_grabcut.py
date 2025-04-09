import os
import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO  # YOLOv8
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.model_zoo import get_config_file

# Paths
YOLO_MODEL_PATH = "/home/satyashish/Desktop/final_image_data/training_yolo_v9_passenger_seats/best_model_m.pt"
MODEL_PATH_TORN = "/home/satyashish/Desktop/final_image_data/training_yolo_v9_passenger_seats/notebooks/model_final_torn.pth"
MODEL_PATH_WRINKLE = "/home/satyashish/Desktop/final_image_data/training_yolo_v9_passenger_seats/notebooks/best_model_wrinkle.pth"
IMAGE_DIR = "/home/satyashish/Desktop/final_image_data/training_yolo_v9_passenger_seats/client_images"
CROP_DIR = "/home/satyashish/Desktop/final_benchmarking_3/cropped"
FOREGROUND_OUTPUT_DIR = "/home/satyashish/Desktop/final_benchmarking_3/foreground"
OUTPUT_DIR = "/home/satyashish/Desktop/final_benchmarking_3/output"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)
os.makedirs(FOREGROUND_OUTPUT_DIR, exist_ok=True)

# Load YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)

# Define class labels
CLASS_NAMES_TORN = ["torn"]
CLASS_NAMES_WRINKLE = ["wrinkle"]

# Function to load Mask R-CNN model
def load_model(model_path, class_names):
    cfg = get_cfg()
    cfg.merge_from_file("/home/satyashish/Desktop/final_image_data/training_yolo_v9_passenger_seats/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

# Load Mask R-CNN models
predictor_torn = load_model(MODEL_PATH_TORN, CLASS_NAMES_TORN)
predictor_wrinkle = load_model(MODEL_PATH_WRINKLE, CLASS_NAMES_WRINKLE)

# Process images
for image_name in os.listdir(IMAGE_DIR):
    start_time = time.time()
    image_path = os.path.join(IMAGE_DIR, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_name}: Unable to read image.")
        continue

    # Step 1: Detect objects with YOLO
    yolo_results = yolo_model(image)
    for i, result in enumerate(yolo_results):
        for j, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            seat_crop = image[y1:y2, x1:x2]
            if seat_crop.size == 0:
                continue

            # Save YOLO cropped image
            crop_path = os.path.join(CROP_DIR, f"{image_name}_crop_{j}.jpg")
            cv2.imwrite(crop_path, seat_crop)

            # Step 2: Apply GrabCut
            mask = np.zeros(seat_crop.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            rect = (10, 10, seat_crop.shape[1] - 10, seat_crop.shape[0] - 10)
            cv2.grabCut(seat_crop, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            grabcut_result = seat_crop * mask2[:, :, np.newaxis]
            fg_output_path = os.path.join(FOREGROUND_OUTPUT_DIR, f"{image_name}_fg_{j}.jpg")
            cv2.imwrite(fg_output_path, grabcut_result)

            # Step 3: Run Mask R-CNN on the extracted foreground
            outputs_torn = predictor_torn(grabcut_result)
            outputs_wrinkle = predictor_wrinkle(grabcut_result)

            # Draw segmentation masks
            def draw_segmentation_masks(mask_instances, color):
                masks = mask_instances.pred_masks.cpu().numpy()
                for mask in masks:
                    resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (x2 - x1, y2 - y1))
                    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) > 2:
                            contour = contour.reshape(-1, 2)
                            points = np.array([[x + x1, y + y1] for x, y in contour], dtype=np.int32)
                            cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
            
            draw_segmentation_masks(outputs_torn["instances"], (0, 0, 255))
            draw_segmentation_masks(outputs_wrinkle["instances"], (0, 255, 0))
    
    # Save the final processed image
    output_path = os.path.join(OUTPUT_DIR, f"seg_{image_name}")
    cv2.imwrite(output_path, image)

    end_time = time.time()
    print(f"Processed {image_name} -> Saved to {output_path} (Time: {end_time - start_time:.2f}s)")

print("Inference complete. Segmented images saved in './output_predictions/segmentation_masks/'")
