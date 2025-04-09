import os
import cv2
import torch
import time
import numpy as np
from ultralytics import YOLO  # YOLOv8
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file

# Paths
INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/AIRBUS_Output_Segment"
OUTPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/AIRBUS_Output_Segment"
YOLO_MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/yolov8n-seg.pt"
MODEL_PATH_TORN = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn/dataset/output/model_final.pth"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"

# Load YOLO model
yolo_model = YOLO(YOLO_MODEL_PATH)

# Define class labels
CLASS_NAMES_TORN = ["torn"]
CLASS_NAMES_WRINKLE = ["wrinkle"]

# Function to load Mask R-CNN model
def load_model(model_path, class_names):
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

# Load Mask R-CNN models
predictor_torn = load_model(MODEL_PATH_TORN, CLASS_NAMES_TORN)
predictor_wrinkle = load_model(MODEL_PATH_WRINKLE, CLASS_NAMES_WRINKLE)

def draw_bounding_boxes(image, boxes, category_name, color):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, category_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_segmentation_masks(image, mask_instances, category_name, color, x1, y1):
    masks = mask_instances.pred_masks.cpu().numpy()
    for mask in masks:
        resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (mask.shape[1], mask.shape[0]))
        contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 2:
                contour = contour.reshape(-1, 2)
                points = np.array([[x + x1, y + y1] for x, y in contour], dtype=np.int32)
                cv2.polylines(image, [points], isClosed=True, color=color, thickness=2)
                x_min, y_min = np.min(points, axis=0)
                x_max, y_max = np.max(points, axis=0)
                draw_bounding_boxes(image, [[x_min, y_min, x_max, y_max]], category_name, color)

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
                
                yolo_results = yolo_model(image)
                for result in yolo_results:
                    for box in result.boxes.xyxy:
                        x1, y1, x2, y2 = map(int, box)
                        seat_crop = image[y1:y2, x1:x2]
                        # if seat_crop.size == 0:
                        #     continue
                        
                        outputs_torn = predictor_torn(seat_crop)
                        outputs_wrinkle = predictor_wrinkle(seat_crop)
                        
                        draw_segmentation_masks(image, outputs_torn["instances"], "Torn", (0, 0, 255), x1, y1)
                        draw_segmentation_masks(image, outputs_wrinkle["instances"], "Wrinkle", (0, 255, 0), x1, y1)
                
                cv2.imwrite(output_path, image)
                print(f"Processed {image_path} -> Saved to {output_path}")

# Run processing
process_images(INPUT_DIR, OUTPUT_DIR)
print("Inference complete. Segmented images saved.")
