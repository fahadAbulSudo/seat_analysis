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
YOLO_MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/best_model_yolo.pt"
MODEL_PATH_TORN = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn/dataset/output/model_final.pth"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"
IMAGE_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/test_image_no"
OUTPUT_DIR = "./output_predictions/segmentation_masks"
CROP_DIR = "./output_predictions/yolo_crops"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)

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

# Ensure tmp directory exists
TMP_DIR = "./tmp"
os.makedirs(TMP_DIR, exist_ok=True)

def draw_bounding_boxes(image, boxes, category_name, color):
    """
    Draw bounding boxes and labels on the image.
    """
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)  # Draw bounding box
        label = f"{category_name}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Process images
for image_name in os.listdir(IMAGE_DIR):
    start_time = time.time()
    image_path = os.path.join(IMAGE_DIR, image_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_name}: Unable to read image.")
        continue

    # Step 1: Detect seats with YOLO
    yolo_results = yolo_model(image)
    for i, result in enumerate(yolo_results):
        for j, box in enumerate(result.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            seat_crop = image[y1:y2, x1:x2]
            # if seat_crop.size == 0:
            #     continue

            # Save YOLO cropped image
            crop_path = os.path.join(CROP_DIR, f"{image_name}_crop_{j}.jpg")
            cv2.imwrite(crop_path, seat_crop)

            # Step 2: Run Mask R-CNN on the cropped seat region
            outputs_torn = predictor_torn(seat_crop)
            outputs_wrinkle = predictor_wrinkle(seat_crop)

            # Step 3: Process segmentation masks and draw polygons
            # def draw_segmentation_masks(mask_instances, category_name, color):
            #     masks = mask_instances.pred_masks.cpu().numpy()
            #     for mask in masks:
            #         resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (x2 - x1, y2 - y1))

            #         # Find contours
            #         contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #         for contour in contours:
            #             if len(contour) > 2:
            #                 contour = contour.reshape(-1, 2)
            #                 polygon = np.array([[x + x1, y + y1] for x, y in contour], dtype=np.int32)
            #                 cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=2)

            #                 # Get bounding box coordinates
            #                 x_min, y_min = np.min(polygon, axis=0)
            #                 x_max, y_max = np.max(polygon, axis=0)
            #                 draw_bounding_boxes(image, [[x_min, y_min, x_max, y_max]], category_name, color)

            #                 # Save the temporary image with drawn polygons and bounding boxes
            #                 tmp_path = os.path.join(TMP_DIR, f"tmp_seg_{image_name}")
            #                 cv2.imwrite(tmp_path, image)
            #                 print(f"Saved temporary image at {tmp_path}")
            def draw_segmentation_masks(mask_instances, category_name, color):
                masks = mask_instances.pred_masks.cpu().numpy()
                for mask in masks:
                    resized_mask = cv2.resize(mask.astype(np.uint8) * 255, (x2 - x1, y2 - y1))

                    # Find contours
                    contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for contour in contours:
                        if len(contour) > 2:
                            contour = contour.reshape(-1, 2)
                            points = np.array([[x + x1, y + y1] for x, y in contour], dtype=np.int32)

                            # Draw individual lines instead of a polyline
                            for k in range(len(points) - 1):
                                cv2.line(image, tuple(points[k]), tuple(points[k + 1]), color, 2)
                            
                            # Optionally, close the shape by connecting the last point to the first
                            cv2.line(image, tuple(points[-1]), tuple(points[0]), color, 2)

                            # Get bounding box coordinates
                            x_min, y_min = np.min(points, axis=0)
                            x_max, y_max = np.max(points, axis=0)
                            draw_bounding_boxes(image, [[x_min, y_min, x_max, y_max]], category_name, color)

                            # Save the temporary image with drawn lines and bounding boxes
                            tmp_path = os.path.join(TMP_DIR, f"tmp_seg_{image_name}")
                            cv2.imwrite(tmp_path, image)
                            print(f"Saved temporary image at {tmp_path}")
            # Draw torn and wrinkle segmentation masks with bounding boxes
            draw_segmentation_masks(outputs_torn["instances"], "Torn", (0, 0, 255))      # Red for torn
            draw_segmentation_masks(outputs_wrinkle["instances"], "Wrinkle", (0, 255, 0))  # Green for wrinkles

    # Save the final processed image
    output_path = os.path.join(OUTPUT_DIR, f"seg_{image_name}")
    cv2.imwrite(output_path, image)

    end_time = time.time()
    print(f"Processed {image_name} -> Saved to {output_path} (Time: {end_time - start_time:.2f}s)")



"""
    segmentation = annotation["segmentation"]
    for segment in segmentation:
        polygon = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)
"""
print("Inference complete. Segmented images saved in './output_predictions/segmentation_masks/'")
