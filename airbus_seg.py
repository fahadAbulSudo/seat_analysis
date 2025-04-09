import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO  # YOLOv8
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file

# Paths
INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/AIRBUS_TEST"
OUTPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/AIRBUS_Output_Segment"
YOLO_MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/yolov8n-seg.pt"
MODEL_PATH_TORN = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn/dataset/output/model_final.pth"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"

# Load YOLO segmentation model
yolo_model = YOLO(YOLO_MODEL_PATH)

# Load Mask R-CNN model
def load_model(model_path, class_names):
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

predictor_torn = load_model(MODEL_PATH_TORN, ["torn"])
predictor_wrinkle = load_model(MODEL_PATH_WRINKLE, ["wrinkle"])

def get_polylines_from_masks(instances):
    polylines = []
    masks = instances.pred_masks.cpu().numpy()
    for mask in masks:
        binary_mask = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 2:
                polylines.append(contour.reshape(-1, 2))
    return polylines

def process_images(input_dir, output_dir):
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
                print(f"Skipping {image_path}: Unable to read image.")
                continue

            yolo_results = yolo_model(image)[0]
            masks = yolo_results.masks.data.cpu().numpy()
            height, width = image.shape[:2]
            all_polylines = []

            for mask in masks:
                binary_mask = cv2.resize((mask > 0.5).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                x, y, w, h = cv2.boundingRect(binary_mask)
                cropped_image = image[y:y+h, x:x+w]
                cropped_mask = binary_mask[y:y+h, x:x+w]
                masked_roi = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

                torn_outputs = predictor_torn(masked_roi)
                wrinkle_outputs = predictor_wrinkle(masked_roi)

                for poly in get_polylines_from_masks(torn_outputs["instances"]):
                    all_polylines.append((poly + np.array([x, y]), (0, 0, 255)))

                for poly in get_polylines_from_masks(wrinkle_outputs["instances"]):
                    all_polylines.append((poly + np.array([x, y]), (0, 255, 0)))

            for poly, color in all_polylines:
                cv2.polylines(image, [poly], isClosed=True, color=color, thickness=2)

            cv2.imwrite(output_path, image)
            print(f"Processed {image_path} -> Saved to {output_path}")

# Run processing
process_images(INPUT_DIR, OUTPUT_DIR)
print("Inference complete. Segmented images saved.")
