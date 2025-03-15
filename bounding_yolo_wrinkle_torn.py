import os
import cv2
import torch
import time
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file
from ultralytics import YOLO  # Import YOLO

# Paths
# MODEL_PATH_TORN = "./latest_image_mask_rcnn_torn.pth"
# MODEL_PATH_WRINKLE = "./best_model_wrinkle.pth"
# MODEL_YOLO = "./best_model_yolo.pt"
# IMAGE_DIR = "./test_images"
# OUTPUT_DIR = "./output_predictions"
MODEL_PATH_TORN = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn/dataset/output/model_final.pth"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"
MODEL_YOLO = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/best_model_yolo.pt"
IMAGE_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/test_image_old"
OUTPUT_DIR = "./output_predictions/bounding_boxes"

# Class Names
CLASS_NAMES_TORN = ["torn"]
CLASS_NAMES_WRINKLE = ["wrinkle"]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mask_rcnn(model_path, class_names):
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

def overlap_ratio(boxA, boxB):
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB
    inter_x1, inter_y1 = max(x1A, x1B), max(y1A, y1B)
    inter_x2, inter_y2 = min(x2A, x2B), min(y2A, y2B)
    inter_width, inter_height = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height
    areaA, areaB = (x2A - x1A) * (y2A - y1A), (x2B - x1B) * (y2B - y1B)
    if inter_area == 0:
        return 0.0, 0.0
    return inter_area / areaA, inter_area / areaB

def merge_bounding_boxes(boxes, scores, threshold=0.1):
    merged_boxes, merged_scores, used = [], [], set()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        if i in used:
            continue
        score, merged = scores[i], False
        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            x1B, y1B, x2B, y2B = boxes[j]
            overlapA, overlapB = overlap_ratio([x1, y1, x2, y2], [x1B, y1B, x2B, y2B])
            if overlapA > 0.85 and overlapB < 0.2 or overlapB > 0.85 and overlapA < 0.2:
                if scores[i] > scores[j]:
                    merged_boxes.append([x1, y1, x2, y2])
                    merged_scores.append(score)
                    used.add(j)
                else:
                    merged_boxes.append([x1B, y1B, x2B, y2B])
                    merged_scores.append(scores[j])
                    used.add(j)
            elif overlapA > 0 or overlapB > 0:
                new_x1, new_y1, new_x2, new_y2 = min(x1, x1B), min(y1, y1B), max(x2, x2B), max(y2, y2B)
                merged_boxes.append([new_x1, new_y1, new_x2, new_y2])
                merged_scores.append(max(score, scores[j]))
                used.add(j)
        if not merged:
            merged_boxes.append([x1, y1, x2, y2])
            merged_scores.append(score)
    return merged_boxes, merged_scores

# Load models
yolo_model = YOLO(MODEL_YOLO)
predictor_torn = load_mask_rcnn(MODEL_PATH_TORN, CLASS_NAMES_TORN)
predictor_wrinkle = load_mask_rcnn(MODEL_PATH_WRINKLE, CLASS_NAMES_WRINKLE)

# Process images
for image_name in os.listdir(IMAGE_DIR):
    image_path = os.path.join(IMAGE_DIR, image_name)
    image = cv2.imread(image_path)
    if image is None:
        continue

    results = yolo_model(image)  # YOLO detects seats
    for result in results:
        seat_boxes = result.boxes.xyxy.cpu().numpy()
        for (sx1, sy1, sx2, sy2) in seat_boxes:
            seat_crop = image[int(sy1):int(sy2), int(sx1):int(sx2)]
            outputs_torn, outputs_wrinkle = predictor_torn(seat_crop), predictor_wrinkle(seat_crop)
            boxes_torn, scores_torn = outputs_torn["instances"].pred_boxes.tensor.cpu().numpy(), outputs_torn["instances"].scores.cpu().numpy()
            boxes_wrinkle, scores_wrinkle = outputs_wrinkle["instances"].pred_boxes.tensor.cpu().numpy(), outputs_wrinkle["instances"].scores.cpu().numpy()
            merged_boxes_torn, merged_scores_torn = merge_bounding_boxes(boxes_torn, scores_torn)
            merged_boxes_wrinkle, merged_scores_wrinkle = merge_bounding_boxes(boxes_wrinkle, scores_wrinkle)
            for (x1, y1, x2, y2), score in zip(merged_boxes_torn, merged_scores_torn):
                cv2.rectangle(image, (int(sx1 + x1), int(sy1 + y1)), (int(sx1 + x2), int(sy1 + y2)), (0, 255, 0), 2)
                cv2.putText(image, f"Torn {score:.2f}", (int(sx1 + x1), int(sy1 + y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            for (x1, y1, x2, y2), score in zip(merged_boxes_wrinkle, merged_scores_wrinkle):
                cv2.rectangle(image, (int(sx1 + x1), int(sy1 + y1)), (int(sx1 + x2), int(sy1 + y2)), (0, 0, 255), 2)
                cv2.putText(image, f"Wrinkle {score:.2f}", (int(sx1 + x1), int(sy1 + y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"output_{image_name}"), image)
print("Processing complete.")
