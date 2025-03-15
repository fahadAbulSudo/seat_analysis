import os
import cv2
import torch
import time
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file

# Paths
MODEL_PATH_TORN = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn/dataset/output/model_final.pth"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"
MODEL_YOLO = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/best_model_yolo.pt"
IMAGE_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/test_image_old"
OUTPUT_DIR = "./output_predictions/bounding_boxes"

# Define class labels
CLASS_NAMES_TORN = ["torn"]
CLASS_NAMES_WRINKLE = ["wrinkle"]

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Function to load model configuration
def load_model(model_path, class_names):
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

# Function to compute overlap percentage
def overlap_ratio(boxA, boxB):
    x1A, y1A, x2A, y2A = boxA
    x1B, y1B, x2B, y2B = boxB

    inter_x1 = max(x1A, x1B)
    inter_y1 = max(y1A, y1B)
    inter_x2 = min(x2A, x2B)
    inter_y2 = min(y2A, y2B)

    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    areaA = (x2A - x1A) * (y2A - y1A)
    areaB = (x2B - x1B) * (y2B - y1B)
    
    if inter_area == 0:
        return 0.0, 0.0

    overlapA = inter_area / areaA
    overlapB = inter_area / areaB

    return overlapA, overlapB

# Custom merging function
def merge_bounding_boxes(boxes, scores, threshold=0.1):
    merged_boxes = []
    merged_scores = []

    used = set()
    for i in range(len(boxes)):
        if i in used:
            continue

        x1, y1, x2, y2 = boxes[i]
        score = scores[i]
        merged = False

        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            
            x1B, y1B, x2B, y2B = boxes[j]
            scoreB = scores[j]

            overlapA, overlapB = overlap_ratio([x1, y1, x2, y2], [x1B, y1B, x2B, y2B])

            # Condition 1: No overlap, keep both
            if overlapA == 0 and overlapB == 0:
                continue

            # Condition 3: Full containment case
            if (overlapA > 0.85 and overlapB < 0.2) or (overlapB > 0.85 and overlapA < 0.2):
                print(overlapA, overlapB)
                if score > scoreB:
                    merged_boxes.append([x1, y1, x2, y2])
                    merged_scores.append(score)
                    used.add(j)
                else:
                    merged_boxes.append([x1B, y1B, x2B, y2B])
                    merged_scores.append(scoreB)
                    used.add(j)
                continue  # No need to check further for this pair

            # Condition 2: Partial overlap, merge with max area
            if overlapA > 0 or overlapB > 0:
                print("second conditi",overlapA, overlapB)
                new_x1 = min(x1, x1B)
                new_y1 = min(y1, y1B)
                new_x2 = max(x2, x2B)
                new_y2 = max(y2, y2B)
                new_score = max(score, scoreB)

                merged_boxes.append([new_x1, new_y1, new_x2, new_y2])
                merged_scores.append(new_score)
                used.add(j)
                continue  # Continue checking for other overlaps

        if not merged:
            merged_boxes.append([x1, y1, x2, y2])
            merged_scores.append(score)
    
    return merged_boxes, merged_scores

# Load both models
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

    # Run inference for torn and wrinkle
    outputs_torn = predictor_torn(image)
    outputs_wrinkle = predictor_wrinkle(image)

    instances_torn = outputs_torn["instances"].to("cpu")
    instances_wrinkle = outputs_wrinkle["instances"].to("cpu")

    # Process torn detections
    if instances_torn.has("pred_boxes"):
        boxes_torn = instances_torn.pred_boxes.tensor.numpy()
        scores_torn = instances_torn.scores.numpy()
        merged_boxes_torn, merged_scores_torn = merge_bounding_boxes(boxes_torn, scores_torn)

        for (x1, y1, x2, y2), score in zip(merged_boxes_torn, merged_scores_torn):
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green for torn
            label = f"Torn {score:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Process wrinkle detections
    if instances_wrinkle.has("pred_boxes"):
        boxes_wrinkle = instances_wrinkle.pred_boxes.tensor.numpy()
        scores_wrinkle = instances_wrinkle.scores.numpy()
        merged_boxes_wrinkle, merged_scores_wrinkle = merge_bounding_boxes(boxes_wrinkle, scores_wrinkle)

        for (x1, y1, x2, y2), score in zip(merged_boxes_wrinkle, merged_scores_wrinkle):
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # Red for wrinkle
            label = f"Wrinkle {score:.2f}"
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save output image
    output_path = os.path.join(OUTPUT_DIR, f"bbox_{image_name}")
    cv2.imwrite(output_path, image)

    print(f"Processed {image_name} -> Saved to {output_path}")

print("Inference complete.")
