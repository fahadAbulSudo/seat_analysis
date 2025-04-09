import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file
import json

# ----------------------------
# Configuration
# ----------------------------
INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/AIRBUS_Output_Segment"
OUTPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/AIRBUS_Output_Segment"
YOLO_MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/yolov8n-seg.pt"
MODEL_PATH_TORN = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn/dataset/output/model_final.pth"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"
JSON_OUTPUT_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/output_annotated_f.json"

# Create necessary output directories
SEAT_DIR = os.path.join(OUTPUT_DIR, "output_seats")
DEFECT_DIR = os.path.join(OUTPUT_DIR, "output_defects")
FINAL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "output_annotated")
os.makedirs(SEAT_DIR, exist_ok=True)
os.makedirs(DEFECT_DIR, exist_ok=True)
os.makedirs(FINAL_OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Model Loading
# ----------------------------

yolo_model = YOLO(YOLO_MODEL_PATH)

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

# ----------------------------
# Helper Functions
# ----------------------------

def draw_mask_and_bbox(image, instances, label, color, offset=(0, 0)):
    masks = instances.pred_masks.cpu().numpy()
    for mask in masks:
        binary = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 2:
                contour = contour.reshape(-1, 2)
                contour = np.array([[pt[0]+offset[0], pt[1]+offset[1]] for pt in contour], dtype=np.int32)
                cv2.polylines(image, [contour], True, color, 2)
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# ----------------------------
# Processing Pipeline
# ----------------------------

def process_images():
    annotations = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "torn"}, {"id": 2, "name": "wrinkle"}]}
    annotation_id = 0
    image_id = 0

    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if not file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            # Get relative path to maintain folder structure
            relative_root = os.path.relpath(root, INPUT_DIR)
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load {file}")
                continue

            h, w = image.shape[:2]
            yolo_result = yolo_model(image)[0]
            
            # Keep track of the current image
            image_id += 1
            annotations["images"].append({"id": image_id, "file_name": os.path.relpath(image_path, INPUT_DIR), "width": w, "height": h})
            if yolo_result.masks is None:
                print(f"No masks found in {file}")
                continue
            for i, seg in enumerate(yolo_result.masks.data):
                mask = seg.cpu().numpy()
                mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                seat_mask = np.zeros_like(image)
                seat_mask[mask_resized == 1] = image[mask_resized == 1]
                
                # Maintain the input folder structure in the output directory
                seat_filename = f"{os.path.splitext(file)[0]}_seat{i}.jpg"
                seat_output_dir = os.path.join(SEAT_DIR, relative_root)
                os.makedirs(seat_output_dir, exist_ok=True)
                seat_path = os.path.join(seat_output_dir, seat_filename)
                cv2.imwrite(seat_path, seat_mask)

                # Detect defects (torn & wrinkle) on seat mask
                outputs_torn = predictor_torn(seat_mask)
                outputs_wrinkle = predictor_wrinkle(seat_mask)

                # Save intermediate defect annotation
                defect_annotated = seat_mask.copy()
                draw_mask_and_bbox(defect_annotated, outputs_torn["instances"], "Torn", (0, 0, 255))
                draw_mask_and_bbox(defect_annotated, outputs_wrinkle["instances"], "Wrinkle", (0, 255, 0))
                defect_output_dir = os.path.join(DEFECT_DIR, relative_root)
                os.makedirs(defect_output_dir, exist_ok=True)
                defect_path = os.path.join(defect_output_dir, seat_filename)
                cv2.imwrite(defect_path, defect_annotated)

                # Save annotations in COCO format
                for instance, label in zip([outputs_torn, outputs_wrinkle], ["torn", "wrinkle"]):
                    for mask in instance["instances"].pred_masks:
                        mask = mask.cpu().numpy()
                        contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            if len(contour) > 2:
                                contour = contour.reshape(-1, 2)
                                annotation_id += 1
                                annotations["annotations"].append({
                                    "id": annotation_id,
                                    "image_id": image_id,
                                    "category_id": 1 if label == "torn" else 2,
                                    "segmentation": [contour.flatten().tolist()],
                                    "area": cv2.contourArea(contour),
                                    "bbox": cv2.boundingRect(contour),
                                    "iscrowd": 0
                                })

                # Draw back the defects on the original image
                draw_mask_and_bbox(image, outputs_torn["instances"], "Torn", (0, 0, 255))
                draw_mask_and_bbox(image, outputs_wrinkle["instances"], "Wrinkle", (0, 255, 0))

            # Maintain folder structure in the final output directory
            final_output_dir = os.path.join(FINAL_OUTPUT_DIR, relative_root)
            os.makedirs(final_output_dir, exist_ok=True)
            final_path = os.path.join(final_output_dir, file)
            cv2.imwrite(final_path, image)
            print(f"Saved: {final_path}")

    # Save annotations as a COCO format JSON
    with open(JSON_OUTPUT_PATH, 'w') as f:
        json.dump(annotations, f)
    print(f"Annotations saved to {JSON_OUTPUT_PATH}")

# ----------------------------
# Run
# ----------------------------

if __name__ == "__main__":
    process_images()
    print("All images processed.")
