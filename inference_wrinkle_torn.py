import os
import cv2
import torch
import time
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.model_zoo import get_config_file

# Paths
MODEL_PATH_TORN = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn/dataset/output/model_final.pth"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"
IMAGE_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/MSN_12763_new/Seats 11-06"
OUTPUT_DIR = "./output_predictions/segmentation_masks"
TMP_DIR = "./tmp2"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# Define class labels
CLASS_NAMES_TORN = ["torn"]
CLASS_NAMES_WRINKLE = ["wrinkle"]

# Function to load model configuration
def load_model(model_path, class_names):
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)    

# Load both models
predictor_torn = load_model(MODEL_PATH_TORN, CLASS_NAMES_TORN)
predictor_wrinkle = load_model(MODEL_PATH_WRINKLE, CLASS_NAMES_WRINKLE)

def draw_bounding_boxes(image, boxes, category_name, color):
    """
    Draw bounding boxes and labels on the image.
    """
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, category_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_segmentation_masks(image, mask_instances, category_name, color):
    masks = mask_instances.pred_masks.cpu().numpy()
    for mask in masks:
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) > 2:
                contour = contour.reshape(-1, 2)
                for k in range(len(contour) - 1):
                    cv2.line(image, tuple(contour[k]), tuple(contour[k + 1]), color, 2)
                cv2.line(image, tuple(contour[-1]), tuple(contour[0]), color, 2)
                
                # Get bounding box coordinates
                x_min, y_min = np.min(contour, axis=0)
                x_max, y_max = np.max(contour, axis=0)
                draw_bounding_boxes(image, [[x_min, y_min, x_max, y_max]], category_name, color)

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

    # Draw segmentation masks and bounding boxes
    draw_segmentation_masks(image, outputs_torn["instances"], "Torn", (0, 0, 255))  # Red for torn
    draw_segmentation_masks(image, outputs_wrinkle["instances"], "Wrinkle", (0, 255, 0))  # Green for wrinkles

    # Save the final processed image
    output_path = os.path.join(OUTPUT_DIR, f"seg_{image_name}")
    cv2.imwrite(output_path, image)

    # Save temp image
    tmp_path = os.path.join(TMP_DIR, f"tmp_seg_{image_name}")
    cv2.imwrite(tmp_path, image)
    
    end_time = time.time()
    print(f"Processed {image_name} -> Saved to {output_path} (Time: {end_time - start_time:.2f}s)")

print("Inference complete. Segmented images saved in './output_predictions/segmentation_masks/'")
