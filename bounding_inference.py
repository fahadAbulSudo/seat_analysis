import os
import cv2
import torch
import time
import numpy as np
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file

# Paths
MODEL_PATH_TORN = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn/dataset/output/model_final.pth"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"
IMAGE_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/test_image_no"
OUTPUT_DIR = "./output_predictions/bounding_box"

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

# Load both models
predictor_torn = load_model(MODEL_PATH_TORN, CLASS_NAMES_TORN)
predictor_wrinkle = load_model(MODEL_PATH_WRINKLE, CLASS_NAMES_WRINKLE)

# Process images
for image_name in os.listdir(IMAGE_DIR):
    start_time = time.time()
    image_path = os.path.join(IMAGE_DIR, image_name)
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_name}: Unable to read image.")
        continue
    
    img_height, img_width = image.shape[:2]
    total_image_area = img_width * img_height

    # Function to draw bounding boxes
def draw_predictions(image, outputs, class_label, color):
    instances = outputs["instances"].to("cpu")
    if instances.has("pred_boxes"):
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.tolist()
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{class_label} {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

    # Run both models sequentially
    outputs_torn = predictor_torn(image)
    image = draw_predictions(image, outputs_torn, "Torn", (0, 255, 0))  # Green boxes for torn
    
    outputs_wrinkle = predictor_wrinkle(image)
    image = draw_predictions(image, outputs_wrinkle, "Wrinkle", (0, 0, 255))  # Red boxes for wrinkle
    
    # Save output
    output_path = os.path.join(OUTPUT_DIR, f"pred_{image_name}")
    cv2.imwrite(output_path, image)
    
    end_time = time.time()
    print(f"Processed {image_name} -> Saved to {output_path} (Time: {end_time - start_time:.2f}s)")

print("Inference complete. Processed images saved in './output_predictions/'")
