import os
import cv2
import torch
import time
import numpy as np
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.model_zoo import get_config_file

# Paths
MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/best_model_old.pth"
IMAGE_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/test_images"
OUTPUT_DIR = "./output_predictions/bounding_box"

# Define class labels
CLASS_NAMES = ["torn", "wrinkle"]  # Modify if more classes exist

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if the model weights exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

# Register a dummy dataset (required for Detectron2)
DATASET_NAME = "custom_dataset"
if DATASET_NAME not in DatasetCatalog.list():
    DatasetCatalog.register(DATASET_NAME, lambda: [])
    MetadataCatalog.get(DATASET_NAME).set(thing_classes=CLASS_NAMES)

# Load model configuration
cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)
cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize predictor
predictor = DefaultPredictor(cfg)

# Process images
for image_name in os.listdir(IMAGE_DIR):
    start_time = time.time()
    image_path = os.path.join(IMAGE_DIR, image_name)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipping {image_name}: Unable to read image.")
        continue

    # Get image dimensions
    img_height, img_width = image.shape[:2]
    total_image_area = img_width * img_height

    # Run inference
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")

    # Extract predictions
    pred_classes = instances.pred_classes.tolist()
    scores = instances.scores.tolist()
    print(f"{image_name} -> Predicted class indices: {pred_classes}")

    # Draw bounding boxes and segmentation masks if detected
    if instances.has("pred_boxes") and instances.has("pred_masks"):
        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()  # Convert to numpy (boolean masks)

        for i, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
            x1, y1, x2, y2 = map(int, box)  # Convert to integers for OpenCV
            width, height = x2 - x1, y2 - y1  # Compute bounding box width & height

            # Compute percentages relative to image dimensions
            width_pct = (width / img_width) * 100
            height_pct = (height / img_height) * 100

            # Compute segment area percentage
            segment_area = np.sum(mask)
            segment_area_pct = (segment_area / total_image_area) * 100

            label = f"Torn {score:.2f}"
            details = f"W: {width_pct:.1f}%, H: {height_pct:.1f}%, Area: {segment_area_pct:.1f}%"

            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Put confidence score
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Put percentage details below the box
            cv2.putText(image, details, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"Bounding Box {i}: x1={x1}, y1={y1}, x2={x2}, y2={y2}, Width={width_pct:.1f}%, Height={height_pct:.1f}%, Area={segment_area_pct:.1f}%, Confidence: {score:.2f}")

    else:
        print(f"No bounding boxes found for {image_name}.")

    # Save the image
    output_path = os.path.join(OUTPUT_DIR, f"pred_{image_name}")
    cv2.imwrite(output_path, image)

    # Log processing time
    end_time = time.time()
    print(f"Processed {image_name} -> Saved to {output_path} (Time: {end_time - start_time:.2f}s)")

print("Inference complete. Processed images saved in './output_predictions/'")
