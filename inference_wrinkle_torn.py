import os
import cv2
import torch
import time
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.model_zoo import get_config_file

# Paths
MODEL_PATH_TORN = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn/dataset/output/model_final.pth"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"
IMAGE_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/test_image_no"
OUTPUT_DIR = "./output_predictions/segmentation_masks"

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

# Register dataset catalogs separately for torn and wrinkle
DATASET_NAME_TORN = "dataset_torn"
DATASET_NAME_WRINKLE = "dataset_wrinkle"

if DATASET_NAME_TORN not in DatasetCatalog.list():
    DatasetCatalog.register(DATASET_NAME_TORN, lambda: [])
    MetadataCatalog.get(DATASET_NAME_TORN).set(thing_classes=CLASS_NAMES_TORN)

if DATASET_NAME_WRINKLE not in DatasetCatalog.list():
    DatasetCatalog.register(DATASET_NAME_WRINKLE, lambda: [])
    MetadataCatalog.get(DATASET_NAME_WRINKLE).set(thing_classes=CLASS_NAMES_WRINKLE)

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

    # Create separate visualizers
    v_torn = Visualizer(image[:, :, ::-1], MetadataCatalog.get(DATASET_NAME_TORN), scale=1.0)
    v_torn = v_torn.draw_instance_predictions(outputs_torn["instances"].to("cpu"))

    v_wrinkle = Visualizer(image[:, :, ::-1], MetadataCatalog.get(DATASET_NAME_WRINKLE), scale=1.0)
    v_wrinkle = v_wrinkle.draw_instance_predictions(outputs_wrinkle["instances"].to("cpu"))

    # Convert drawn images back to numpy arrays
    mask_torn = v_torn.get_image()
    mask_wrinkle = v_wrinkle.get_image()

    # Blend the two images using weighted sum (to ensure visibility of both)
    blended_image = cv2.addWeighted(mask_torn, 0.5, mask_wrinkle, 0.5, 0)

    # Save the final blended image
    output_path = os.path.join(OUTPUT_DIR, f"seg_{image_name}")
    cv2.imwrite(output_path, blended_image)

    end_time = time.time()
    print(f"Processed {image_name} -> Saved to {output_path} (Time: {end_time - start_time:.2f}s)")

print("Inference complete. Segmented images saved in './output_predictions/segmentation_masks/'")
