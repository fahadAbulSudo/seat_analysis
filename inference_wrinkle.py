import os
import cv2
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.model_zoo import get_config_file, get_checkpoint_url
import time
# Paths
MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"
IMAGE_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/test_image_no"
OUTPUT_DIR = "./output_predictions/wrinkle"

# Define your class labels
CLASS_NAMES = ["wrinkle"]  # Add more if needed

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Check if the model weights exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

# Register a dummy dataset (required for custom models)
def dummy_function():
    return []

DATASET_NAME = "custom_dataset"
if DATASET_NAME not in DatasetCatalog.list():
    DatasetCatalog.register(DATASET_NAME, dummy_function)
    MetadataCatalog.get(DATASET_NAME).set(thing_classes=CLASS_NAMES)

# Load model configuration
cfg = get_cfg()
cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))  # Using Model Zoo
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASS_NAMES)  # Dynamically set number of classes
cfg.MODEL.WEIGHTS = MODEL_PATH  # Load trained weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set confidence threshold
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

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

    # Run inference
    outputs = predictor(image)

    # Debugging: Print predicted class indices
    pred_classes = outputs["instances"].pred_classes.tolist()
    print(f"{image_name} -> Predicted class indices: {pred_classes}")

    # Visualize results
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(DATASET_NAME), scale=1.0)

    output_image = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    end_time = time.time()
    delta_time = end_time - start_time
    print(delta_time)
    # Save the image
    output_path = os.path.join(OUTPUT_DIR, f"pred_{image_name}")
    
    cv2.imwrite(output_path, output_image.get_image()[:, :, ::-1])

    print(f"Processed {image_name} -> Saved to {output_path}")
print("Inference complete. Segmented images saved in './output_predictions/'")
