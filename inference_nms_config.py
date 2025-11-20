# -------------------- Imports --------------------
import os
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO  # YOLOv8 for seat detection
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
from detectron2.layers.nms import batched_nms

# -------------------- Apply NMS --------------------
def apply_nms(instances, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping predicted instances.
    Args:
        instances: Detectron2 Instances object
        iou_threshold: IOU threshold to suppress overlapping boxes
    Returns:
        Filtered Instances after NMS
    """
    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    classes = instances.pred_classes
    keep = batched_nms(boxes, scores, classes, iou_threshold)
    return instances[keep]

# -------------------- Load Config --------------------
with open("config.json", "r") as f:
    config = json.load(f)

INPUT_DIR = config["INPUT_DIR"]
OUTPUT_DIR = config["OUTPUT_DIR"]
YOLO_MODEL_PATH = config["YOLO_MODEL_PATH"]
MODEL_PATH_TORN = config["MODEL_PATH_TORN"]
MODEL_PATH_WRINKLE = config["MODEL_PATH_WRINKLE"]
# Register custom dataset metadata if not already present
if "torn_wrinkle_dataset" not in MetadataCatalog.list():
    MetadataCatalog.get("torn_wrinkle_dataset").set(thing_classes=["torn", "wrinkle"])
metadata = MetadataCatalog.get("torn_wrinkle_dataset")

# -------------------- Model Loaders --------------------
# Load YOLOv8 seat detector
yolo_model = YOLO(YOLO_MODEL_PATH)

def load_model(model_path, class_names):
    """
    Load a Detectron2 Mask R-CNN model with given weights and class names.
    """
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

# Load torn and wrinkle segmentation models
predictor_torn = load_model(MODEL_PATH_TORN, ["torn"])
predictor_wrinkle = load_model(MODEL_PATH_WRINKLE, ["wrinkle"])

# -------------------- Utilities --------------------
def relabel_instances(instances, class_offset):
    """
    Adjust class labels by offset (useful when merging predictions from different models).
    """
    if len(instances) > 0:
        instances.pred_classes += class_offset
    return instances

def offset_instances(instances, offset_x, offset_y, full_image_shape):
    """
    Shift instance predictions (boxes and masks) to their correct position in the full image.
    Args:
        instances: Detected objects in a cropped region
        offset_x/y: Pixel offset from full image
        full_image_shape: Shape of the full image
    Returns:
        Instances with corrected spatial coordinates
    """
    instances = instances.to("cpu")
    instances.pred_boxes.tensor += torch.tensor([offset_x, offset_y, offset_x, offset_y])
    masks = instances.pred_masks.numpy()
    h, w = masks.shape[1:]
    padded_masks = np.zeros((len(masks), full_image_shape[0], full_image_shape[1]), dtype=np.uint8)
    for i, mask in enumerate(masks):
        padded_masks[i, offset_y:offset_y+h, offset_x:offset_x+w] = mask
    instances.pred_masks = torch.from_numpy(padded_masks)
    instances._image_size = full_image_shape[:2]
    return instances

def rotate_image(image, direction):
    """Rotate image based on direction ('cw' or 'ccw')."""
    if direction == "cw":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif direction == "ccw":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

# -------------------- Image Processor --------------------
def process_images(input_dir, output_dir):
    """
    Main processing loop:
    - Detect seats using YOLO
    - For each seat region, run wrinkle segmentation using Detectron2
    - Overlay masks and bounding boxes on image
    - Save output visualizations
    """
    for root, _, files in os.walk(input_dir):
        # image_files = sorted(
        #     [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        # )

        # # Skip if not exactly 8 images
        # if len(image_files) != 8:
        #     continue

        count = 0
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

            # Decide rotation direction
            # if count < 4:
            #     image = rotate_image(image, "ccw")   # Clockwise
            # else:
            #     image = rotate_image(image, "cw")  # Counterclockwise

            # count += 1

            yolo_results = yolo_model(image)[0]  # Seat detection
            try:
                masks = yolo_results.masks.data.cpu().numpy()
                height, width = image.shape[:2]
                all_instances = []

                boxes_data = yolo_results.boxes
                class_ids = boxes_data.cls.cpu().numpy().astype(int)  # YOLO class IDs
                id_to_name = yolo_model.names

                for idx, mask in enumerate(masks):
                    cls_name = id_to_name[class_ids[idx]].lower()

                    # Skip if not seat
                    if cls_name != "seat":
                        print(f"Skipping mask {idx} ({cls_name}) in {file}")
                        continue

                    # Post-process YOLO mask
                    binary_mask = cv2.resize((mask > 0.3).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                    x, y, w, h = cv2.boundingRect(binary_mask)
                    cropped_image = image[y:y+h, x:x+w]
                    cropped_mask = binary_mask[y:y+h, x:x+w]
                    masked_roi = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

                    # Run wrinkle prediction on seat region
                    wrinkle_outputs = predictor_wrinkle(masked_roi)
                    offset_wrinkle = offset_instances(wrinkle_outputs["instances"], x, y, image.shape[:2])
                    offset_wrinkle = relabel_instances(offset_wrinkle, 1)
                    if len(offset_wrinkle) > 0:
                        all_instances.append(offset_wrinkle)

                # Visualize and save
                if all_instances:
                    combined_instances = Instances.cat(all_instances)
                    combined_instances = apply_nms(combined_instances, iou_threshold=0.3)

                    output_image = image.copy()
                    print(combined_instances)
                    for i in range(len(combined_instances)):
                        mask = combined_instances.pred_masks[i].numpy().astype(np.uint8)
                        score = combined_instances.scores[i].item() * 100

                        # Color and transparency by confidence
                        if 30 <= score < 50:
                            color = (0, 255, 255)  # Yellow
                            alpha = 0.2
                        elif 50 <= score < 70:
                            color = (0, 165, 255)  # Orange
                            alpha = 0.2
                        elif score >= 70:
                            color = (0, 0, 255)    # Red
                            alpha = 0.2
                        else:
                            continue

                        # Overlay mask with transparency
                        for c in range(3):
                            output_image[:, :, c] = np.where(
                                mask == 1,
                                (1 - alpha) * output_image[:, :, c] + alpha * color[c],
                                output_image[:, :, c]
                            ).astype(np.uint8)

                        # Draw bounding box and label
                        x1, y1, x2, y2 = combined_instances.pred_boxes.tensor[i].int().tolist()
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                        label = f"{metadata.thing_classes[combined_instances.pred_classes[i]]}:{int(score)}%"
                        cv2.putText(output_image, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    output_image = image  # No predictions

                cv2.imwrite(output_path, output_image)
                print(f"Processed {image_path} -> Saved to {output_path}")

            except AttributeError as e:
                print(f"Error accessing YOLO results masks: {e}")

# -------------------- Run --------------------
process_images(INPUT_DIR, OUTPUT_DIR)
print("Inference complete. Segmented images saved.")
