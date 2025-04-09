import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO  # YOLOv8
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import Instances
# Paths
INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/AIRBUS_TEST"
OUTPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/AIRBUS_Output_Segment"
YOLO_MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/yolov8n-seg.pt"
MODEL_PATH_TORN = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn/dataset/output/model_final.pth"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"

# Register metadata
if "torn_wrinkle_dataset" not in MetadataCatalog.list():
    MetadataCatalog.get("torn_wrinkle_dataset").set(thing_classes=["torn", "wrinkle"])
metadata = MetadataCatalog.get("torn_wrinkle_dataset")

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

def relabel_instances(instances, class_offset):
    if len(instances) > 0:
        instances.pred_classes += class_offset
    return instances

def offset_instances(instances, offset_x, offset_y, full_image_shape):
    instances = instances.to("cpu")
    instances.pred_boxes.tensor += torch.tensor([offset_x, offset_y, offset_x, offset_y])
    masks = instances.pred_masks.numpy()
    h, w = masks.shape[1:]
    padded_masks = np.zeros((len(masks), full_image_shape[0], full_image_shape[1]), dtype=np.uint8)
    for i, mask in enumerate(masks):
        padded_masks[i, offset_y:offset_y+h, offset_x:offset_x+w] = mask
    instances.pred_masks = torch.from_numpy(padded_masks)

    # ✅ Set full image size
    instances._image_size = full_image_shape[:2]
    return instances

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

            all_instances = []

            for mask in masks:
                binary_mask = cv2.resize((mask > 0.5).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                x, y, w, h = cv2.boundingRect(binary_mask)
                cropped_image = image[y:y+h, x:x+w]
                cropped_mask = binary_mask[y:y+h, x:x+w]
                masked_roi = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

                torn_outputs = predictor_torn(masked_roi)
                wrinkle_outputs = predictor_wrinkle(masked_roi)

                # Torn model (class 0 in thing_classes)
                offset_torn = offset_instances(torn_outputs["instances"], x, y, image.shape[:2])
                offset_torn = relabel_instances(offset_torn, 0)  # torn stays at class 0

                # Wrinkle model (should become class 1)
                offset_wrinkle = offset_instances(wrinkle_outputs["instances"], x, y, image.shape[:2])
                offset_wrinkle = relabel_instances(offset_wrinkle, 1)  # wrinkle becomes class 1

                # offset_torn = offset_instances(torn_outputs["instances"], x, y, image.shape[:2])
                # offset_wrinkle = offset_instances(wrinkle_outputs["instances"], x, y, image.shape[:2])

                if len(offset_torn) > 0:
                    all_instances.append(offset_torn)
                if len(offset_wrinkle) > 0:
                    all_instances.append(offset_wrinkle)

            if all_instances:
                combined_instances = Instances.cat(all_instances)

                v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
                output_image = v.draw_instance_predictions(combined_instances).get_image()[:, :, ::-1]
            else:
                output_image = image

            cv2.imwrite(output_path, output_image)
            print(f"Processed {image_path} -> Saved to {output_path}")

# Run processing
process_images(INPUT_DIR, OUTPUT_DIR)
print("Inference complete. Segmented images saved.")
