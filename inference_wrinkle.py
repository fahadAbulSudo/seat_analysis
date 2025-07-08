import os
import cv2
import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances

# Paths
INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/output_predictions/cut/images"
OUTPUT_DIR = "./output_predictions/wrinkle_only_zoom"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/model_3rd/model_23rd.pth"

# Register metadata
if "torn_wrinkle_dataset" not in MetadataCatalog.list():
    MetadataCatalog.get("torn_wrinkle_dataset").set(thing_classes=["wrinkle"])
metadata = MetadataCatalog.get("torn_wrinkle_dataset")

# Load Mask R-CNN model
def load_model(model_path, class_names):
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

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
    instances._image_size = full_image_shape[:2]
    return instances

def zoom_image(image, zoom_factor=1.2):
    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2
    new_w, new_h = int(w / zoom_factor), int(h / zoom_factor)
    start_x = max(center_x - new_w // 2, 0)
    start_y = max(center_y - new_h // 2, 0)
    end_x = min(center_x + new_w // 2, w)
    end_y = min(center_y + new_h // 2, h)
    cropped = image[start_y:end_y, start_x:end_x]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return zoomed

def process_segmented_wrinkle_only(input_dir, output_dir, use_zoom=False):
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

            if use_zoom:
                image = zoom_image(image)

            wrinkle_outputs = predictor_wrinkle(image)
            instances = wrinkle_outputs["instances"]
            offset_wrinkle = offset_instances(instances, 0, 0, image.shape[:2])
            offset_wrinkle = relabel_instances(offset_wrinkle, 0)

            if len(offset_wrinkle) > 0:
                combined_instances = Instances.cat([offset_wrinkle])
                v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
                output_image = v.draw_instance_predictions(combined_instances).get_image()[:, :, ::-1]
            else:
                output_image = image

            cv2.imwrite(output_path, output_image)
            print(f"[WrinkleOnly] Processed {image_path} -> Saved to {output_path}")

# Run without zoom
process_segmented_wrinkle_only(INPUT_DIR, OUTPUT_DIR, use_zoom=True)

# Run with zoom augmentation (optional)
# process_segmented_wrinkle_only(INPUT_DIR, OUTPUT_DIR + "_zoomed", use_zoom=True)

print("Inference complete.")
