import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances

INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/MSN 12723-20250526T042942Z-1-001"
OUTPUT_DIR = "./output_predictions/segmentation_masks"
YOLO_MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/yolov8n-seg.pt"
MODEL_PATH_TORN = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn/dataset/output/model_final.pth"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"
PIXELS_PER_MM = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

if "torn_wrinkle_dataset" not in MetadataCatalog.list():
    MetadataCatalog.get("torn_wrinkle_dataset").set(thing_classes=["torn", "wrinkle"])
metadata = MetadataCatalog.get("torn_wrinkle_dataset")
thing_classes = metadata.thing_classes

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

def process_single_seat(image, mask, idx, image_path):
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary_mask = (mask_resized > 0.5).astype(np.uint8)

    x, y, w, h = cv2.boundingRect(binary_mask)
    cropped_image = image[y:y+h, x:x+w]
    cropped_mask = binary_mask[y:y+h, x:x+w]
    isolated = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

    torn_result = predictor_torn(isolated)
    wrinkle_result = predictor_wrinkle(isolated)

    instances = []

    for model_result, class_offset in [(torn_result, 0), (wrinkle_result, 1)]:
        instance = model_result["instances"].to("cpu")
        if len(instance) == 0:
            continue

        boxes = instance.pred_boxes.tensor.numpy().astype(int)
        masks = instance.pred_masks.numpy().astype(np.uint8)
        classes = instance.pred_classes.numpy() + class_offset

        aligned_masks = []
        aligned_boxes = []

        for i in range(len(masks)):
            bbox = boxes[i]
            shifted_bbox = [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y]
            aligned_boxes.append(shifted_bbox)

            full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            full_mask[y:y+h, x:x+w] = masks[i]
            aligned_masks.append(torch.from_numpy(full_mask))

        instance.pred_masks = torch.stack(aligned_masks)
        instance.pred_boxes.tensor = torch.tensor(aligned_boxes, dtype=torch.float32)
        instance.pred_classes = torch.tensor(classes)  # ✅ Update class labels with offset

        instances.append(instance)

    return instances


def draw_all(image, all_instances):
    for instance in all_instances:
        for i in range(len(instance)):
            cls = int(instance.pred_classes[i].item())
            label = thing_classes[cls]
            bbox = instance.pred_boxes[i].tensor.numpy().astype(int)[0]
            x1, y1, x2, y2 = bbox
            mask = instance.pred_masks[i].numpy().astype(np.uint8)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            width_mm = (x2 - x1) / PIXELS_PER_MM
            height_mm = (y2 - y1) / PIXELS_PER_MM
            label_text = f"{label}: {width_mm:.1f}mm x {height_mm:.1f}mm"

            color = (0, 0, 255) if label == "torn" else (0, 255, 0)

            for cnt in contours:
                cv2.polylines(image, [cnt], isClosed=True, color=color, thickness=2)

            cv2.putText(
                image, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2
            )
    return image

def process_images():
    for file in os.listdir(INPUT_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        image_path = os.path.join(INPUT_DIR, file)
        image = cv2.imread(image_path)
        if image is None:
            continue

        yolo_results = yolo_model(image)[0]
        masks = yolo_results.masks.data.cpu().numpy()
        all_instances = []

        for idx, mask in enumerate(masks):
            instance_group = process_single_seat(image, mask, idx, image_path)
            all_instances.extend(instance_group)

        if all_instances:
            image = draw_all(image, all_instances)

        cv2.imwrite(os.path.join(OUTPUT_DIR, file), image)
        print(f"Processed {file}")

process_images()

print("Inference complete.")
