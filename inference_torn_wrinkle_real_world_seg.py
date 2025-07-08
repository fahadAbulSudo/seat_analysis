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
from scipy.ndimage import distance_transform_edt
from detectron2.structures import Boxes

INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/test_new"
OUTPUT_DIR = "./output_predictions/segmentation_masks"
YOLO_MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/yolov8n-seg.pt"
MODEL_PATH_TORN = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn/dataset/output/model_final.pth"
MODEL_PATH_WRINKLE = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_wrinkle/best_model_wrinkle.pth"
TMP_EDGE_DIR = "./tmp"
PIXELS_PER_MM = 5
EDGE_OFFSET = 5

os.makedirs(TMP_EDGE_DIR, exist_ok=True)
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

def extract_edge_mask(isolated_image, cropped_mask, offset=5):
    # 1. Canny edge detection on isolated image
    gray = cv2.cvtColor(isolated_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    # 2. Find contours of the seat mask (binary)
    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Create mask for seat contour border (peripheral edge)
    seat_edge_mask = np.zeros_like(cropped_mask)
    cv2.drawContours(seat_edge_mask, contours, -1, 255, offset)

    # 4. Remove seat border edges from the Canny edges
    edges_filtered = cv2.bitwise_and(edges, cv2.bitwise_not(seat_edge_mask))

    # 5. Create overlay
    edge_overlay = isolated_image.copy()
    edge_overlay[edges_filtered > 0] = [0, 0, 255]  # Red overlay for remaining edges

    return edge_overlay, edges_filtered

def process_single_seat(image, mask, idx, image_path):
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary_mask = (mask_resized > 0.5).astype(np.uint8)

    x, y, w, h = cv2.boundingRect(binary_mask)
    cropped_image = image[y:y+h, x:x+w]
    cropped_mask = binary_mask[y:y+h, x:x+w]
    isolated = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)

    # Edge mask from isolated image (grayscale)
    gray_isolated = cv2.cvtColor(isolated, cv2.COLOR_BGR2GRAY)
    edge_mask = cv2.Canny(gray_isolated, 100, 200)
    edge_mask_dilated = cv2.dilate(edge_mask, np.ones((5, 5), np.uint8), iterations=1)
    edge_mask_dilated = (edge_mask_dilated > 0).astype(np.uint8)

    torn_result = predictor_torn(isolated)
    wrinkle_result = predictor_wrinkle(isolated)

    instances_with_flags = []

    for model_result, class_offset in [(torn_result, 0), (wrinkle_result, 1)]:
        instance = model_result["instances"].to("cpu")
        if len(instance) == 0:
            continue

        boxes = instance.pred_boxes.tensor.numpy().astype(int)
        masks = instance.pred_masks.numpy().astype(np.uint8)
        classes = instance.pred_classes.numpy() + class_offset

        aligned_masks = []
        aligned_boxes = []
        is_near_edge_flags = []

        for i in range(len(masks)):
            bbox = boxes[i]
            shifted_bbox = [bbox[0] + x, bbox[1] + y, bbox[2] + x, bbox[3] + y]
            aligned_boxes.append(shifted_bbox)

            # Create full-size mask for overlap checking
            full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            full_mask[y:y+h, x:x+w] = masks[i]
            aligned_masks.append(torch.from_numpy(full_mask))

            # Check if the mask (within cropped region) overlaps the dilated edge
            overlap = masks[i] & edge_mask_dilated
            is_near_edge = np.any(overlap)
            is_near_edge_flags.append(is_near_edge)

        # Attach updated predictions
        instance.pred_masks = torch.stack(aligned_masks)
        instance.pred_boxes.tensor = torch.tensor(aligned_boxes, dtype=torch.float32)
        instance.pred_classes = torch.tensor(classes)
        instances_with_flags.append(instance)
        instances_with_flags.append(is_near_edge_flags)
        # for i in range(len(instance)):
        #     single_instance = Instances(image.shape[:2])
        #     single_instance.pred_masks = instance.pred_masks[i].unsqueeze(0)
            
        #     # Fix: extract tensor and unsqueeze
        #     box_tensor = instance.pred_boxes.tensor[i].unsqueeze(0)
        #     single_instance.pred_boxes = Boxes(box_tensor)
            
        #     single_instance.pred_classes = instance.pred_classes[i].unsqueeze(0)
        #     instances_with_flags.append((single_instance, is_near_edge_flags[i]))
    return instances_with_flags

def draw_all(image, all_instances_with_flags):
    debug_info = {"empty_masks": []}

    # Step through the flat list in pairs
    for idx in range(0, len(all_instances_with_flags), 2):
        instance = all_instances_with_flags[idx]
        is_near_edge = all_instances_with_flags[idx + 1]

        cls = int(instance.pred_classes[0].item())
        label = thing_classes[cls]

        bbox = instance.pred_boxes[0].tensor.numpy().astype(int)[0]
        x1, y1, x2, y2 = bbox

        mask = instance.pred_masks[0].numpy().astype(np.uint8)

        # Check if the mask is all zeros
        if np.all(mask == 0):
            print(f"Warning: Mask for instance {idx//2} is all zeros!")
            debug_info["empty_masks"].append({
                "index": idx//2,
                "label": label,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "near_edge": bool(is_near_edge)
            })
            continue  # Skip drawing

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        width_mm = (x2 - x1) / PIXELS_PER_MM
        height_mm = (y2 - y1) / PIXELS_PER_MM
        extra = " (near edge)" if is_near_edge else ""
        label_text = f"{label}: {width_mm:.1f}mm x {height_mm:.1f}mm{extra}"

        # Fixed color: red for torn, green for wrinkle
        color = (0, 0, 255) if label == "torn" else (0, 255, 0)

        for cnt in contours:
            cv2.polylines(image, [cnt], isClosed=True, color=color, thickness=2)

        cv2.putText(
            image, label_text, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=color, thickness=2
        )

    print("Debug Info:", debug_info)
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
        all_instances_with_flags = []

        for idx, mask in enumerate(masks):
            instances = process_single_seat(image, mask, idx, image_path)
            all_instances_with_flags.extend(instances)

        if all_instances_with_flags:
            image = draw_all(image, all_instances_with_flags)

        cv2.imwrite(os.path.join(OUTPUT_DIR, file), image)
        print(f"Processed {file}")

process_images()
print("Inference complete.")
