# -------------------- Imports --------------------
import os
import cv2
import json
import torch
import numpy as np
from ultralytics import YOLO
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
from detectron2.layers.nms import batched_nms

# -------------------- Apply NMS --------------------
def apply_nms(instances, iou_threshold=0.5):
    boxes = instances.pred_boxes.tensor
    scores = instances.scores
    classes = instances.pred_classes
    keep = batched_nms(boxes, scores, classes, iou_threshold)
    return instances[keep]


# -------------------- Config --------------------
with open("config.json", "r") as f:
    config = json.load(f)

INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/MSN_12918_Left-20250901T124444Z-1-001/MSN_12918_Left/seat_1/raw_images/test"
OUTPUT_DIR = os.path.join(INPUT_DIR, "visualized")
YOLO_MODEL_PATH = config["YOLO_MODEL_PATH"]
MODEL_PATH_WRINKLE = config["MODEL_PATH_WRINKLE"]
TEMP_DIR = os.path.join(OUTPUT_DIR, "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

# Metadata
if "torn_wrinkle_dataset" not in MetadataCatalog.list():
    MetadataCatalog.get("torn_wrinkle_dataset").set(thing_classes=["torn", "wrinkle"])
metadata = MetadataCatalog.get("torn_wrinkle_dataset")

# -------------------- Load Models --------------------
yolo_model = YOLO(YOLO_MODEL_PATH)

def load_model(model_path, class_names):
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

predictor_wrinkle = load_model(MODEL_PATH_WRINKLE, ["wrinkle"])


# -------------------- Helpers --------------------
def normalize_wrinkle(mask, bbox, image_shape):
    """
    Normalize wrinkle coordinates relative to the full original image space.
    """
    x1, y1, x2, y2 = bbox
    H, W = image_shape[:2]  # full original image size

    coords = np.column_stack(np.where(mask > 0))  # (y, x) in cropped ROI
    norm_coords = [((x + x1) / W, (y + y1) / H) for (y, x) in coords]  # offset + normalize globally

    return norm_coords

def project_wrinkle(norm_coords, seat_box):
    x1, y1, x2, y2 = seat_box
    return [(int(x1 + nx * (x2 - x1)), int(y1 + ny * (y2 - y1))) for nx, ny in norm_coords]

def translate_wrinkles_x_left(wrinkle_coords, bbox_src, bbox_dst):
    """
    Maps wrinkle coordinates from source seat to destination seat along X-axis from left.
    Also adds the Y offset (y1_s) from the source seat.
    """
    x1_s, y1_s, x2_s, y2_s = bbox_src
    x1_d, y1_d, x2_d, y2_d = bbox_dst

    translated = []
    for (x, y) in wrinkle_coords:
        # Compute horizontal translation relative to right edge of source
        dx = x - x2_s  # distance from right edge of source seat
        new_x = x2_d + dx  # shift same distance from dest right edge

        # Adjust vertical position with source offset (to global Y)
        new_y = y + y1_s

        translated.append((new_x, new_y))

    return translated

def get_seat_roles(yolo_results, yolo_model, image_name, idx):
    seat_boxes = []

    # --- Extract YOLO segmentation results ---
    if not hasattr(yolo_results, "masks") or yolo_results.masks is None:
        print(f"❌ No masks found for {image_name}")
        return {image_name: {}}

    masks = yolo_results.masks.data.cpu().numpy()
    class_ids = yolo_results.boxes.cls.cpu().numpy()
    orig_img = yolo_results.orig_img  # original image from YOLO result

    print(f"🧩 Found {len(masks)} masks in {image_name}")

    # --- Iterate through each detected object ---
    for i, (mask, cls_id) in enumerate(zip(masks, class_ids)):
        class_name = yolo_model.names[int(cls_id)].lower()

        # Only process seats
        if class_name == "seat":
            ys, xs = np.where(mask > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue

            # Compute bounding box from mask
            x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
            cx = (x1 + x2) / 2

            seat_boxes.append({
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "mask": mask,
                "cx": cx
            })

    # --- Sort seats left→right ---
    seat_boxes = sorted(seat_boxes, key=lambda b: b["cx"])
    base_roles = ["left", "middle", "right"][:len(seat_boxes)]

    # --- Assign roles depending on image index ---
    if idx == 0:
        roles = ["middle", "right"][:len(seat_boxes)]
    elif idx == 1:
        roles = base_roles
    elif idx == 2:
        roles = ["left", "middle"][:len(seat_boxes)]
    else:
        roles = base_roles

    # --- Build final seat-role mapping ---
    result = {
        image_name: {
            roles[i]: {
                "bbox": seat_boxes[i]["bbox"],
                "mask": seat_boxes[i]["mask"],
            }
            for i in range(len(roles))
        }
    }

    # # --- ✅ DEBUG VISUALIZATION (save combined segmentation) ---
    # if len(seat_boxes) > 0:
    #     vis_image = orig_img.copy()
    #     h, w = vis_image.shape[:2]

    #     for i, seat in enumerate(seat_boxes):
    #         # Resize YOLO mask (which is in 640x480) to match original image size
    #         mask_resized = cv2.resize(
    #             seat["mask"].astype(np.uint8),
    #             (w, h),
    #             interpolation=cv2.INTER_NEAREST
    #         )

    #         # Create a color mask overlay
    #         color = (0, 255, 0) if i == 0 else (255, 0, 0) if i == 1 else (0, 0, 255)
    #         color_mask = np.zeros_like(vis_image, dtype=np.uint8)
    #         for c in range(3):
    #             color_mask[:, :, c] = mask_resized * color[c]

    #         # Blend mask with image
    #         blended = cv2.addWeighted(vis_image, 0.7, color_mask, 0.3, 0)

    #         # Draw bbox + label
    #         x1, y1, x2, y2 = seat["bbox"]
    #         cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)
    #         cv2.putText(
    #             blended,
    #             roles[i],
    #             (x1, max(0, y1 - 10)),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             0.7,
    #             color,
    #             2
    #         )
    #         vis_image = blended

    #     base_name = os.path.splitext(os.path.basename(image_name))[0]
    #     save_path = os.path.join(TEMP_DIR, f"{base_name}_segmentation_debug.png")
    #     cv2.imwrite(save_path, vis_image)
    #     print(f"✅ Saved segmentation visualization → {save_path}")

    return result

# -------------------- Visualizer --------------------
def draw_wrinkles_on_image(image, wrinkles, color=(0, 0, 255)):
    """
    Overlay wrinkle coordinates as a filled mask (red transparent area).
    """
    overlay = image.copy()
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for wrinkle_set in wrinkles:
        wrinkle_np = np.array(wrinkle_set, dtype=np.int32)
        for (x, y) in wrinkle_np:
            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                mask[y, x] = 255

    # Apply transparency
    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = color
    alpha = 0.4
    overlay = np.where(mask[..., None] == 255,
                       (1 - alpha) * overlay + alpha * colored_mask,
                       overlay)
    return overlay.astype(np.uint8)

def relabel_instances(instances, class_offset):
    """
    Adjust class labels by offset (useful when merging predictions from different models).
    """
    if len(instances) > 0:
        instances.pred_classes += class_offset
    return instances

def offset_wrinkle_coords(wrinkle_set, bbox):
    """
    Convert wrinkle coordinates from seat-local space to full image space.
    bbox = [x1, y1, x2, y2]
    wrinkle_set = [[x, y], [x2, y2], ...]
    """
    x_offset, y_offset = bbox[0], bbox[1]
    return [[x + x_offset, y + y_offset] for [x, y] in wrinkle_set]

def offset_wrinkle_mixed_coords(wrinkle_set, bbox_src, bbox_dst):
    """
    Offset wrinkle coordinates using mixed references:
    - X coordinates are adjusted using destination bbox (bbox_dst)
    - Y coordinates are adjusted using source bbox (bbox_src)

    Args:
        wrinkle_set (list[list[int]]): Wrinkle coordinates in seat-local space
        bbox_src (list[int]): Source seat bounding box [x1, y1, x2, y2]
        bbox_dst (list[int]): Destination seat bounding box [x1, y1, x2, y2]
    
    Returns:
        list[list[int]]: Translated wrinkle coordinates in global image space
    """
    x_offset_dst = bbox_dst[0]  # X shift comes from destination seat
    y_offset_src = bbox_src[1]  # Y shift comes from source seat

    return [[x + x_offset_dst, y + y_offset_src] for [x, y] in wrinkle_set]

# -------------------- Main Multi-view Processor --------------------
def process_three_images(img_paths):
    seat_data = {}
    yolo_results_list = []

    # Run YOLO on all images first
    for idx, img_path in enumerate(img_paths):
        image_name = os.path.basename(img_path)
        image = cv2.imread(img_path)
        yolo_results = yolo_model(image)[0]
        yolo_results_list.append((image_name, image, yolo_results))

    # Detect wrinkles per seat
    for idx, (image_name, image, yolo_results) in enumerate(yolo_results_list):
        height, width = image.shape[:2]
        seat_roles = get_seat_roles(yolo_results, yolo_model, image_name, idx)
        # print("seat_roles",seat_roles)
        seat_data[image_name] = {}
        # print("seat_data",seat_data)

        for role, seat_info in seat_roles[image_name].items():
            print(image_name)
            print(role)
            seat_box, seat_mask = seat_info["bbox"], seat_info["mask"]
            x1, y1, x2, y2 = seat_box
            # --- Get original image shape ---
            height, width = image.shape[:2]

            # --- Get YOLO mask shape (usually 640x480 or 640x640) ---
            mask_h, mask_w = seat_mask.shape[:2]

            # --- Scale bbox from YOLO mask → original image dimensions ---
            scale_x = width / mask_w
            scale_y = height / mask_h

            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)
            seat_box = (x1, y1, x2, y2)
            # base_name = os.path.splitext(os.path.basename(image_name))[0]
            # save_path = os.path.join(TEMP_DIR, f"{base_name}_{role}_roi.png")
            # cv2.imwrite(save_path, seat_mask)
            # print(f"✅ Saved masked ROI → {save_path}")

            height, width = image.shape[:2]
            # print(image)    
            binary_mask = cv2.resize((seat_mask > 0.3).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
            # seat_mask = cv2.resize(seat_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            # binary_mask = (seat_mask[y1:y2, x1:x2] > 0.3).astype(np.uint8)
            cropped_image = image[y1:y2, x1:x2]
            cropped_mask = binary_mask[y1:y2, x1:x2]
            masked_roi = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)
            # construct save path inside OUTPUT_DIR/temp
            base_name = os.path.splitext(os.path.basename(image_name))[0]
            save_path = os.path.join(TEMP_DIR, f"{base_name}_{role}_roi.png")

            # # save the masked ROI
            # cv2.imwrite(save_path, masked_roi)
            # print(f"Saved masked ROI → {save_path}")
            wrinkle_outputs = predictor_wrinkle(masked_roi)
            instances = wrinkle_outputs["instances"].to("cpu")
 
            wrinkles = []
            for mask in instances.pred_masks.numpy():
                # norm_coords = normalize_wrinkle(mask, (x1, y1, x2, y2), image.shape)
                y_indices, x_indices = np.where(mask > 0)
                # Stack them into coordinate pairs (x, y)
                coords = np.column_stack((x_indices, y_indices))
                # Store as list of [x, y] pairs
                wrinkles.append(coords.tolist())

            seat_data[image_name][role] = {
                "bbox": seat_box,
                "wrinkles": wrinkles
            }
            debug_full = image.copy()
            # for coords in wrinkles:
            #     if len(coords) == 0:
            #         continue
            #     pts = np.array(coords, dtype=np.int32)
            #     # Offset wrinkle coordinates to global coordinates
            #     pts[:, 0] += x1
            #     pts[:, 1] += y1
            #     color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
            #     cv2.polylines(debug_full, [pts], isClosed=False, color=color, thickness=2)

            # # Save full image visualization
            # debug_full_path = os.path.join(TEMP_DIR, f"{base_name}_{role}_wrinkle_full_debug.png")
            # cv2.imwrite(debug_full_path, debug_full)
            # print(f"🌍 Full-image wrinkle overlay saved → {debug_full_path}")

    # -------------------- Integrate across views --------------------
    count = 0
    integrated_data = {}

    for i, (image_name, _, _) in enumerate(yolo_results_list):
        main_roles = seat_data[image_name]

        # Load original image for debugging visualization
        img_path = [p for p in img_paths if os.path.basename(p) == image_name][0]
        image = cv2.imread(img_path)
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        if "middle" in main_roles:
            bbox = main_roles["middle"]["bbox"]
            abs_wrinkles = []
            for wrinkle_set in main_roles["middle"]["wrinkles"]:
                abs_wrinkles.append(offset_wrinkle_coords(wrinkle_set, bbox))
            main_roles["middle"]["wrinkles"] = abs_wrinkles
        if i == 0:
            center_roles = seat_data[os.path.basename(img_paths[1])]
            if "middle" in main_roles and "left" in center_roles:
                src_bbox = center_roles["left"]["bbox"]
                dst_bbox = main_roles["middle"]["bbox"]

                for idx, wrinkle_set in enumerate(center_roles["left"]["wrinkles"]):
                    count += 1
                    translated = translate_wrinkles_x_left(wrinkle_set, src_bbox, dst_bbox)
                    main_roles["middle"]["wrinkles"].append(translated)

                    # ---- DEBUG DRAW for this wrinkle ----
                    debug_img = image.copy()
                    x1, y1, x2, y2 = map(int, dst_bbox)
                    pts = np.array(translated, dtype=np.int32)
                    color = (0, 255, 0)
                    cv2.polylines(debug_img, [pts], isClosed=False, color=color, thickness=2)
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

                    save_debug_path = os.path.join(TEMP_DIR, f"{base_name}_L_wrinkle_{idx}_{count}.png")
                    cv2.imwrite(save_debug_path, debug_img)
                    print(f"🟢 Saved Left→Middle wrinkle debug: {save_debug_path}")

        elif i == 1:
            # CENTER IMAGE → integrate wrinkles from both LEFT and RIGHT images
            left_roles = seat_data[os.path.basename(img_paths[0])]
            right_roles = seat_data[os.path.basename(img_paths[2])]

            if "middle" in main_roles:
                dst_bbox = main_roles["middle"]["bbox"]

                # Map right seat of left image → middle seat
                if "right" in left_roles:
                    for idx, wrinkle_set in enumerate(left_roles["right"]["wrinkles"]):
                        count += 1
                        src_bbox = right_roles["left"]["bbox"]
                        translated_offset = offset_wrinkle_mixed_coords(wrinkle_set, src_bbox, dst_bbox)
                        main_roles["middle"]["wrinkles"].append(translated_offset)

                        # ---- DEBUG DRAW for this wrinkle ----
                        debug_img = image.copy()
                        x1, y1, x2, y2 = map(int, dst_bbox)
                        pts = np.array(translated_offset, dtype=np.int32)
                        # pts[:, 0] += x1
                        # pts[:, 1] += y1
                        color = (255, 0, 0)
                        cv2.polylines(debug_img, [pts], isClosed=False, color=color, thickness=2)
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

                        save_debug_path = os.path.join(TEMP_DIR, f"{base_name}_C_fromL_{idx}_{count}.png")
                        cv2.imwrite(save_debug_path, debug_img)
                        print(f"🔵 Saved Left→Center wrinkle debug: {save_debug_path}")

                # Map left seat of right image → middle seat
                if "left" in right_roles:
                    src_bbox = right_roles["left"]["bbox"]
                    for idx, wrinkle_set in enumerate(right_roles["left"]["wrinkles"]):
                        count += 1
                        translated = translate_wrinkles_x_left(wrinkle_set, src_bbox, dst_bbox)
                        main_roles["middle"]["wrinkles"].append(translated)

                        # ---- DEBUG DRAW for this wrinkle ----
                        debug_img = image.copy()
                        x1, y1, x2, y2 = map(int, dst_bbox)
                        pts = np.array(translated_offset, dtype=np.int32)
                        color = (0, 0, 255)
                        cv2.polylines(debug_img, [pts], isClosed=False, color=color, thickness=2)
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

                        save_debug_path = os.path.join(TEMP_DIR, f"{base_name}_C_fromR_{idx}_{count}.png")
                        cv2.imwrite(save_debug_path, debug_img)
                        print(f"🔴 Saved Right→Center wrinkle debug: {save_debug_path}")

        elif i == 2:
            center_roles = seat_data[os.path.basename(img_paths[1])]
            if "middle" in main_roles and "right" in center_roles:
                dst_bbox = main_roles["middle"]["bbox"]
                for idx, wrinkle_set in enumerate(center_roles["right"]["wrinkles"]):
                    count += 1
                    src_bbox = right_roles["left"]["bbox"]
                    translated_offset = offset_wrinkle_mixed_coords(wrinkle_set, src_bbox, dst_bbox)
                    main_roles["middle"]["wrinkles"].append(translated_offset)

                    # ---- DEBUG DRAW for this wrinkle ----
                    debug_img = image.copy()
                    x1, y1, x2, y2 = map(int, dst_bbox)
                    pts = np.array(translated_offset, dtype=np.int32)
                    color = (255, 255, 0)
                    cv2.polylines(debug_img, [pts], isClosed=False, color=color, thickness=2)
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 255, 0), 2)

                    save_debug_path = os.path.join(TEMP_DIR, f"{base_name}_R_wrinkle_{idx}_{count}.png")
                    cv2.imwrite(save_debug_path, debug_img)
                    print(f"🟡 Saved Center→Right wrinkle debug: {save_debug_path}")

        # -------- After all translations → final middle-seat visualization --------
        if "middle" in main_roles:
            wrinkles = main_roles["middle"]["wrinkles"]
            x1, y1, x2, y2 = map(int, main_roles["middle"]["bbox"])
            debug_full = image.copy()

            for coords in wrinkles:
                if len(coords) == 0:
                    continue
                pts = np.array(coords, dtype=np.int32)
                # pts[:, 0] += x1
                # pts[:, 1] += y1
                color = tuple(int(x) for x in np.random.randint(0, 255, size=3))
                cv2.polylines(debug_full, [pts], isClosed=False, color=color, thickness=2)

            cv2.rectangle(debug_full, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(debug_full, "middle", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            debug_full_path = os.path.join(TEMP_DIR, f"{base_name}_middle_final_debug.png")
            cv2.imwrite(debug_full_path, debug_full)
            print(f"🌍 Final middle-seat wrinkle overlay saved → {debug_full_path}")

        # ---- Store final integrated data ----
        if "middle" in main_roles:
            integrated_data[image_name] = {
                "bbox": main_roles["middle"]["bbox"],
                "wrinkles": main_roles["middle"]["wrinkles"]
            }
        else:
            integrated_data[image_name] = {}


        # -------------------- Store back integrated data --------------------
        # if "middle" in main_roles:
        #     integrated_data[image_name] = {
        #         "bbox": main_roles["middle"]["bbox"],
        #         "wrinkles": main_roles["middle"]["wrinkles"]
        #     }
        # else:
        #     integrated_data[image_name] = {}

    for image_name, data in integrated_data.items():
        if not data:  # skip if empty
            continue

        img_path = [p for p in img_paths if os.path.basename(p) == image_name][0]
        image = cv2.imread(img_path)

        wrinkles = data["wrinkles"]
        bbox = data["bbox"]

        if len(wrinkles) > 0:
            image = draw_wrinkles_on_image(image, wrinkles, color=(0, 0, 255))
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(image, "middle", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        save_name = os.path.splitext(image_name)[0] + "_wrinkle_vis.png"
        save_path = os.path.join(OUTPUT_DIR, save_name)
        cv2.imwrite(save_path, image)
        print(f"✅ Saved wrinkle visualization: {save_path}")


# -------------------- Run Example --------------------
if __name__ == "__main__":
    example_paths = [
        os.path.join(INPUT_DIR, "left.jpg"),
        os.path.join(INPUT_DIR, "center.jpg"),
        os.path.join(INPUT_DIR, "right.jpg"),
    ]
    process_three_images(example_paths)
    print("✅ Visualization complete.")
