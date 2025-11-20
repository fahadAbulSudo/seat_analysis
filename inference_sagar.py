
import os
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import get_config_file
from detectron2.data import MetadataCatalog
from detectron2.structures import Instances
from detectron2.layers.nms import batched_nms

# -------------------- Config load --------------------
with open("inference/config.json", "r") as f:
    config = json.load(f)

INPUT_DIR = config["INPUT_DIR"]
OUTPUT_DIR = config["OUTPUT_DIR"]
YOLO_MODEL_PATH = config["YOLO_MODEL_PATH"]
MODEL_PATH_WRINKLE = config.get("MODEL_PATH_WRINKLE")
MODEL_PATH_TORN = config.get("MODEL_PATH_TORN")  # optional

TEMP_SUBDIR_NAME = "temp"

# ensure metadata
if "torn_wrinkle_dataset" not in MetadataCatalog.list():
    MetadataCatalog.get("torn_wrinkle_dataset").set(thing_classes=["torn", "wrinkle"])
metadata = MetadataCatalog.get("torn_wrinkle_dataset")

# -------------------- Load models --------------------
print("[INFO] Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

def load_detectron2_model(model_path, class_names):
    cfg = get_cfg()
    cfg.merge_from_file(get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

predictor_wrinkle = load_detectron2_model(MODEL_PATH_WRINKLE, ["wrinkle"])
predictor_torn = None
if MODEL_PATH_TORN:
    predictor_torn = load_detectron2_model(MODEL_PATH_TORN, ["torn"])

# -------------------- Utilities --------------------
def apply_nms(instances: Instances, iou_threshold=0.5):
    """
    Apply NMS using batched_nms on a Detectron2 Instances object.
    Returns filtered Instances.
    """
    if len(instances) == 0:
        return instances
    boxes = instances.pred_boxes.tensor  # Nx4 tensor
    scores = instances.scores
    classes = instances.pred_classes
    keep = batched_nms(boxes, scores, classes, iou_threshold)
    return instances[keep]

def relabel_instances(instances: Instances, class_offset: int):
    if len(instances) > 0:
        instances.pred_classes = instances.pred_classes + class_offset
    return instances

def offset_instances(instances: Instances, offset_x: int, offset_y: int, full_image_shape):
    """
    Shift boxes and expand masks from an roi-relative Instances object to full-image coordinates.
    Instances should be on CPU.
    """
    if len(instances) == 0:
        return instances
    instances = instances.to("cpu")
    # shift boxes
    instances.pred_boxes.tensor = instances.pred_boxes.tensor + torch.tensor([offset_x, offset_y, offset_x, offset_y])
    # pad masks back to full image
    masks_np = instances.pred_masks.numpy()  # (N, h_roi, w_roi)
    n, h_roi, w_roi = masks_np.shape
    H, W = full_image_shape[:2]
    padded = np.zeros((n, H, W), dtype=np.uint8)
    for i in range(n):
        y1 = offset_y
        y2 = offset_y + h_roi
        x1 = offset_x
        x2 = offset_x + w_roi
        # clip if necessary
        y1_cl, y2_cl = max(0, y1), min(H, y2)
        x1_cl, x2_cl = max(0, x1), min(W, x2)
        src_y1 = max(0, -y1)
        src_x1 = max(0, -x1)
        src_y2 = src_y1 + (y2_cl - y1_cl)
        src_x2 = src_x1 + (x2_cl - x1_cl)
        if (y2_cl - y1_cl) > 0 and (x2_cl - x1_cl) > 0:
            padded[i, y1_cl:y2_cl, x1_cl:x2_cl] = masks_np[i, src_y1:src_y2, src_x1:src_x2]
    instances.pred_masks = torch.from_numpy(padded.astype(np.uint8))
    instances._image_size = (H, W)
    return instances

def map_points_proportional_global(points_global, bbox_src, bbox_dst):
    """
    Map list of [x,y] points from bbox_src -> proportionally inside bbox_dst.
    Uses pixel coordinates (absolute).
    """
    x1_s, y1_s, x2_s, y2_s = bbox_src
    x1_d, y1_d, x2_d, y2_d = bbox_dst
    w_s = max(1, x2_s - x1_s)
    h_s = max(1, y2_s - y1_s)
    w_d = max(1, x2_d - x1_d)
    h_d = max(1, y2_d - y1_d)
    mapped = []
    for (x,y) in points_global:
        nx = (x - x1_s) / float(w_s)
        ny = (y - y1_s) / float(h_s)
        new_x = int(round(x1_d + nx * w_d))
        new_y = int(round(y1_d + ny * h_d))
        mapped.append([new_x, new_y])
    return mapped

def draw_masks_and_boxes_on_image(image, masks_list, boxes_list, scores_list=None, classes_list=None):
    out = image.copy()
    H, W = image.shape[:2]
    for idx, mask in enumerate(masks_list):
        if mask.sum() == 0:
            continue
        score = None if scores_list is None else scores_list[idx]
        # color by score buckets (same scheme as you had)
        if score is None:
            color = (0, 0, 255); alpha = 0.2
        else:
            sc = float(score) * 100.0
            if 30 <= sc < 50:
                color = (0, 255, 255); alpha = 0.2
            elif 50 <= sc < 70:
                color = (0, 165, 255); alpha = 0.2
            elif sc >= 70:
                color = (0, 0, 255); alpha = 0.2
            else:
                continue
        for c in range(3):
            out[:, :, c] = np.where(mask == 1,
                                    (1 - alpha) * out[:, :, c] + alpha * color[c],
                                    out[:, :, c]).astype(np.uint8)
        if boxes_list is not None:
            x1, y1, x2, y2 = boxes_list[idx]
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = "wrinkle"
            if scores_list is not None:
                label += f":{int(float(scores_list[idx]) * 100)}%"
            cv2.putText(out, label, (int(x1), max(0, int(y1) - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out

def draw_wrinkle_points_overlay(image, wrinkles, color=(0,0,255), alpha=0.4):
    overlay = image.copy().astype(np.float32)
    H, W = image.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)
    for coords in wrinkles:
        if not coords:
            continue
        pts = np.array(coords, dtype=np.int32)
        for (x,y) in pts:
            if 0 <= x < W and 0 <= y < H:
                mask[y,x] = 255
    colored_mask = np.zeros_like(image, dtype=np.float32)
    colored_mask[:] = color
    overlay = np.where(mask[..., None] == 255, (1 - alpha) * overlay + alpha * colored_mask, overlay)
    return overlay.astype(np.uint8)

def poly_from_points(points):
    return np.array(points, dtype=np.int32)

# -------------------- Seat detection & wrinkle extraction per image --------------------
def detect_seats_and_wrinkles(image, image_name, idx):
    """
    Runs YOLO on the image, extracts seat bbox masks, runs wrinkle predictor per seat,
    returns dict: role -> {"bbox":(x1,y1,x2,y2), "wrinkles":[list of [x,y] global coords], "instances":Instances}
    """
    res = yolo_model(image)[0]
    if not hasattr(res, "masks") or res.masks is None:
        print(f"[WARN] No YOLO masks for {image_name}")
        return {}

    masks = res.masks.data.cpu().numpy()   # (N, mask_h, mask_w)
    class_ids = res.boxes.cls.cpu().numpy().astype(int)
    id_to_name = yolo_model.names
    H, W = image.shape[:2]

    seat_boxes = []
    for i, m in enumerate(masks):
        cls_name = id_to_name[int(class_ids[i])].lower()
        if cls_name != "seat":
            continue
        ys, xs = np.where(m > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x1, y1, x2, y2 = int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys))
        # convert to full image pixel coords by resizing mask to full size then boundingRect
        mask_full = cv2.resize((m > 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        bx, by, bw, bh = cv2.boundingRect(mask_full)
        if bw == 0 or bh == 0:
            continue
        cx = bx + bw/2.0
        seat_boxes.append({"bbox": (bx, by, bx + bw, by + bh), "mask_full": mask_full, "cx": cx})

    if len(seat_boxes) == 0:
        return {}

    # sort by center x (left -> right)
    seat_boxes = sorted(seat_boxes, key=lambda s: s["cx"])
    base_roles = ["left", "middle", "right"][:len(seat_boxes)]
    # role assignment similar to your pattern
    if idx == 0:
        roles = ["middle", "right"][:len(seat_boxes)]
    elif idx == 1:
        roles = base_roles
    elif idx == 2:
        roles = ["left", "middle"][:len(seat_boxes)]
    else:
        roles = base_roles

    result = {}
    for i, seat in enumerate(seat_boxes):
        if i >= len(roles):
            break
        role = roles[i]
        x1, y1, x2, y2 = map(int, seat["bbox"])
        # crop masked ROI
        cropped_mask = seat["mask_full"][y1:y2, x1:x2]
        cropped_image = image[y1:y2, x1:x2].copy()
        if cropped_image.size == 0:
            result[role] = {"bbox": (x1,y1,x2,y2), "wrinkles": [], "instances": None}
            continue
        masked_roi = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)
        # run wrinkle predictor
        try:
            outputs = predictor_wrinkle(masked_roi)
            instances = outputs["instances"].to("cpu")
            global_wrinkles = []
            # convert mask -> list of [x,y] in global coords
            if len(instances) > 0:
                masks_np = instances.pred_masks.numpy()
                for mask_local in masks_np:
                    ys, xs = np.where(mask_local > 0)
                    if len(xs) == 0:
                        continue
                    coords = np.column_stack((xs, ys))
                    coords_global = [[int(x + x1), int(y + y1)] for (x, y) in coords]
                    global_wrinkles.append(coords_global)
            else:
                global_wrinkles = []
            # offset instances to full image coords for NMS later
            instances_offset = offset_instances(instances, x1, y1, image.shape[:2]) if len(instances) > 0 else Instances(image.shape[:2])
            result[role] = {"bbox": (x1,y1,x2,y2), "wrinkles": global_wrinkles, "instances": instances_offset}
        except Exception as e:
            print(f"[ERROR] wrinkle prediction failed for {image_name} {role}: {e}")
            result[role] = {"bbox": (x1,y1,x2,y2), "wrinkles": [], "instances": None}

    return result

# -------------------- Mapping logic for three images --------------------
def pick_left_full_seat_bbox(roles_dict, image):
    # pick left-most seat bbox in image
    if not roles_dict:
        return None
    best = None
    best_x = float("inf")
    for role, info in roles_dict.items():
        x1,y1,x2,y2 = info["bbox"]
        cx = (x1 + x2) / 2.0
        if cx < best_x:
            best_x = cx
            best = info["bbox"]
    return best

def pick_right_full_seat_bbox(roles_dict, image):
    if not roles_dict:
        return None
    best = None
    best_x = -float("inf")
    for role, info in roles_dict.items():
        x1,y1,x2,y2 = info["bbox"]
        cx = (x1 + x2) / 2.0
        if cx > best_x:
            best_x = cx
            best = info["bbox"]
    return best

def pick_center_full_seat_bbox(roles_dict, image):
    if not roles_dict:
        return None
    H, W = image.shape[:2]
    img_cx = W / 2.0
    # prefer "middle"
    if "middle" in roles_dict:
        return roles_dict["middle"]["bbox"]
    # otherwise pick seat closest to image center
    best = None; best_dist = float("inf")
    for role, info in roles_dict.items():
        x1,y1,x2,y2 = info["bbox"]
        cx = (x1 + x2) / 2.0
        dist = abs(cx - img_cx)
        if dist < best_dist:
            best_dist = dist
            best = info["bbox"]
    return best

def process_three_images_for_mapping(img_paths, save_map_dir):
    """
    img_paths: [left_img_path, center_img_path, right_img_path]
    Produces mapping outputs under save_map_dir
    """
    assert len(img_paths) == 3
    images = [cv2.imread(p) for p in img_paths]
    names = [os.path.basename(p) for p in img_paths]

    # run detection on each
    seat_data = {}
    for idx, p in enumerate(img_paths):
        img = images[idx]
        if img is None:
            print(f"[WARN] can't read {p}")
            seat_data[os.path.basename(p)] = {}
            continue
        seat_data[os.path.basename(p)] = detect_seats_and_wrinkles(img, os.path.basename(p), idx)

    left_name, center_name, right_name = names
    left_roles = seat_data.get(left_name, {})
    center_roles = seat_data.get(center_name, {})
    right_roles = seat_data.get(right_name, {})

    # identify full seat bboxes
    full_left_bbox = pick_left_full_seat_bbox(left_roles, images[0]) or pick_center_full_seat_bbox(left_roles, images[0])
    full_middle_bbox = pick_center_full_seat_bbox(center_roles, images[1])
    full_right_bbox = pick_right_full_seat_bbox(right_roles, images[2]) or pick_center_full_seat_bbox(right_roles, images[2])

    integrated_left = []
    integrated_middle = []
    integrated_right = []

    # helper mapping function for one wrinkle set
    def map_wrinkle_from(image_idx, src_roles, wrinkle_coords, src_bbox):
        # compute centroid
        if len(wrinkle_coords) == 0:
            return None
        xs = [p[0] for p in wrinkle_coords]; ys = [p[1] for p in wrinkle_coords]
        cx = sum(xs) / len(xs)
        img = images[image_idx]
        H, W = img.shape[:2]
        img_cx = W / 2.0

        # image 0 (left)
        if image_idx == 0:
            # if on right-half of left image -> belongs to full_middle
            if cx > img_cx and full_middle_bbox is not None:
                return ("middle", map_points_proportional_global(wrinkle_coords, src_bbox, full_middle_bbox))
            else:
                if full_left_bbox is not None:
                    return ("left", map_points_proportional_global(wrinkle_coords, src_bbox, full_left_bbox))
                else:
                    return None
        # image 1 (center)
        elif image_idx == 1:
            # left-half -> full_left; right-half -> full_right; center region / inside middle bbox -> full_middle
            if full_middle_bbox is not None:
                x1m,y1m,x2m,y2m = full_middle_bbox
                if x1m <= cx <= x2m:
                    return ("middle", map_points_proportional_global(wrinkle_coords, src_bbox, full_middle_bbox))
            if cx < img_cx and full_left_bbox is not None:
                return ("left", map_points_proportional_global(wrinkle_coords, src_bbox, full_left_bbox))
            elif cx > img_cx and full_right_bbox is not None:
                return ("right", map_points_proportional_global(wrinkle_coords, src_bbox, full_right_bbox))
            else:
                # fallback to middle if available
                if full_middle_bbox is not None:
                    return ("middle", map_points_proportional_global(wrinkle_coords, src_bbox, full_middle_bbox))
                return None
        # image 2 (right)
        elif image_idx == 2:
            if cx < img_cx and full_middle_bbox is not None:
                return ("middle", map_points_proportional_global(wrinkle_coords, src_bbox, full_middle_bbox))
            else:
                if full_right_bbox is not None:
                    return ("right", map_points_proportional_global(wrinkle_coords, src_bbox, full_right_bbox))
                else:
                    return None
        return None

    # map wrinkles from all seat roles in each image
    for idx, name in enumerate([left_name, center_name, right_name]):
        roles = seat_data.get(name, {})
        for role, info in roles.items():
            src_bbox = info["bbox"]
            for w in info.get("wrinkles", []):
                mapped = map_wrinkle_from(idx, roles, w, src_bbox)
                if mapped is None:
                    continue
                tgt_role, coords = mapped
                if tgt_role == "left":
                    integrated_left.append(coords)
                elif tgt_role == "middle":
                    integrated_middle.append(coords)
                elif tgt_role == "right":
                    integrated_right.append(coords)

    # create mapping dir and temp dir
    os.makedirs(save_map_dir, exist_ok=True)
    temp_dir = os.path.join(save_map_dir, TEMP_SUBDIR_NAME)
    os.makedirs(temp_dir, exist_ok=True)

    # Save visualizations for each full seat (if bbox exists)
    outputs = {}
    # LEFT
    if full_left_bbox is not None:
        vis_left = images[0].copy()
        vis_left = draw_wrinkle_points_overlay(vis_left, integrated_left, color=(0,0,255), alpha=0.4)
        x1,y1,x2,y2 = map(int, full_left_bbox)
        cv2.rectangle(vis_left, (x1,y1), (x2,y2), (255,255,0), 2)
        cv2.putText(vis_left, "full_left", (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        save_left = os.path.join(save_map_dir, os.path.splitext(left_name)[0] + "_full_left_integrated.png")
        cv2.imwrite(save_left, vis_left)
        outputs["full_left"] = save_left

    # MIDDLE (visualize on center image)
    if full_middle_bbox is not None:
        vis_mid = images[1].copy()
        vis_mid = draw_wrinkle_points_overlay(vis_mid, integrated_middle, color=(0,0,255), alpha=0.4)
        x1,y1,x2,y2 = map(int, full_middle_bbox)
        cv2.rectangle(vis_mid, (x1,y1), (x2,y2), (255,255,0), 2)
        cv2.putText(vis_mid, "full_middle", (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        save_mid = os.path.join(save_map_dir, os.path.splitext(center_name)[0] + "_full_middle_integrated.png")
        cv2.imwrite(save_mid, vis_mid)
        outputs["full_middle"] = save_mid

    # RIGHT
    if full_right_bbox is not None:
        vis_right = images[2].copy()
        vis_right = draw_wrinkle_points_overlay(vis_right, integrated_right, color=(0,0,255), alpha=0.4)
        x1,y1,x2,y2 = map(int, full_right_bbox)
        cv2.rectangle(vis_right, (x1,y1), (x2,y2), (255,255,0), 2)
        cv2.putText(vis_right, "full_right", (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        save_right = os.path.join(save_map_dir, os.path.splitext(right_name)[0] + "_full_right_integrated.png")
        cv2.imwrite(save_right, vis_right)
        outputs["full_right"] = save_right

    # Save JSON summary of mapping (coordinates)
    mapping_summary = {
        "full_left_bbox": full_left_bbox,
        "full_middle_bbox": full_middle_bbox,
        "full_right_bbox": full_right_bbox,
        "integrated_left_count": len(integrated_left),
        "integrated_middle_count": len(integrated_middle),
        "integrated_right_count": len(integrated_right),
        "integrated_left": integrated_left,
        "integrated_middle": integrated_middle,
        "integrated_right": integrated_right
    }
    with open(os.path.join(save_map_dir, "mapping_summary.json"), "w") as f:
        json.dump(mapping_summary, f, indent=2)

    return outputs, mapping_summary

# -------------------- Single-image inference (per-file) --------------------
def single_image_inference_and_save(image_path, save_inference_dir):
    """
    Run YOLO -> crop seats -> run wrinkle predictor -> offset -> NMS -> save visualization
    Also returns seat_data for mapping if caller wants to reuse.
    """
    os.makedirs(save_inference_dir, exist_ok=True)
    image = cv2.imread(image_path)
    if image is None:
        print(f"[WARN] can't read {image_path}")
        return {}
    H, W = image.shape[:2]
    res = yolo_model(image)[0]
    masks = None
    try:
        masks = res.masks.data.cpu().numpy()
    except Exception:
        masks = None

    if masks is None:
        # save original image as-is
        out_path = os.path.join(save_inference_dir, os.path.basename(image_path))
        cv2.imwrite(out_path, image)
        return {}

    class_ids = res.boxes.cls.cpu().numpy().astype(int)
    id_to_name = yolo_model.names

    all_instances = []
    seat_role_data = {}
    # iterate masks
    for idx, m in enumerate(masks):
        cls_name = id_to_name[int(class_ids[idx])].lower()
        if cls_name != "seat":
            continue
        mask_full = cv2.resize((m > 0.3).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        bx, by, bw, bh = cv2.boundingRect(mask_full)
        if bw == 0 or bh == 0:
            continue
        x1, y1, x2, y2 = bx, by, bx + bw, by + bh
        cropped_image = image[y1:y2, x1:x2]
        cropped_mask = mask_full[y1:y2, x1:x2]
        masked_roi = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)
        # run wrinkle predictor
        try:
            outputs = predictor_wrinkle(masked_roi)
            instances = outputs["instances"].to("cpu")
            if len(instances) > 0:
                instances_off = offset_instances(instances, x1, y1, image.shape[:2])
                # optional: relabel_instances(instances_off, 1)
                all_instances.append(instances_off)
                # also store wrinkle points in global coords
                masks_np = instances.pred_masks.numpy()
                wrinkles = []
                for mask_local in masks_np:
                    ys, xs = np.where(mask_local > 0)
                    if len(xs) == 0:
                        continue
                    coords = np.column_stack((xs, ys))
                    coords_global = [[int(x + x1), int(y + y1)] for (x,y) in coords]
                    wrinkles.append(coords_global)
            else:
                wrinkles = []
        except Exception as e:
            print(f"[ERROR] predictor_wrinkle failed on {image_path} ROI: {e}")
            wrinkles = []
        # store seat role data (we'll assign roles later in mapping step)
        seat_role_data[f"seat_{idx}"] = {"bbox": (x1,y1,x2,y2), "wrinkles": wrinkles}

    # merge instances and NMS
    if len(all_instances) > 0:
        combined = Instances.cat(all_instances)
        combined = apply_nms(combined, iou_threshold=0.3)
        # prepare masks and boxes, scores for visualization
        masks_out = combined.pred_masks.numpy().astype(np.uint8)
        boxes_out = combined.pred_boxes.tensor.numpy().astype(int).tolist()
        scores_out = combined.scores.numpy().tolist()
        vis = draw_masks_and_boxes_on_image(image.copy(), masks_out, boxes_out, scores_out)
    else:
        vis = image.copy()

    out_name = os.path.splitext(os.path.basename(image_path))[0] + "_wrinkle_vis.png"
    out_path = os.path.join(save_inference_dir, out_name)
    cv2.imwrite(out_path, vis)
    return seat_role_data

# -------------------- Folder orchestration --------------------
def process_folder_tree(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        image_files = sorted([f for f in files if f.lower().endswith((".jpg",".jpeg",".png"))])
        if len(image_files) == 0:
            continue
        rel = os.path.relpath(root, input_dir)
        # create canonical seat folder under output_dir
        seat_dir = os.path.join(output_dir, rel)
        inference_dir = os.path.join(seat_dir, "inference")
        mapping_dir = os.path.join(seat_dir, "mapping")
        os.makedirs(inference_dir, exist_ok=True)
        os.makedirs(mapping_dir, exist_ok=True)

        # special-case: exactly 8 images -> run 3-image mapping on first 3
        if len(image_files) == 8:
            print(f"[INFO] folder {root} contains 8 images -> running mapping on first 3")
            first_three = image_files[:3]
            first_three_paths = [os.path.join(root, f) for f in first_three]
            # run per-image inference for all images (saves in inference_dir)
            for f in image_files:
                p = os.path.join(root, f)
                single_image_inference_and_save(p, inference_dir)
            # run mapping on first 3 images
            outputs, summary = process_three_images_for_mapping(first_three_paths, mapping_dir)
            print(f"[INFO] mapping outputs saved for {root}: {outputs.keys()}")
        else:
            print(f"[INFO] folder {root} contains {len(image_files)} images -> running single-image inference on all")
            for f in image_files:
                p = os.path.join(root, f)
                single_image_inference_and_save(p, inference_dir)

if __name__ == "__main__":
    print("[START] Running merged inference + mapping pipeline")
    process_folder_tree(INPUT_DIR, OUTPUT_DIR)
    print("[DONE] All processing finished.")
