import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon

# === Paths ===
annotations_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/aug_8/close/result.json"
images_dir = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/aug_8/close/images"
vis_dir = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/aug_8/close/extract"
os.makedirs(vis_dir, exist_ok=True)

# === Target images to fix ===
target_images = {
    "3bae7f67-20250630_130738.jpg",
    "37b396b3-20250630_131056.jpg",
    "dca30044-20250630_130426.jpg",
    "e6017fb1-20250630_131937.jpg",
    "e9011798-20250630_131013.jpg"
}

# === Load JSON ===
with open(annotations_path) as f:
    coco = json.load(f)

image_id_map = {img['id']: img for img in coco['images']}
annotations_by_image = {}
for ann in coco['annotations']:
    annotations_by_image.setdefault(ann['image_id'], []).append(ann)

category_id_map = {cat['id']: cat['name'] for cat in coco.get("categories", [])}
updated_annotations = []

# === Main Fix and Visualize Loop ===
for image_id, image_info in image_id_map.items():
    filename = image_info["file_name"]

    # Only process target images
    if filename not in target_images:
        updated_annotations.extend(annotations_by_image[image_id])
        continue

    json_width, json_height = image_info["width"], image_info["height"]
    image_path = os.path.join(images_dir, filename)

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        continue

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load {image_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    actual_height, actual_width = img.shape[:2]

    needs_fix = (json_width, json_height) == (actual_height, actual_width) or \
                (json_width == actual_height and json_height == actual_width)

    if needs_fix:
        image_info["width"], image_info["height"] = actual_width, actual_height

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)

        for ann in annotations_by_image[image_id]:
            # Only fix wrinkle annotations
            if category_id_map.get(ann["category_id"], "").lower() != "wrinkle":
                updated_annotations.append(ann)
                continue

            # --- Fix bbox ---
            x, y, w, h = ann["bbox"]
            norm_x = x / json_width
            norm_y = y / json_height
            norm_w = w / json_width
            norm_h = h / json_height

            draw_x = norm_x * actual_width
            draw_y = norm_y * actual_height
            draw_w = norm_w * actual_width
            draw_h = norm_h * actual_height
            ann["bbox"] = [draw_x, draw_y, draw_w, draw_h]

            # --- Fix segmentation ---
            new_seg = []
            for seg in ann["segmentation"]:
                norm_pts = [
                    (seg[i] / json_width * actual_width,
                     seg[i + 1] / json_height * actual_height)
                    for i in range(0, len(seg), 2)
                ]
                flat = [coord for pt in norm_pts for coord in pt]
                new_seg.append(flat)

                poly = Polygon(norm_pts, closed=True, edgecolor='red', facecolor='red', alpha=0.4)
                ax.add_patch(poly)

            ann["segmentation"] = new_seg
            ann["area"] = draw_w * draw_h

            # --- Draw bbox ---
            rect = patches.Rectangle((draw_x, draw_y), draw_w, draw_h,
                                     linewidth=2, edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            ax.text(draw_x, draw_y - 5, "wrinkle", color='blue', fontsize=10)

            updated_annotations.append(ann)

        vis_path = os.path.join(vis_dir, f"bbox_{filename}")
        fig.savefig(vis_path)
        plt.close(fig)
        print(f"✅ Saved: {vis_path}")
    else:
        updated_annotations.extend(annotations_by_image[image_id])

# === Save Updated JSON in place ===
coco["annotations"] = updated_annotations
with open(annotations_path, "w") as f:  # Overwrite same file
    json.dump(coco, f)
print(f"✅ Updated JSON saved to: {annotations_path}")
