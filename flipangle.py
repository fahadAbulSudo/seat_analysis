import os
import cv2

INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/test_new"
OUTPUT_DIR = os.path.join(INPUT_DIR, "augmented_flips")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Flip codes:
# 1 = horizontal, 0 = vertical, -1 = both
flip_types = {
    "flip_horizontal": 1,
    "flip_vertical": 0,
    "flip_both": -1
}

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(INPUT_DIR, filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Failed to load image: {filename}")
        continue

    base_name, ext = os.path.splitext(filename)

    for flip_name, flip_code in flip_types.items():
        flipped = cv2.flip(image, flip_code)
        save_path = os.path.join(OUTPUT_DIR, f"{base_name}_{flip_name}{ext}")
        cv2.imwrite(save_path, flipped)

    print(f"Augmented {filename}")
