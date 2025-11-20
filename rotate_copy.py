import os
import cv2

# Base folder containing seat directories
BASE_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/MSN12864_Left-20250812T102814Z-1-001/MSN12864_Left"

# Destination folder for rotated copies
DEST_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/back"

# Ensure destination folder exists
os.makedirs(DEST_DIR, exist_ok=True)

# Valid image extensions
VALID_EXTS = (".jpg", ".jpeg", ".png")

def rotate_and_copy(src_path, dest_path):
    """Read, rotate 90° counterclockwise, and save to destination path."""
    img = cv2.imread(src_path)
    if img is None:
        print(f"⚠ Skipping unreadable image: {src_path}")
        return
    rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    cv2.imwrite(dest_path, rotated)
    print(f"✅ Rotated & copied: {dest_path}")

def main():
    for seat_num in range(1, 41):  # seat_21 → seat_40
        seat_folder = f"seat_{seat_num}"
        raw_images_dir = os.path.join(BASE_DIR, seat_folder, "raw_images")
        if not os.path.exists(raw_images_dir):
            print(f"⚠ Skipping {raw_images_dir} (not found)")
            continue

        for file in os.listdir(raw_images_dir):
            if not file.lower().endswith(VALID_EXTS):
                continue

            # Process only files starting with 4_, 5_, or 6_
            if file.startswith(("0_", "1_", "2_")):
                src_path = os.path.join(raw_images_dir, file)
                dest_path = os.path.join(DEST_DIR, file)
                rotate_and_copy(src_path, dest_path)

if __name__ == "__main__":
    main()
    print("🎯 Rotation + Copy complete for seat_21 → seat_40")
