import os
import cv2

# Base directory containing seat folders
BASE_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/MSN12864_Left-20250812T102814Z-1-001/MSN12864_Left"

# Supported image extensions
VALID_EXTS = (".jpg", ".jpeg", ".png")

def rotate_image(image, direction):
    """Rotate image based on direction ('cw' or 'ccw')."""
    if direction == "cw":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif direction == "ccw":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image

def process_seat_folder(seat_path):
    raw_images_dir = os.path.join(seat_path, "raw_images")
    if not os.path.exists(raw_images_dir):
        return

    files = sorted([f for f in os.listdir(raw_images_dir) if f.lower().endswith(VALID_EXTS)])
    if len(files) < 8:
        print(f"⚠ Skipping {raw_images_dir} (not enough images)")
        return

    # First 4 images: rotate clockwise
    # for f in files[:4]:
    #     img_path = os.path.join(raw_images_dir, f)
    #     img = cv2.imread(img_path)
    #     if img is None:
    #         print(f"Skipping unreadable image: {img_path}")
    #         continue
    #     rotated = rotate_image(img, "cw")
    #     cv2.imwrite(img_path, rotated)
    #     print(f"✅ Rotated CW: {img_path}")

    # Last 4 images: rotate counterclockwise
    for f in files[4:8]:
        img_path = os.path.join(raw_images_dir, f)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping unreadable image: {img_path}")
            continue
        rotated = rotate_image(img, "ccw")
        cv2.imwrite(img_path, rotated)
        print(f"✅ Rotated CCW: {img_path}")

def main():
    for seat_folder in sorted(os.listdir(BASE_DIR)):
        seat_path = os.path.join(BASE_DIR, seat_folder)
        if os.path.isdir(seat_path) and seat_folder.startswith("seat_"):
            process_seat_folder(seat_path)

if __name__ == "__main__":
    main()
    print("🎯 Rotation complete for all seat folders.")
