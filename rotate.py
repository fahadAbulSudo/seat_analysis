import os
import cv2

# Directories to process
base_dirs = [
    "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/zal/cam_7",
]

# Supported image extensions
valid_exts = (".jpg", ".jpeg", ".png")

def rotate_image(image):
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

def rotate_images_in_dir(directory):
    for file in os.listdir(directory):
        if file.lower().endswith(valid_exts):
            path = os.path.join(directory, file)
            image = cv2.imread(path)
            if image is None:
                print(f"Skipping unreadable image: {path}")
                continue
            rotated = rotate_image(image)
            cv2.imwrite(path, rotated)
            print(f"Rotated: {path}")

# Apply rotation
for dir_path in base_dirs:
    rotate_images_in_dir(dir_path)

print("Rotation complete.")
