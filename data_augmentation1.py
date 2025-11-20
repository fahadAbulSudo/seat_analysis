import os
import cv2
import numpy as np

# === Color Augmentation Functions ===

def adjust_brightness(img, delta=30):
    """Increase or decrease brightness in HSV space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = cv2.split(hsv)
    v = np.clip(v + delta, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(img, alpha=1.5, beta=0):
    """Contrast = alpha * pixel + beta."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def adjust_saturation(img, scale=1.5):
    """Scale the S channel in HSV space."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def shift_hue(img, shift=30):
    """Rotate the hue channel by `shift` degrees."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[...,0] = (hsv[...,0] + shift) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

def gamma_correction(img, gamma=1.2):
    """Apply gamma correction."""
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def per_channel_scaling(img, scales=(1.2, 1.0, 0.8)):
    """Multiply each BGR channel by a different factor."""
    b, g, r = cv2.split(img.astype(np.float32))
    b = np.clip(b * scales[0], 0, 255)
    g = np.clip(g * scales[1], 0, 255)
    r = np.clip(r * scales[2], 0, 255)
    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])

def channel_permutation(img, order=(2, 0, 1)):
    """Reorder the BGR channels by `order` tuple."""
    chans = cv2.split(img)
    return cv2.merge([chans[i] for i in order])

# === Main Augmentation Loop ===

def augment_dataset(input_dir, output_dir):
    """
    Reads all images from input_dir, applies color-based augmentations,
    and writes results to output_dir with annotated filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        src_path = os.path.join(input_dir, fname)
        img = cv2.imread(src_path)
        if img is None:
            continue

        basename, ext = os.path.splitext(fname)

        # List of (augmented_image, suffix)
        variants = [
            (adjust_brightness(img, delta=40), "_bright"),
            (adjust_contrast(img, alpha=1.7, beta=0), "_contrast"),
            (adjust_saturation(img, scale=1.8), "_saturated"),
            (shift_hue(img, shift=45), "_hue"),
            (gamma_correction(img, gamma=1.3), "_gamma"),
            (per_channel_scaling(img, scales=(1.3,0.7,1.0)), "_chscale"),
            (channel_permutation(img, order=(2,1,0)), "_permute")
        ]

        for aug_img, suffix in variants:
            out_name = f"{basename}{suffix}{ext}"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, aug_img)

    print("All augmentations complete. Images saved to:", output_dir)

# === Usage ===

if __name__ == "__main__":
    INPUT_DIR = "/home/swapnil/AIR_BUS/May/7th_data_augmentation_for_training_yolo/dataset/"
    OUTPUT_DIR = "/home/swapnil/AIR_BUS/May/7th_data_augmentation_for_training_yolo/dataset/augmented_output/"
    augment_dataset(INPUT_DIR, OUTPUT_DIR)
