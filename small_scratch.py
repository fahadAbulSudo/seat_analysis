import numpy as np
import cv2
from noise import pnoise2
import matplotlib.pyplot as plt
import random

def generate_perlin_noise(width, height, scale=10.0, octaves=1, persistence=0.5, lacunarity=2.0, seed=None):
    if seed is None:
        seed = np.random.randint(0, 10000)
    noise_img = np.zeros((height, width))
    for y in range(height):
        for x in range(width):
            noise_val = pnoise2(x / scale, 
                                y / scale, 
                                octaves=octaves, 
                                persistence=persistence, 
                                lacunarity=lacunarity, 
                                repeatx=width, 
                                repeaty=height, 
                                base=seed)
            noise_img[y][x] = noise_val
    noise_img = (noise_img - noise_img.min()) / (noise_img.max() - noise_img.min())
    return noise_img

def generate_scratch_mask(width=256, height=256, num_scratches=3, length_range=(50, 120), thickness_range=(1, 3)):
    mask = np.zeros((height, width), dtype=np.uint8)
    for _ in range(num_scratches):
        x1 = random.randint(0, width - 1)
        y1 = random.randint(0, height - 1)
        angle = random.uniform(0, 2*np.pi)
        length = random.randint(*length_range)
        thickness = random.randint(*thickness_range)

        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        x2 = np.clip(x2, 0, width - 1)
        y2 = np.clip(y2, 0, height - 1)

        cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=thickness)

    return mask

def perturb_mask_with_noise(mask, noise_strength=0.4, noise_scale=10):
    height, width = mask.shape
    noise = generate_perlin_noise(width, height, scale=noise_scale)
    perturbed = mask.astype(np.float32) / 255.0
    perturbed = perturbed * (1.0 - noise_strength * noise)
    return (perturbed * 255).astype(np.uint8)

def apply_defect_to_image(image, mask, alpha=0.5):
    """Overlay a white scratch mask onto the image with some transparency."""
    h, w, _ = image.shape
    mh, mw = mask.shape
    x_offset = random.randint(0, w - mw)
    y_offset = random.randint(0, h - mh)

    # Create color mask
    color_mask = np.stack([mask]*3, axis=-1)

    # Define ROI
    roi = image[y_offset:y_offset+mh, x_offset:x_offset+mw]
    blended = cv2.addWeighted(roi, 1 - alpha, color_mask.astype(np.uint8), alpha, 0)
    image[y_offset:y_offset+mh, x_offset:x_offset+mw] = blended
    return image

# === MAIN EXECUTION ===
image_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/test/5_20250605_042038.jpg"
output_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/test/5_20250605_042038_with_scratch.jpg"

# Read image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Resize if too small
if image.shape[0] < 256 or image.shape[1] < 256:
    image = cv2.resize(image, (max(image.shape[1], 256), max(image.shape[0], 256)))

# Generate defect
scratch_mask = generate_scratch_mask(256, 256, num_scratches=5)
perturbed_mask = perturb_mask_with_noise(scratch_mask, noise_strength=0.5, noise_scale=20)

# Apply defect and save
defected_image = apply_defect_to_image(image.copy(), perturbed_mask, alpha=0.6)
cv2.imwrite(output_path, defected_image)
print(f"Saved defect image to {output_path}")
