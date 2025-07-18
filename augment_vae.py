import os
import cv2
import albumentations as A
from glob import glob
from tqdm import tqdm


# Set paths
INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/Inspector Pictures from Google Notes/dataset/scratch"
OUTPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/Inspector Pictures from Google Notes/dataset/augment_scratch"


def load_image(image_path):
    """Load image using OpenCV and convert to RGB"""
    image = cv2.imread(image_path)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None


def save_image(image, output_path):
    """Save RGB image as BGR to path"""
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, image_bgr)


def get_augmentations():
    """Returns a dictionary of augmentation name and corresponding transform"""
    return {
        "hflip": A.HorizontalFlip(p=1.0),
        "rotate": A.Rotate(limit=30, p=1.0),
        "brightness_contrast": A.RandomBrightnessContrast(p=1.0),
        "gaussian_noise": A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        "gaussian_blur": A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        "color_jitter": A.ColorJitter(p=1.0),
        "crop_resize": A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0)
    }


def augment_and_save(image, base_name, augmentations):
    """Applies each augmentation and saves one image per type"""
    for aug_name, aug in augmentations.items():
        transformed = aug(image=image)
        aug_img = transformed['image']
        save_path = os.path.join(OUTPUT_DIR, f"{base_name}_{aug_name}.jpg")
        save_image(aug_img, save_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    image_paths = glob(os.path.join(INPUT_DIR, "*"))
    augmentations = get_augmentations()

    for image_path in tqdm(image_paths, desc="Augmenting"):
        img = load_image(image_path)
        if img is None:
            continue
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        augment_and_save(img, base_name, augmentations)


if __name__ == "__main__":
    main()
