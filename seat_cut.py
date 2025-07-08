import os
import random
from PIL import Image

def random_cut_image(image_path, output_dir):
    # Load image
    image = Image.open(image_path)
    width, height = image.size

    # Randomly choose cut direction
    cut_direction = random.choice(['horizontal', 'vertical'])

    # Decide the cut position (avoid too small slices)
    if cut_direction == 'horizontal':
        cut_line = random.randint(int(height * 0.3), int(height * 0.7))
        top_part = image.crop((0, 0, width, cut_line))
        bottom_part = image.crop((0, cut_line, width, height))
        top_part.save(os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_top.jpg"))
        bottom_part.save(os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_bottom.jpg"))
    else:
        cut_line = random.randint(int(width * 0.3), int(width * 0.7))
        left_part = image.crop((0, 0, cut_line, height))
        right_part = image.crop((cut_line, 0, width, height))
        left_part.save(os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_left.jpg"))
        right_part.save(os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_right.jpg"))

    print(f"Image {os.path.basename(image_path)} cut {cut_direction} at {cut_line}")

# Example usage:
input_dir = "/home/fahadabul/mask_rcnn_skyhub/output_predictions/yolo_segmented_seats"
output_dir = "/home/fahadabul/mask_rcnn_skyhub/output_predictions/cut/images"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        random_cut_image(os.path.join(input_dir, filename), output_dir)
