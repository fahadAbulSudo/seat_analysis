import cv2
import numpy as np
import os
import json
from imantics import Mask

# Define folders
image_folder = "/home/fahadabul/fcc-intro-to-llms/latest_image_mask_rcnn/torn"
mask_folder = "/home/fahadabul/fcc-intro-to-llms/latest_image_mask_rcnn/brush_labels_to_png_format_img_segmentation"
output_folder = "/home/fahadabul/fcc-intro-to-llms/latest_image_mask_rcnn/annotated_images"
polygon_output = "/home/fahadabul/fcc-intro-to-llms/latest_image_mask_rcnn/polygon_annotations"
coco_json_path = "/home/fahadabul/fcc-intro-to-llms/latest_image_mask_rcnn/annotations.json"

# Create output folders if they don't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(polygon_output, exist_ok=True)

# Initialize COCO annotation format
coco_annotations = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 1, "name": "object", "supercategory": "none"}]
}

# Get sorted lists of image and mask files
image_files = sorted(os.listdir(image_folder))
mask_files = sorted(os.listdir(mask_folder))

# Track annotation ID and image ID
annotation_id = 1
image_id = 1

# Process each pair of images and masks
for img_name, mask_name in zip(image_files, mask_files):
    img_path = os.path.join(image_folder, img_name)
    mask_path = os.path.join(mask_folder, mask_name)
    output_img_path = os.path.join(output_folder, img_name)
    polygon_txt_path = os.path.join(polygon_output, os.path.splitext(img_name)[0] + ".txt")
    
    # Read images
    image = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert mask to float format
    mask_float = mask.astype(np.float32)
    
    # Extract polygons from mask
    polygons = Mask(mask_float).polygons()

    # Get image dimensions
    height, width = image.shape[:2]

    # Add image details to COCO JSON
    coco_annotations["images"].append({
        "id": image_id,
        "file_name": img_name,
        "width": width,
        "height": height
    })

    # Open text file to save polygon points
    with open(polygon_txt_path, "w") as f:
        for segmentation in polygons.segmentation:  # Correctly using .segmentation
            # Flatten segmentation list
            segmentation = [int(coord) for coord in segmentation]

            # Write polygon points to file
            points_str = " ".join([f"{segmentation[i]},{segmentation[i+1]}" for i in range(0, len(segmentation), 2)])
            f.write(points_str + "\n")

            # Convert segmentation to NumPy array for visualization
            polygon_points = np.array(segmentation, dtype=np.int32).reshape((-1, 1, 2))

            # Draw polygon on image
            cv2.polylines(image, [polygon_points], isClosed=True, color=(0, 255, 0), thickness=2)

            # Add annotation to COCO JSON
            coco_annotations["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [segmentation],  # Directly storing segmentation
                "area": cv2.contourArea(polygon_points.astype(np.int32)),  # Approximate area
                "bbox": cv2.boundingRect(polygon_points.astype(np.int32)),  # Bounding box
                "iscrowd": 0
            })

            annotation_id += 1

    # Save the annotated image
    cv2.imwrite(output_img_path, image)
    
    # Increment image ID
    image_id += 1

# Save COCO JSON file
with open(coco_json_path, "w") as json_file:
    json.dump(coco_annotations, json_file, indent=4)

print("Processing completed. COCO JSON, annotated images, and polygon data saved.")

