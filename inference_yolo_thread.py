import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Define input and output directories
INPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/thread"
OUTPUT_DIR = "/home/fahadabul/mask_rcnn_skyhub/thread/new"
YOLO_MODEL_PATH = "/home/fahadabul/mask_rcnn_skyhub/thread/best.pt"

# Load the YOLO model
try:
    yolo_model = YOLO(YOLO_MODEL_PATH)
    print(f"YOLO model loaded successfully from {YOLO_MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLO model from {YOLO_MODEL_PATH}: {e}")
    exit() # Exit if model cannot be loaded

def process_images(input_dir, output_dir):
    """
    Processes images in the input directory, performs object detection using YOLO,
    draws bounding boxes on detected objects, and saves the results to the output directory.

    Args:
        input_dir (str): The path to the directory containing input images.
        output_dir (str): The path to the directory where processed images will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Processing images from {input_dir} and saving to {output_dir}")

    # Walk through all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            # Check if the file is a supported image format
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(root, file)
            # Determine the relative path to maintain directory structure in output
            relative_path = os.path.relpath(root, input_dir)
            save_dir = os.path.join(output_dir, relative_path)
            os.makedirs(save_dir, exist_ok=True) # Create output subdirectory if it doesn't exist
            output_path = os.path.join(save_dir, file)

            print(f"Processing image: {image_path}")

            # Read the image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                print(f"Skipping {image_path}: Unable to read image. Check file permissions or corruption.")
                continue

            # Perform object detection with the YOLO model
            # The model returns a list of Results objects. We take the first one [0]
            # to get the results for the current image.
            try:
                yolo_results = yolo_model(image)[0]
            except Exception as e:
                print(f"Error during YOLO inference for {image_path}: {e}")
                continue

            # Iterate through each detected bounding box
            # yolo_results.boxes contains the bounding box data
            # yolo_results.names maps class IDs to class names
            for box in yolo_results.boxes:
                # Get bounding box coordinates (x1, y1, x2, y2)
                # .xyxy returns a tensor, convert to numpy array and then to int
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

                # Get confidence score
                confidence = box.conf[0].cpu().numpy()

                # Get class ID
                class_id = int(box.cls[0].cpu().numpy())
                class_name = yolo_results.names[class_id]

                # Define color for bounding box and text (BGR format)
                color = (0, 255, 0) # Green color

                # Draw the bounding box on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2) # 2 is the thickness of the line

                # Prepare the label text
                label = f"{class_name} {confidence:.2f}"

                # Define text properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

                # Calculate text position (above the bounding box)
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10

                # Draw background rectangle for text for better readability
                cv2.rectangle(image, (text_x, text_y - text_size[1] - 5),
                              (text_x + text_size[0] + 5, text_y + 5),
                              color, -1) # -1 fills the rectangle

                # Put the label text on the image
                cv2.putText(image, label, (text_x, text_y), font, font_scale,
                            (0, 0, 0), font_thickness, cv2.LINE_AA) # Black text

            # Save the image with bounding boxes
            cv2.imwrite(output_path, image)
            print(f"Saved processed image to: {output_path}")

# Main execution block
if __name__ == "__main__":
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        print("Please ensure the INPUT_DIR path is correct and accessible.")
    elif not os.path.isdir(INPUT_DIR):
        print(f"Error: Input path '{INPUT_DIR}' is not a directory.")
    else:
        # Call the function to process images
        process_images(INPUT_DIR, OUTPUT_DIR)
        print("Image processing complete.")


