import cv2
import math
import numpy as np
import os

# Constants
FOCAL_LENGTH_MM = 4.0          # Focal length of the camera in millimeters
SENSOR_Width_MM = 3.6          # Sensor width in millimeters
SENSOR_HEIGHT_MM = 4.5         # Sensor height in millimeters
TILT_ANGLE_DEGREES = [0, 0, 0]        # Tilt angle of the camera in degrees

# Image paths and corresponding points and distances
image_paths = [
    "/home/fahadabul/mask_rcnn_skyhub/Subtitles/straight100.jpg",
    "/home/fahadabul/mask_rcnn_skyhub/Subtitles/straigh50.jpg"
]
# image_paths = [
#     "/home/fahadabul/mask_rcnn_skyhub/Subtitles/Tilt10.jpg",
#     "/home/fahadabul/mask_rcnn_skyhub/Subtitles/Tilt20.jpg",
#     "/home/fahadabul/mask_rcnn_skyhub/Subtitles/tilt30.jpg"
# ]
points = [
    [(2090, 4370), (2500, 3240)],  # for first image
    [(2220, 5790), (3000, 3470)]   # for second image
]
# points = [
#     [(3260, 4320), (4080, 2040)],
#     [(3180, 3180), (4090, 510)],
#     [(1850, 3450), (2820, 770)]
# ]
# DISTANCE_TO_OBJECT_MM = [485.0, 460.0, 430.0]
DISTANCE_TO_OBJECT_MM = [1020.0, 1020.0]

mask = False

for idx, image_path in enumerate(image_paths):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")



    # Extract points for current image
    point1, point2 = points[idx]

    if mask:
        # Calculate min/max for rectangular mask
        x_min = min(point1[0], point2[0])
        x_max = max(point1[0], point2[0])
        y_min = min(point1[1], point2[1])
        y_max = max(point1[1], point2[1])

        # Create blank mask
        # Get image dimensions
        image_height, image_width = image.shape[:2]
        mask_img = np.zeros((image_height, image_width), dtype=np.uint8)
        # Fill rectangle region in mask
        mask_img[y_min:y_max, x_min:x_max] = 255

        # Save mask
        mask_dir = '/home/fahadabul/mask_rcnn_skyhub/tmp'
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, f'mask_{idx}.png')
        cv2.imwrite(mask_path, mask_img)
        print(f"Mask saved to {mask_path}")

        # Crop image to masked region (optional)
        image_cropped = image[y_min:y_max, x_min:x_max]
    else:
        # Calculate min/max for rectangular mask
        x_min = min(point1[0], point2[0])
        x_max = max(point1[0], point2[0])
        y_min = min(point1[1], point2[1])
        y_max = max(point1[1], point2[1])
        image_cropped = image
        mask_img = None
    # Get image dimensions
    image_height, image_width = image_cropped.shape[:2]
    # Calculate pixel distances inside mask region
    pixel_width = x_max - x_min
    pixel_height = y_max - y_min

    # Convert tilt angle to radians
    tilt_angle_rad = math.radians(TILT_ANGLE_DEGREES[idx])

    # Calculate real-world width using pinhole camera model adjusted for tilt
    real_width_mm = (pixel_width * SENSOR_Width_MM * DISTANCE_TO_OBJECT_MM[idx]) / (
        image_width * FOCAL_LENGTH_MM * math.cos(tilt_angle_rad))

    # Calculate real-world height using pinhole camera model adjusted for tilt
    real_height_mm = (pixel_height * SENSOR_HEIGHT_MM * DISTANCE_TO_OBJECT_MM[idx]) / (
        image_height * FOCAL_LENGTH_MM * math.cos(tilt_angle_rad))

    # Convert to centimeters
    real_width_cm = real_width_mm / 10
    real_height_cm = real_height_mm / 10

    # Output results
    print(f"Image {idx+1}:")
    print(f"  Pixel Width: {pixel_width} pixels")
    print(f"  Real-World Width: {real_width_mm:.2f} mm ({real_width_cm:.2f} cm)")
    print(f"  Pixel Height: {pixel_height} pixels")
    print(f"  Real-World Height: {real_height_mm:.2f} mm ({real_height_cm:.2f} cm)\n")
