import cv2
import math
import numpy as np
import os

# Constants
FOCAL_LENGTH_MM = 4.0          
SENSOR_WIDTH_MM = 3.6        # Approximate sensor width for 1/3.4"
SENSOR_HEIGHT_MM = 4.5         # Approximate sensor height for 1/3.4"
TILT_ANGLE_DEGREES = 0          # Assume 0° tilt (update if camera is tilted)

# Single image path
image_paths = ["/home/fahadabul/mask_rcnn_skyhub/Subtitles/tape.jpg",
               "/home/fahadabul/mask_rcnn_skyhub/Subtitles/Seat.jpg",
               "/home/fahadabul/mask_rcnn_skyhub/Subtitles/Seat.jpg"]
# === Camera Intrinsics ===
CALIBRATION_PATH = "/home/fahadabul/mask_rcnn_skyhub/Subtitles/calibration_data.npz"

# Points: [(x1, y1), (x2, y2)] for each object
points = [
    [(2610, 3500), (2760, 2200)],  
    [(2520, 4910), (3690, 2800)],  
    [(350, 4830), (720, 2920)]     
]

# Distances from the camera in mm for each object
DISTANCE_TO_OBJECT_MM = [
    610.0,
    890.0,
    710.0
]
for idx, image_path in enumerate(image_paths):
    # Load the image once
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")


    image_height, image_width = image.shape[:2]
    tilt_angle_rad = math.radians(TILT_ANGLE_DEGREES)
    x1, y1 = points[idx][0]
    x2, y2 = points[idx][1]
    # Sort coordinates
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Calculate pixel dimensions
    pixel_width = x_max - x_min
    pixel_height = y_max - y_min

    # Get corresponding distance
    distance_mm = DISTANCE_TO_OBJECT_MM[idx]
    print(distance_mm)
    # Real-world width using pinhole model (adjusted for tilt)
    real_width_mm = (pixel_width * SENSOR_WIDTH_MM * distance_mm) / (
        image_width * FOCAL_LENGTH_MM * math.cos(tilt_angle_rad))

    real_height_mm = (pixel_height * SENSOR_HEIGHT_MM * distance_mm) / (
        image_height * FOCAL_LENGTH_MM * math.cos(tilt_angle_rad))

    # Step 1: Compute vertical field of view (FOV) in radians
    # fov_vertical_rad = 2 * math.atan((4.5 / 2) / FOCAL_LENGTH_MM)
    # Step 2: Real-world height covered by entire image
    # real_world_image_height_mm = 2 * distance_mm * math.tan(fov_vertical_rad / 2)
    # print(real_world_image_height_mm)

    # Step 3: Height per pixel
    # mm_per_pixel_vertical = real_world_image_height_mm / image_height

    # Step 4: Multiply by pixel height to get real-world height
    # real_height_mm_2 = pixel_height * mm_per_pixel_vertical
    # Convert to cm
    real_width_cm = real_width_mm / 10
    # real_height_cm = real_height_mm / 10 * 1.75
    real_height_cm = real_height_mm / 10 
    # real_height_cm_2 = real_height_mm_2 / 10 

    # Save cropped region for reference (optional)
    # crop = image[y_min:y_max, x_min:x_max]
    # crop_path = os.path.join(crop_dir, f"crop_object_{idx+1}.png")
    # cv2.imwrite(crop_path, crop)

    # Output
    print(f"Object {idx+1}:")
    print(f"  Pixel Width: {pixel_width}px")
    print(f"  Real-World Width: {real_width_cm:.2f} cm")
    print(f"  Pixel Height: {pixel_height}px")
    print(f"  Real-World Height: {real_height_cm:.2f} cm")
    # print(f"  Real-World Height_2: {real_height_cm_2:.2f} cm")
    # print(f"  Cropped image saved to: {crop_path}\n")


# Constants
FOCAL_LENGTH_MM = 4.0
SENSOR_WIDTH_MM = 3.6        # mm
SENSOR_HEIGHT_MM = 4.5       # mm
TILT_ANGLE_DEGREES = 0

# Paths
image_paths = [
    "/home/fahadabul/mask_rcnn_skyhub/Subtitles/tape.jpg",
    "/home/fahadabul/mask_rcnn_skyhub/Subtitles/Seat.jpg",
    "/home/fahadabul/mask_rcnn_skyhub/Subtitles/Seat.jpg"
]
CALIBRATION_PATH = "/home/fahadabul/mask_rcnn_skyhub/Subtitles/calibration_data.npz"

# Original points (from distorted images)
points = [
    [(2610, 3500), (2760, 2200)], 
    [(2520, 4910), (3690, 2800)],    
    [(350, 4830), (720, 2920)]    
]

# Distances from camera to object in mm
DISTANCE_TO_OBJECT_MM = [610.0, 890.0, 710.0]

# Load camera calibration
data = np.load(CALIBRATION_PATH)
K = data["K"]
dist = data["dist"]

# Iterate over images and measure
for idx, image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_height, image_width = image.shape[:2]
    tilt_angle_rad = math.radians(TILT_ANGLE_DEGREES)
    
    # Prepare points for undistortion (must be shape (N, 1, 2))
    pts = np.array([[points[idx][0]], [points[idx][1]]], dtype=np.float32)

    # Undistort points
    undistorted_pts = cv2.undistortPoints(pts, K, dist, P=K)  # back to pixel coords
    (x1, y1), (x2, y2) = undistorted_pts.reshape(2, 2)

    # Compute pixel dimensions
    pixel_width = abs(x2 - x1)
    pixel_height = abs(y2 - y1)

    # Distance to object in mm
    distance_mm = DISTANCE_TO_OBJECT_MM[idx]

    # Real-world size calculations
    real_width_mm = (pixel_width * SENSOR_WIDTH_MM * distance_mm) / (
        image_width * FOCAL_LENGTH_MM * math.cos(tilt_angle_rad))
    
    real_height_mm = (pixel_height * SENSOR_HEIGHT_MM * distance_mm) / (
        image_height * FOCAL_LENGTH_MM * math.cos(tilt_angle_rad))

    # Convert to cm
    real_width_cm = real_width_mm / 10
    real_height_cm = real_height_mm / 10

    # Output
    print(f"Object {idx+1}:")
    print(f"  Undistorted Pixel Width: {pixel_width:.2f}px")
    print(f"  Real-World Width: {real_width_cm:.2f} cm")
    print(f"  Undistorted Pixel Height: {pixel_height:.2f}px")
    print(f"  Real-World Height: {real_height_cm:.2f} cm")