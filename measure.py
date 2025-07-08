import cv2
import numpy as np

# Constants
CALIBRATION_PATH = "/home/fahadabul/mask_rcnn_skyhub/Subtitles/calibration_data.npz"
IMAGE_PATH = "/home/fahadabul/mask_rcnn_skyhub/Subtitles/taper.jpg"
FOCAL_LENGTH_MM = 4.0
SENSOR_WIDTH_MM = 3.6
SENSOR_HEIGHT_MM = 4.5

# Tape corner pairs (x, y)
seat_tape_points = [
    [(3430, 5510), (3760, 5300)],  
    [(2680, 2780), (2990, 2490)],
    [(2300, 1797), (2530, 2260)],
]

# Corresponding Z-distances in mm (depths)
seat_distances_mm = [
    (600, 660),
    (1030, 1050),
    (1020, 1030)
]

# Load calibration
data = np.load(CALIBRATION_PATH)
K = data["K"]
dist = data["dist"]

# Load image (to get resolution)
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}")
image_height, image_width = image.shape[:2]

# Function to undistort and return pixel coordinates
def undistort_pixel(pixel, K, dist):
    pts = np.array([[pixel]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, K, dist, P=K)
    x_pix, y_pix = undistorted[0][0]
    return x_pix, y_pix

# Function to project undistorted pixel to real-world coordinate using pinhole formula
def project_pixel_to_real_world(x_pix, y_pix, Z, fx, fy, cx, cy):
    X = (x_pix) * Z
    Y = (y_pix) * Z
    return X, Y  # in mm

# Intrinsics
fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

# Process each tape
for i, ((pt1, pt2), (z1, z2)) in enumerate(zip(seat_tape_points, seat_distances_mm)):
    # Undistort points (get corrected pixel coords)
    x1_u, y1_u = undistort_pixel(pt1, K, dist)
    x2_u, y2_u = undistort_pixel(pt2, K, dist)

    # Back-project to real-world
    X1, Y1 = project_pixel_to_real_world(x1_u, y1_u, z1, fx, fy, cx, cy)
    X2, Y2 = project_pixel_to_real_world(x2_u, y2_u, z2, fx, fy, cx, cy)

    # Scale using sensor geometry
    dx = abs(X2 - X1) * SENSOR_WIDTH_MM / (image_width * FOCAL_LENGTH_MM)
    dy = abs(Y2 - Y1) * SENSOR_HEIGHT_MM / (image_height * FOCAL_LENGTH_MM)

    # Compute full length
    real_length_mm = np.sqrt(dx**2 + dy**2)
    real_length_cm = real_length_mm / 10

    print(f"Tape {i+1} (Seat Region):")
    print(f"  X-axis Length: {dx:.2f} mm")
    print(f"  Y-axis Length: {dy:.2f} mm")
    print(f"  Estimated Tape Length: {real_length_cm:.2f} cm\n")


# import cv2
# import numpy as np

# # Load calibration
# calib_data = np.load("/home/fahadabul/mask_rcnn_skyhub/Subtitles/calibration_data.npz")
# K = calib_data["K"]         # Intrinsic camera matrix
# dist = calib_data["dist"]   # Distortion coefficients

# # Original distorted points (Seat Tape Coordinates)
# seat_tape_points = [
#     [(3210, 5500), (3350, 5300), (3430, 5510), (3760, 5300)],  # tape 1
#     [(2770, 5180), (2850, 4980), (2300, 1760), (2530, 2260)]   # tape 2
# ]

# # Flatten the list of all points
# all_points = [pt for tape in seat_tape_points for pt in tape]
# pts_array = np.array(all_points, dtype=np.float32).reshape(-1, 1, 2)

# # Undistort points using cv2.undistortPoints (returns normalized coords),
# # then project back to pixel space using P=K to get pixel coordinates
# undistorted_pts = cv2.undistortPoints(pts_array, K, dist, P=K)
# undistorted_pts = undistorted_pts.reshape(-1, 2)

# # Output
# print("Undistorted Tape Points (in pixel coordinates):")
# for i, pt in enumerate(undistorted_pts):
#     print(f"Point {i+1}: ({pt[0]:.2f}, {pt[1]:.2f})")`
