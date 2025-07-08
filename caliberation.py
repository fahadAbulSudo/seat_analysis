import numpy as np
import cv2
import glob
import os

chessboard_size = (8, 6)
square_size = 21.0  # in mm

# 3D object points template
objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
objp *= square_size

objpoints = []  # 3D points
imgpoints = []  # 2D points

image_paths = glob.glob('/home/fahadabul/mask_rcnn_skyhub/Subtitles/caliberation_images/*.jpg')

for fname in image_paths:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Calibration', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()
print(f"Used {len(objpoints)} valid images for calibration.")

# ---------- Step 1: Calibrate Camera ----------
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("Camera Matrix (K):\n", K)
print("Distortion Coefficients:\n", dist)

# ---------- Step 2: Undistort One Image (Optional) ----------
h, w = gray.shape[:2]
new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(img, K, dist, None, new_K)
cv2.imshow("Undistorted", undistorted)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# ---------- Step 3: Save Calibration ----------
output_path = "/home/fahadabul/mask_rcnn_skyhub/Subtitles/calibration_data.npz"
np.savez(output_path, K=K, dist=dist, new_K=new_K, rvecs=rvecs, tvecs=tvecs)
print(f"Calibration saved to: {output_path}")
