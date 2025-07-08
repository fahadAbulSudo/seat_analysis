import cv2
import os

# video_path = "/home/fahadabul/mask_rcnn_skyhub/Subtitles/seat.mp4"
# output_dir = "/home/fahadabul/mask_rcnn_skyhub/Subtitles/calibration_frames"
# os.makedirs(output_dir, exist_ok=True)

# cap = cv2.VideoCapture(video_path)
# frame_id = 0
# frame_interval = 10  # Capture every 10th frame

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     if frame_id % frame_interval == 0:
#         frame_path = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
#         cv2.imwrite(frame_path, frame)
#     frame_id += 1

# cap.release()

# directory = "/home/fahadabul/mask_rcnn_skyhub/Subtitles/caliberation_images"

# # Get list of files and filter out non-image files (optional: adjust extensions as needed)
# image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
# files = [f for f in os.listdir(directory) if f.lower().endswith(image_extensions)]

# # Sort the files to ensure consistent renaming
# files.sort()

# # Rename files to 1.jpg, 2.jpg, ...
# for i, filename in enumerate(files, start=1):
#     ext = os.path.splitext(filename)[1]  # Keep the original extension
#     new_name = f"{i}{ext}"
#     src = os.path.join(directory, filename)
#     dst = os.path.join(directory, new_name)
#     os.rename(src, dst)

# print("Renaming complete.")

import numpy as np

calibration_path = "/home/fahadabul/mask_rcnn_skyhub/Subtitles/calibration_data.npz"
data = np.load(calibration_path)

print("Keys in .npz file:")
print(data.files)
