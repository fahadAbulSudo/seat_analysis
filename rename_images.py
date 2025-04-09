import os
import shutil

# Define source and destination directories
src_folder = "/home/fahadabul/mask_rcnn_skyhub/need"
dst_folder = "/home/fahadabul/mask_rcnn_skyhub/rename_green_seat"

# Ensure the destination folder exists
os.makedirs(dst_folder, exist_ok=True)

# List all image files in the source folder and sort them
image_files = sorted(os.listdir(src_folder))

# Rename and copy images
for i, filename in enumerate(image_files, start=991):
    old_path = os.path.join(src_folder, filename)
    new_filename = f"image{i}{os.path.splitext(filename)[1]}"  # Keep original extension
    new_path = os.path.join(dst_folder, new_filename)
    
    shutil.copy(old_path, new_path)  # Copy and rename

print(f"Renamed {len(image_files)} images successfully!")
