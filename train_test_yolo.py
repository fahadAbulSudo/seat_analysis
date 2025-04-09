import os
import shutil
from sklearn.model_selection import train_test_split

# Define paths
image_dir = '/home/fahadabul/mask_rcnn_skyhub/segment/images'
label_dir = '/home/fahadabul/mask_rcnn_skyhub/segment/labels'
output_dir = '/home/fahadabul/mask_rcnn_skyhub/segment'
train_ratio = 0.8  

# Create output directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

# Get all image filenames
image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Split into train and validation sets
train_files, val_files = train_test_split(image_filenames, train_size=train_ratio, random_state=42)

# Function to move files
def move_files(file_list, split):
    for filename in file_list:
        # Move image
        shutil.copy(os.path.join(image_dir, filename), os.path.join(output_dir, 'images', split, filename))
        # Move corresponding label
        label_filename = filename.replace('.jpg', '.txt')
        shutil.copy(os.path.join(label_dir, label_filename), os.path.join(output_dir, 'labels', split, label_filename))

# Move files to respective directories
move_files(train_files, 'train')
move_files(val_files, 'val')
