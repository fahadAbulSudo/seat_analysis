import os
import zipfile

# Paths
source_folder = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/Inspector Pictures from Google Notes"
output_folder = os.path.join(source_folder, "images")

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all .docx files in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith(".docx"):
        docx_path = os.path.join(source_folder, filename)
        with zipfile.ZipFile(docx_path, 'r') as docx_zip:
            for file in docx_zip.namelist():
                if file.startswith("word/media/"):
                    image_data = docx_zip.read(file)
                    # Create a unique image filename
                    image_filename = f"{os.path.splitext(filename)[0]}_{os.path.basename(file)}"
                    image_path = os.path.join(output_folder, image_filename)
                    # Save image
                    with open(image_path, 'wb') as image_file:
                        image_file.write(image_data)
                    print(f"Extracted: {image_filename}")
