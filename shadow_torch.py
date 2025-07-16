import torch
import segmentation_models_pytorch as smp
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

# Paths
model_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/shadow/best_model_0_5-epoch.pth"
image_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/MSN_12763_new/seat_11-06/PXL_20250611_092306682.jpg"

# Output mask save path
save_path = os.path.join(os.path.dirname(model_path), "predicted_shadow_mask.png")

# Config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESIZE_SHAPE = 256

# Load model
model = torch.load(model_path, map_location=DEVICE, weights_only=False)
model.eval()

# Define transform (same as during training)
transform = A.Compose([
    A.Resize(RESIZE_SHAPE, RESIZE_SHAPE),
    A.Normalize(),
    ToTensorV2()
])

# Load and preprocess input image
original_image = np.array(Image.open(image_path).convert("RGB"))
transformed = transform(image=original_image)
input_tensor = transformed["image"].unsqueeze(0).to(DEVICE)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    pred_mask = output.squeeze().cpu().numpy()
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Threshold to binary mask

# Resize mask back to original image size
pred_mask_resized = cv2.resize(pred_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

# Save the predicted mask
cv2.imwrite(save_path, pred_mask_resized * 255)  # save as binary mask (0 or 255)

print(f"✅ Predicted shadow mask saved at: {save_path}")
