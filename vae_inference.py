import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ======= Configuration =======
decoder_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/vae/model_jul16/vae_decoder.h5"
output_dir = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/vae/images/july_16"
num_samples = 10
# =============================

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the decoder model
decoder = load_model(decoder_path, compile=False)

# Get the latent dimension from decoder input shape
latent_dim = decoder.input_shape[1]  # Example: 32

# Generate random latent vectors from standard normal distribution
random_latent_vectors = np.random.normal(size=(num_samples, latent_dim))

# Decode the latent vectors to generate images
generated_images = decoder.predict(random_latent_vectors)

# Save each image
for i, img_array in enumerate(generated_images):
    # If grayscale, remove channel dimension
    if img_array.shape[-1] == 1:
        img_array = img_array.squeeze(axis=-1)

    # Rescale to 0–255 and convert to uint8
    img_uint8 = (img_array * 255).clip(0, 255).astype(np.uint8)

    # Convert to PIL image
    img_pil = Image.fromarray(img_uint8)

    # Convert to proper mode
    if img_pil.mode != "RGB" and len(img_uint8.shape) == 2:
        img_pil = img_pil.convert("L")
    elif img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")

    # Save the image
    img_path = os.path.join(output_dir, f"generated_{i+1:02d}.png")
    img_pil.save(img_path)

print(f"✅ Saved {num_samples} generated images to: {output_dir}")
