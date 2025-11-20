import cv2
import numpy as np
import os
import math
import torch
# --- Include your noise functions ---
# Ensure functions like generate_fractal_noise_2d are defined above this

# --- Load image ---
image_path = "/home/fahadabul/mask_rcnn_skyhub/latest_image_mask_rcnn_torn_wrinkle/output/test/5_20250605_042038.jpg"
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

if img is None:
    raise FileNotFoundError(f"Could not load image at {image_path}")

def lerp_np(x,y,w):
    fin_out = (y-x)*w + x
    return fin_out

def generate_fractal_noise_2d(shape, res, octaves=1, persistence=0.5):
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise

def generate_perlin_noise_2d(shape, res):
    def f(t):
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (math.ceil(shape[0] / res[0]), math.ceil(shape[1] / res[1]))

    grid_y, grid_x = np.meshgrid(
        np.linspace(0, res[0], shape[0], endpoint=False),
        np.linspace(0, res[1], shape[1], endpoint=False),
        indexing="ij"
    )
    grid = np.stack((grid_y % 1, grid_x % 1), axis=2)

    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    def repeat_grad(g):
        return g.repeat(d[0], axis=0).repeat(d[1], axis=1)[:shape[0], :shape[1]]

    g00 = repeat_grad(gradients[0:-1, 0:-1])
    g10 = repeat_grad(gradients[1:, 0:-1])
    g01 = repeat_grad(gradients[0:-1, 1:])
    g11 = repeat_grad(gradients[1:, 1:])

    # Ramps
    n00 = np.sum(grid * g00, axis=2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, axis=2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, axis=2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, axis=2)

    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
    dot = lambda grad, shift: (
                np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0]), torch.arange(0, res[1], delta[1])), dim=-1) % 1
    angles = 2 * math.pi * torch.rand(res[0] + 1, res[1] + 1)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0],
                                                                                                              0).repeat_interleave(
        d[1], 1)
    dot = lambda grad, shift: (
                torch.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                            dim=-1) * grad[:shape[0], :shape[1]]).sum(dim=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])

    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])


def rand_perlin_2d_octaves(shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(shape, (frequency * res[0], frequency * res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise

# Normalize and convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = img_rgb / 255.0  # Normalize to [0,1]

# --- Generate noise mask ---
shape = img_rgb.shape[:2]  # height, width
res = (4, 4)

noise_mask = generate_fractal_noise_2d(shape, res, octaves=4, persistence=0.5)

# Normalize noise
noise_mask = (noise_mask - noise_mask.min()) / (noise_mask.max() - noise_mask.min())

# Threshold to create defect mask
threshold = 0.5
binary_mask = (noise_mask > threshold).astype(np.float32)

# Apply the mask to darken regions
mask_rgb = np.stack([binary_mask]*3, axis=-1)
defect_img = img_rgb.copy()
defect_img[mask_rgb > 0] *= 0.3  # Darken masked region

# Convert everything back to [0,255] uint8
img_rgb_uint8 = (img_rgb * 255).astype(np.uint8)
defect_img_uint8 = (defect_img * 255).astype(np.uint8)
mask_gray_uint8 = (binary_mask * 255).astype(np.uint8)

# Convert RGB to BGR for OpenCV saving
img_bgr = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2BGR)
defect_bgr = cv2.cvtColor(defect_img_uint8, cv2.COLOR_RGB2BGR)

# Prepare output filenames
base_dir = os.path.dirname(image_path)
base_name = os.path.splitext(os.path.basename(image_path))[0]

cv2.imwrite(os.path.join(base_dir, f"{base_name}_original.jpg"), img_bgr)
cv2.imwrite(os.path.join(base_dir, f"{base_name}_defect_mask.jpg"), mask_gray_uint8)
cv2.imwrite(os.path.join(base_dir, f"{base_name}_with_defect.jpg"), defect_bgr)

print("Images saved to:", base_dir)
