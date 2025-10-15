"""
Low-Light and Noisy Data Generation Script
------------------------------------------
This script simulates real-world illumination degradation and sensor noise
for camouflaged object detection datasets (e.g., CAMO, COD10K, NC4K).

It applies multi-stage degradation:
  (1) Illumination reduction (linear + gamma)
  (2) Shadow region simulation
  (3) Color temperature distortion
  (4) Noise injection (Gaussian, Poisson, Speckle)
  (5) Optional Gaussian blur for low-light motion effect

Author: Group A & B
Date: 2025-10-15
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import random


# ==========================================================
# 1. Configuration
# ==========================================================
INPUT_DIR = "data/original_images"
OUTPUT_DIR = "data/degraded_images"
DEGRADED_LEVELS = ["mild", "medium", "severe"]

CONFIG = {
    "mild":   {"alpha": 0.6, "gamma": 1.5, "sigma": 0.01},
    "medium": {"alpha": 0.4, "gamma": 2.2, "sigma": 0.03},
    "severe": {"alpha": 0.2, "gamma": 3.0, "sigma": 0.06},
}


# ==========================================================
# 2. Degradation Components
# ==========================================================

def adjust_illumination(img, alpha=0.5, gamma=2.0):
    """
    Reduce brightness and adjust illumination via linear scaling and gamma correction.
    Args:
        img (np.ndarray): input RGB image [0-255]
        alpha (float): linear brightness scale
        gamma (float): non-linear gamma exponent
    """
    img = img.astype(np.float32) / 255.0
    img = np.clip(img * alpha, 0, 1)
    img = np.power(img, gamma)
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def add_shadow(img, intensity=0.5):
    """
    Simulate uneven lighting by adding a soft shadow mask.
    """
    h, w, _ = img.shape
    mask = np.ones((h, w), np.float32)
    # Random ellipse shadow
    cx, cy = random.randint(0, w), random.randint(0, h)
    axes = (random.randint(w//4, w//2), random.randint(h//4, h//2))
    angle = random.randint(0, 180)
    cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, (intensity,), -1)
    shadow = img.astype(np.float32) * mask[..., None]
    return np.clip(shadow, 0, 255).astype(np.uint8)


def color_temperature_shift(img, factor=1.0):
    """
    Apply slight color temperature shift (simulate tungsten or cold light).
    """
    b, g, r = cv2.split(img.astype(np.float32))
    # warm or cool tone
    shift = random.choice([-1, 1])
    if shift > 0:  # warmer
        r *= 1.0 + 0.1 * factor
        b *= 1.0 - 0.05 * factor
    else:          # cooler
        r *= 1.0 - 0.05 * factor
        b *= 1.0 + 0.1 * factor
    merged = cv2.merge([b, g, r])
    return np.clip(merged, 0, 255).astype(np.uint8)


# -----------------------------
# Noise simulation
# -----------------------------
def add_gaussian_noise(img, sigma=0.03):
    noise = np.random.normal(0, sigma, img.shape)
    out = np.clip(img / 255.0 + noise, 0, 1)
    return (out * 255).astype(np.uint8)

def add_poisson_noise(img):
    vals = len(np.unique(img))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(img * vals) / float(vals)
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_speckle_noise(img, sigma=0.03):
    noise = np.random.randn(*img.shape) * sigma
    out = img + img * noise
    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================================================
# 3. Degradation Pipeline
# ==========================================================
def degrade_image(img, level="medium"):
    """Combine multiple degradations to generate realistic low-light effects."""
    params = CONFIG[level]
    alpha, gamma, sigma = params["alpha"], params["gamma"], params["sigma"]

    # Step 1: Reduce illumination
    degraded = adjust_illumination(img, alpha=alpha, gamma=gamma)

    # Step 2: Add shadow region (probability 0.6)
    if random.random() < 0.6:
        degraded = add_shadow(degraded, intensity=random.uniform(0.4, 0.8))

    # Step 3: Slight color temperature shift (probability 0.5)
    if random.random() < 0.5:
        degraded = color_temperature_shift(degraded, factor=random.uniform(0.8, 1.2))

    # Step 4: Add random noise type
    noise_type = random.choice(["gaussian", "poisson", "speckle"])
    if noise_type == "gaussian":
        degraded = add_gaussian_noise(degraded, sigma)
    elif noise_type == "poisson":
        degraded = add_poisson_noise(degraded)
    else:
        degraded = add_speckle_noise(degraded, sigma)

    # Step 5: Optional motion blur (simulate camera shake)
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        degraded = cv2.GaussianBlur(degraded, (ksize, ksize), 0)

    return degraded


# ==========================================================
# 4. Dataset Generation
# ==========================================================
def generate_lowlight_dataset(input_dir, output_dir):
    """
    Create low-light + noisy dataset with 3 degradation levels.
    """
    os.makedirs(output_dir, exist_ok=True)
    img_files = [f for f in os.listdir(input_dir)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for level in DEGRADED_LEVELS:
        save_dir = os.path.join(output_dir, level)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Generating {level} degraded images...")

        for img_name in tqdm(img_files):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            degraded_img = degrade_image(img, level)
            out_path = os.path.join(save_dir, img_name)
            cv2.imwrite(out_path, degraded_img)

    print("Dataset generation completed.")
    print(f"Low-light images saved under: {output_dir}")


# ==========================================================
# 5. Main
# ==========================================================
if __name__ == "__main__":
    generate_lowlight_dataset(INPUT_DIR, OUTPUT_DIR)
