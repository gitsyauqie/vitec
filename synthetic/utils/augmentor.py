"""
augmentor.py — Stochastic image augmentation pipeline for ITIEC-Syn.

Mimics real-world photo degradation: blur, noise, JPEG artefacts,
brightness shifts, perspective distortion, and random cropping.

All transforms are probabilistic — not every image gets every transform.
This controlled randomness prevents the model from learning augmentation
artefacts rather than text features.
"""

import random
import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

# Optional: albumentations for geometric transforms
try:
    import albumentations as A
    import cv2
    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False


# ── Individual transform functions ────────────────────────────────────────────

def gaussian_blur(img: Image.Image, sigma: float | None = None, p: float = 0.5, rng=None) -> Image.Image:
    """
    Apply Gaussian blur (σ ∈ [0.5, 2.5]).
    Simulates motion blur / out-of-focus camera.
    """
    rng = rng or random
    if rng.random() > p:
        return img
    sigma = sigma or rng.uniform(0.5, 2.5)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def gaussian_noise(img: Image.Image, std: float | None = None, p: float = 0.4, rng=None) -> Image.Image:
    """
    Add Gaussian noise (σ ∈ [5, 15]).
    Simulates sensor noise in low-light smartphone photos.
    """
    rng = rng or random
    if rng.random() > p:
        return img
    std = std or rng.uniform(5, 15)
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def jpeg_compression(img: Image.Image, quality: int | None = None, p: float = 0.6, rng=None) -> Image.Image:
    """
    Simulate JPEG compression (quality ∈ [55, 95]).
    Simulates WhatsApp-forwarded photo degradation — very common in
    Indonesian social media context.
    """
    rng = rng or random
    if rng.random() > p:
        return img
    quality = quality or rng.randint(55, 95)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def brightness_contrast_jitter(
    img: Image.Image,
    brightness_range: tuple = (0.7, 1.3),
    contrast_range: tuple = (0.8, 1.2),
    p: float = 0.5,
    rng=None,
) -> Image.Image:
    """
    Jitter brightness (±30%) and contrast (±20%).
    Simulates varying lighting conditions (sunlight, shade, indoor).
    """
    rng = rng or random
    if rng.random() > p:
        return img
    # Brightness
    b_factor = rng.uniform(*brightness_range)
    img = ImageEnhance.Brightness(img).enhance(b_factor)
    # Contrast
    c_factor = rng.uniform(*contrast_range)
    img = ImageEnhance.Contrast(img).enhance(c_factor)
    return img


def random_rotation(img: Image.Image, max_angle: float = 3.0, p: float = 0.3, rng=None) -> Image.Image:
    """
    Apply slight random rotation (±3°).
    Simulates imperfect phone alignment when photographing text.
    """
    rng = rng or random
    if rng.random() > p:
        return img
    angle = rng.uniform(-max_angle, max_angle)
    # expand=False keeps canvas size; fillcolor matches rough background
    bg = _estimate_background_color(img)
    return img.rotate(angle, expand=False, fillcolor=bg, resample=Image.BICUBIC)


def random_crop_pad(img: Image.Image, crop_fraction: float | None = None, p: float = 0.3, rng=None) -> Image.Image:
    """
    Crop to 90–100% of original area then pad back to original size.
    Simulates partial text capture / slight misframing.
    """
    rng = rng or random
    if rng.random() > p:
        return img
    crop_fraction = crop_fraction or rng.uniform(0.90, 1.0)
    w, h = img.size
    new_w = int(w * crop_fraction)
    new_h = int(h * crop_fraction)
    x0 = rng.randint(0, w - new_w)
    y0 = rng.randint(0, h - new_h)
    cropped = img.crop((x0, y0, x0 + new_w, y0 + new_h))
    # Pad back to original size
    bg = _estimate_background_color(img)
    padded = Image.new("RGB", (w, h), color=bg)
    padded.paste(cropped, (0, 0))
    return padded


def elastic_distortion(img: Image.Image, alpha: float = 30, sigma: float = 3, p: float = 0.2, rng=None) -> Image.Image:
    """
    Apply elastic distortion (α=30, σ=3).
    Simulates slight paper warping or screen reflection.
    Requires albumentations + opencv.
    """
    rng = rng or random
    if rng.random() > p or not _ALBUMENTATIONS_AVAILABLE:
        return img
    arr = np.array(img)
    transform = A.ElasticTransform(alpha=alpha, sigma=sigma, p=1.0)
    result = transform(image=arr)["image"]
    return Image.fromarray(result)


def perspective_warp(img: Image.Image, distortion: float = 0.05, p: float = 0.2, rng=None) -> Image.Image:
    """
    Apply slight perspective warp.
    Simulates photographing text at a slight angle.
    Requires albumentations + opencv.
    """
    rng = rng or random
    if rng.random() > p or not _ALBUMENTATIONS_AVAILABLE:
        return img
    arr = np.array(img)
    transform = A.Perspective(scale=(0.02, distortion), p=1.0)
    result = transform(image=arr)["image"]
    return Image.fromarray(result)


# ── Helper ────────────────────────────────────────────────────────────────────

def _estimate_background_color(img: Image.Image) -> tuple:
    """Estimate dominant background colour from image corners."""
    w, h = img.size
    corners = [
        img.getpixel((0, 0)),
        img.getpixel((w - 1, 0)),
        img.getpixel((0, h - 1)),
        img.getpixel((w - 1, h - 1)),
    ]
    r = int(np.mean([c[0] for c in corners]))
    g = int(np.mean([c[1] for c in corners]))
    b = int(np.mean([c[2] for c in corners]))
    return (r, g, b)


# ── Full pipeline ─────────────────────────────────────────────────────────────

def augment(
    img: Image.Image,
    seed: int | None = None,
    intensity: str = "medium",
) -> Image.Image:
    """
    Apply the full stochastic augmentation pipeline to *img*.

    Parameters
    ----------
    img : PIL.Image.Image
    seed : int, optional
        For reproducibility.
    intensity : str
        'light'  — gentle augmentation, preserve text legibility
        'medium' — balanced (default, matches paper design)
        'heavy'  — aggressive, stress-tests domain gap

    Returns
    -------
    PIL.Image.Image  (always RGB)
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    img = img.convert("RGB")

    # Probability multipliers per intensity
    mult = {"light": 0.4, "medium": 1.0, "heavy": 1.8}[intensity]
    m = min(mult, 1.0)  # cap individual probs at 1.0

    # Apply each transform
    img = gaussian_blur(img,             p=min(0.5 * mult, 0.9), rng=rng)
    img = gaussian_noise(img,            p=min(0.4 * mult, 0.85), rng=rng)
    img = jpeg_compression(img,          p=min(0.6 * mult, 0.95), rng=rng)
    img = brightness_contrast_jitter(img, p=min(0.5 * mult, 0.9), rng=rng)
    img = random_rotation(img,           p=min(0.3 * mult, 0.7), rng=rng)
    img = random_crop_pad(img,           p=min(0.3 * mult, 0.7), rng=rng)
    img = elastic_distortion(img,        p=min(0.2 * mult, 0.5), rng=rng)
    img = perspective_warp(img,          p=min(0.2 * mult, 0.5), rng=rng)

    return img


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from renderer import render_text_image
    import os

    out_dir = "/tmp/itiec_augment_test"
    os.makedirs(out_dir, exist_ok=True)

    test_texts = [
        "saya mw pergi ke rmh km skrng, bisa gk?",
        "gue bgt suka sama lagu itu, bgt2 keren banget deh",
    ]

    for i, txt in enumerate(test_texts):
        base = render_text_image(txt, seed=i * 100)
        base.save(os.path.join(out_dir, f"base_{i}.png"))
        for intensity in ["light", "medium", "heavy"]:
            aug = augment(base, seed=i * 10, intensity=intensity)
            aug.save(os.path.join(out_dir, f"aug_{i}_{intensity}.png"))

    print(f"Saved augmented images to {out_dir}")
    print(f"albumentations available: {_ALBUMENTATIONS_AVAILABLE}")
