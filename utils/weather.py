"""Weather noise utilities — standalone helpers for observation augmentation.

These are called by ObservationBuilder but can also be used independently
for data augmentation or offline frame processing.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from env.schemas import WeatherCondition


def apply_weather(
    img: Image.Image,
    condition: WeatherCondition,
    rng: np.random.Generator,
) -> Image.Image:
    """Apply weather-specific visual degradation to a PIL image.

    Args:
        img:       Input RGB PIL image.
        condition: WeatherCondition enum value.
        rng:       Seeded numpy RNG for reproducible noise.

    Returns:
        Augmented PIL image (same size, RGB).
    """
    if condition == WeatherCondition.CLEAR:
        return img
    if condition == WeatherCondition.CLOUDY:
        return _fog_overlay(img, alpha=60)
    if condition == WeatherCondition.RAIN:
        img = _rain_overlay(img, rng)
        return img.filter(ImageFilter.GaussianBlur(radius=0.6))
    if condition == WeatherCondition.FOG:
        return _fog_overlay(img, alpha=145)
    if condition == WeatherCondition.NIGHT:
        return _darken(img, factor=0.35)
    return img


def _fog_overlay(img: Image.Image, alpha: int = 120) -> Image.Image:
    fog = Image.new("RGBA", img.size, (200, 200, 210, alpha))
    base = img.convert("RGBA")
    base.alpha_composite(fog)
    return base.convert("RGB")


def _rain_overlay(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    arr = np.array(img, dtype=np.float32)
    H, W = arr.shape[:2]
    n_drops = max(10, (H * W) // 300)
    ys = rng.integers(0, H, size=n_drops)
    xs = rng.integers(0, W, size=n_drops)
    lengths = rng.integers(4, 12, size=n_drops)
    for y, x, length in zip(ys, xs, lengths):
        for dy in range(int(length)):
            ny = min(y + dy, H - 1)
            arr[ny, x] = np.clip(arr[ny, x] + 60, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _darken(img: Image.Image, factor: float = 0.4) -> Image.Image:
    arr = (np.array(img, dtype=np.float32) * factor).astype(np.uint8)
    return Image.fromarray(arr)


def random_camera_blur(
    img: Image.Image,
    rng: np.random.Generator,
    max_radius: float = 1.5,
) -> Image.Image:
    """Apply random Gaussian blur to simulate camera defocus."""
    radius = float(rng.uniform(0.0, max_radius))
    if radius < 0.2:
        return img
    return img.filter(ImageFilter.GaussianBlur(radius=radius))
