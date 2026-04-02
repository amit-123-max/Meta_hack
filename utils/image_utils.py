"""Image utility helpers — frame conversion and encoding."""
from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image


def frame_to_png_bytes(frame: np.ndarray) -> bytes:
    """Convert (H, W, 3) uint8 array to PNG bytes."""
    img = Image.fromarray(frame.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert (H, W, 3) uint8 array to base64-encoded PNG string."""
    return base64.b64encode(frame_to_png_bytes(frame)).decode("utf-8")


def resize_frame(frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize frame to (H, W) using nearest-neighbour (fast, no aliasing)."""
    H, W = size
    img = Image.fromarray(frame.astype(np.uint8))
    img = img.resize((W, H), Image.NEAREST)
    return np.array(img, dtype=np.uint8)


def stack_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Stack a list of (H, W, 3) arrays → (T, H, W, 3)."""
    return np.stack(frames, axis=0)


def normalise_frame(frame: np.ndarray) -> np.ndarray:
    """Normalise uint8 (H, W, 3) → float32 [0, 1]."""
    return frame.astype(np.float32) / 255.0
