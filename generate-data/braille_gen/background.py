"""Composition on real background images."""

import numpy as np
import cv2
import random
import os
import glob
from typing import Optional
from .utils import ensure_3ch, rand


# ===========================================================================
# STAGE 7 — COMPOSITION ON REAL BACKGROUND (SynthText-style)
# ===========================================================================

_background_cache: list = []
_bg_warning_shown = False


def _load_backgrounds(bg_dir: str) -> list:
    global _background_cache
    if _background_cache:
        return _background_cache
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(os.path.join(bg_dir, p)))
        paths.extend(glob.glob(os.path.join(bg_dir, "**", p), recursive=True))
    _background_cache = paths
    return paths


def _has_backgrounds(bg_dir: str) -> bool:
    """
    True iff `bg_dir` exists and contains at least one usable image. Used to
    gate Stage 7 so that it is skipped entirely (not silently "composed
    onto nothing") when no background images are available — instead of
    calling compose_on_background() every time and having it quietly
    no-op.
    """
    global _bg_warning_shown
    paths = _load_backgrounds(bg_dir)
    if not paths and not _bg_warning_shown:
        print(
            f"⚠ No background images found in '{bg_dir}'. "
            f"Stage 7 (background composition) will be skipped for all samples. "
            f"Drop images in that folder to enable it."
        )
        _bg_warning_shown = True
    return len(paths) > 0


def compose_on_background(
    img: np.ndarray, hmap: np.ndarray, bg_dir: str
) -> tuple[np.ndarray, Optional[str]]:
    """
    Composite the braille render onto a real background image using
    multiply blend, calibrated by local luminance (SynthText strategy).

    Returns (composited_image, background_filename_or_None). Caller is
    expected to have already checked `_has_backgrounds(bg_dir)`.
    """
    paths = _load_backgrounds(bg_dir)
    if not paths:
        return img, None

    bg_path = random.choice(paths)
    try:
        bg = cv2.imread(bg_path)
        if bg is None:
            return img, None
    except Exception:
        return img, None

    h, w = img.shape[:2]
    bg = cv2.resize(bg, (w, h))

    render3 = ensure_3ch(img) if img.ndim == 2 else img.copy()
    bg3 = ensure_3ch(bg)

    render_f = render3.astype(np.float32) / 255.0
    bg_f = bg3.astype(np.float32) / 255.0

    hmap3 = np.dstack([hmap] * 3)

    alpha = hmap3 * rand(0.6, 0.95)
    result = bg_f * (1 - alpha) + render_f * alpha
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result, os.path.basename(bg_path)
