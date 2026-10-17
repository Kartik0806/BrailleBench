"""Geometric augmentations with bounding-box tracking, and the per-sample render plan."""

import numpy as np
import cv2
import random
import math
from .config import PipelineConfig
from .utils import (
    _clip_quad_to_image,
    _order_quad,
    _transform_quads_through_remap,
    coin,
    ensure_3ch,
    rand,
    randi,
)


# ===========================================================================
# STAGE 3 — GEOMETRIC & PERSPECTIVE REALISM
# (each has a "_tracked" variant that also updates word quads)
# ===========================================================================


def apply_perspective_warp(img: np.ndarray, max_tilt_deg: float = 12.0) -> np.ndarray:
    h, w = img.shape[:2]
    margin = int(min(h, w) * rand(0.02, 0.08))
    pts_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    pts_dst = pts_src + np.random.uniform(-margin, margin, pts_src.shape).astype(
        np.float32
    )
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    crop = margin
    warped = warped[crop : h - crop, crop : w - crop]
    warped = cv2.resize(warped, (w, h))
    return warped


def apply_perspective_warp_tracked(
    img: np.ndarray, quads: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = img.shape[:2]
    margin = int(min(h, w) * rand(0.02, 0.08))
    pts_src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    pts_dst = pts_src + np.random.uniform(-margin, margin, pts_src.shape).astype(
        np.float32
    )
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    crop = margin
    if crop > 0 and h > 2 * crop and w > 2 * crop:
        warped = warped[crop : h - crop, crop : w - crop]
        cropped_h, cropped_w = warped.shape[:2]
        warped = cv2.resize(warped, (w, h))
        sx = w / cropped_w
        sy = h / cropped_h
    else:
        crop = 0
        sx = sy = 1.0

    new_quads = []
    for q in quads:
        pts = q.reshape(-1, 1, 2).astype(np.float32)
        pts_t = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        # Match OpenCV resize's pixel-centre convention.
        pts_t[:, 0] = (pts_t[:, 0] - crop + 0.5) * sx - 0.5
        pts_t[:, 1] = (pts_t[:, 1] - crop + 0.5) * sy - 0.5
        new_quads.append(_clip_quad_to_image(_order_quad(pts_t), w, h))
    return warped, new_quads


def apply_page_curl(img: np.ndarray) -> np.ndarray:
    """Sinusoidal warp along one axis (page curl simulation)."""
    h, w = img.shape[:2]
    amplitude = rand(3, 15)
    frequency = rand(0.3, 1.5)
    axis = random.choice(["x", "y"])

    map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)

    if axis == "x":
        map_x += amplitude * np.sin(2 * np.pi * frequency * map_y / h)
    else:
        map_y += amplitude * np.sin(2 * np.pi * frequency * map_x / w)

    return cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def apply_page_curl_tracked(
    img: np.ndarray, quads: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = img.shape[:2]
    amplitude = rand(3, 15)
    frequency = rand(0.3, 1.5)
    axis = random.choice(["x", "y"])

    map_x = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    map_y = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)
    if axis == "x":
        map_x = map_x + amplitude * np.sin(2 * np.pi * frequency * map_y / h)
    else:
        map_y = map_y + amplitude * np.sin(2 * np.pi * frequency * map_x / w)

    warped = cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    new_quads = _transform_quads_through_remap(quads, map_x, map_y)
    return warped, new_quads


def apply_elastic_warp(img: np.ndarray) -> np.ndarray:
    """Low-res random displacement field → elastic deformation."""
    h, w = img.shape[:2]
    alpha = rand(5, 20)
    sigma = rand(8, 20)

    dx = np.random.randn(h, w).astype(np.float32) * alpha
    dy = np.random.randn(h, w).astype(np.float32) * alpha
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)

    map_x = (np.tile(np.arange(w), (h, 1)) + dx).astype(np.float32)
    map_y = (np.tile(np.arange(h), (w, 1)).T + dy).astype(np.float32)

    return cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )


def apply_elastic_warp_tracked(
    img: np.ndarray, quads: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = img.shape[:2]
    alpha = rand(5, 20)
    sigma = rand(8, 20)

    dx = np.random.randn(h, w).astype(np.float32) * alpha
    dy = np.random.randn(h, w).astype(np.float32) * alpha
    dx = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy = cv2.GaussianBlur(dy, (0, 0), sigma)

    map_x = (np.tile(np.arange(w), (h, 1)) + dx).astype(np.float32)
    map_y = (np.tile(np.arange(h), (w, 1)).T + dy).astype(np.float32)
    warped = cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    new_quads = _transform_quads_through_remap(quads, map_x, map_y)
    return warped, new_quads


def apply_skew(img: np.ndarray, max_deg: float = 8.0) -> np.ndarray:
    """Small bounded-angle rotation ("skew") in a random direction."""
    h, w = img.shape[:2]
    angle = rand(-max_deg, max_deg)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def apply_skew_tracked(
    img: np.ndarray,
    quads: list[np.ndarray],
    max_deg: float = 8.0,
    boundary_mode: str = "expand",
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Small bounded-angle rotation ("skew") in a random direction.

    boundary_mode:
      "expand" (default) — pads the canvas before rotating so the rotated
          corners of the *entire image* stay inside the frame. No pixels
          are pushed off-canvas and no word quad is clipped by this stage.
          The output image is slightly larger than the input.
      "crop" — legacy behaviour: rotate in place at the original canvas
          size. Content near the edges/corners can rotate outside the
          frame and be lost (and the corresponding word quad gets clipped
          to whatever sliver remains visible).
    """
    h, w = img.shape[:2]
    angle = rand(-max_deg, max_deg)

    if boundary_mode == "expand":
        # Padding needed so that rotating the full WxH canvas by `angle`
        # keeps every original corner inside the new, larger canvas.
        rad = np.deg2rad(abs(angle))
        pad_x = int(np.ceil(h * np.sin(rad)))
        pad_y = int(np.ceil(w * np.sin(rad)))
        pad_x = max(pad_x, 0)
        pad_y = max(pad_y, 0)

        padded = cv2.copyMakeBorder(
            img, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE
        )
        new_h, new_w = padded.shape[:2]
        center = ((new_w - 1) / 2, (new_h - 1) / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        warped = cv2.warpAffine(
            padded, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE
        )

        new_quads = []
        for q in quads:
            pts = q.reshape(-1, 1, 2).astype(np.float32)
            pts[:, 0, 0] += pad_x
            pts[:, 0, 1] += pad_y
            pts_t = cv2.transform(pts, M).reshape(-1, 2)
            new_quads.append(_clip_quad_to_image(_order_quad(pts_t), new_w, new_h))
        return warped, new_quads

    # boundary_mode == "crop" (legacy, may clip content)
    M = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
    warped = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    new_quads = []
    for q in quads:
        pts = q.reshape(-1, 1, 2).astype(np.float32)
        pts_t = cv2.transform(pts, M).reshape(-1, 2)
        new_quads.append(_clip_quad_to_image(_order_quad(pts_t), w, h))
    return warped, new_quads


def apply_vertical_flip_tracked(
    img: np.ndarray, quads: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Flip top↔bottom and track all word quads exactly."""
    h, w = img.shape[:2]
    flipped = cv2.flip(img, 0)
    new_quads = []
    for q in quads:
        pts = np.asarray(q, dtype=np.float32).copy()
        pts[:, 1] = (h - 1) - pts[:, 1]
        new_quads.append(_clip_quad_to_image(_order_quad(pts), w, h))
    return flipped, new_quads


def apply_right_angle_rotation_tracked(
    img: np.ndarray,
    quads: list[np.ndarray],
    angle: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Rotate by an exact multiple of 90° clockwise and track word quads.

    Supported values are 0, 90, 180, and 270. Right-angle rotations use
    ``cv2.rotate`` so there is no resampling/interpolation blur.
    """
    angle = int(angle) % 360
    if angle not in (0, 90, 180, 270):
        raise ValueError(f"rotation angle must be one of 0, 90, 180, 270; got {angle}")

    h, w = img.shape[:2]
    if angle == 0:
        return img, [np.asarray(q, dtype=np.float32).copy() for q in quads]
    if angle == 90:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        new_h, new_w = w, h
    elif angle == 180:
        rotated = cv2.rotate(img, cv2.ROTATE_180)
        new_h, new_w = h, w
    else:  # 270° clockwise == 90° counter-clockwise
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_h, new_w = w, h

    new_quads = []
    for q in quads:
        pts = np.asarray(q, dtype=np.float32).copy()
        x = pts[:, 0].copy()
        y = pts[:, 1].copy()

        if angle == 90:
            pts[:, 0] = (h - 1) - y
            pts[:, 1] = x
        elif angle == 180:
            pts[:, 0] = (w - 1) - x
            pts[:, 1] = (h - 1) - y
        else:  # 270
            pts[:, 0] = y
            pts[:, 1] = (w - 1) - x

        new_quads.append(_clip_quad_to_image(_order_quad(pts), new_w, new_h))

    return rotated, new_quads


def apply_random_boundary_tracked(
    img: np.ndarray,
    quads: list[np.ndarray],
    min_px: int = 4,
    max_px: int = 36,
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Add a random-color outer border and translate bboxes with the padding."""
    top = randi(min_px, max_px)
    bottom = randi(min_px, max_px)
    left = randi(min_px, max_px)
    right = randi(min_px, max_px)
    color = tuple(randi(0, 255) for _ in range(3))  # BGR for OpenCV
    out = cv2.copyMakeBorder(
        ensure_3ch(img),
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )
    h2, w2 = out.shape[:2]
    new_quads = []
    for q in quads:
        pts = np.asarray(q, dtype=np.float32).copy()
        pts[:, 0] += left
        pts[:, 1] += top
        new_quads.append(_clip_quad_to_image(_order_quad(pts), w2, h2))
    meta = {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "bgr": list(color),
    }
    return out, new_quads, meta


def apply_random_resolution_tracked(
    img: np.ndarray,
    quads: list[np.ndarray],
    target_width: int,
    target_height: int,
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Reduce image resolution to fit inside a random target box, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(target_width / max(w, 1), target_height / max(h, 1), 1.0)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return (
            img,
            quads,
            {
                "requested_width": target_width,
                "requested_height": target_height,
                "applied": False,
                "output_width": w,
                "output_height": h,
            },
        )
    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    sx, sy = new_w / w, new_h / h
    new_quads = []
    for q in quads:
        pts = np.asarray(q, dtype=np.float32).copy()
        pts[:, 0] *= sx
        pts[:, 1] *= sy
        new_quads.append(_clip_quad_to_image(_order_quad(pts), new_w, new_h))
    return (
        out,
        new_quads,
        {
            "requested_width": target_width,
            "requested_height": target_height,
            "applied": True,
            "output_width": new_w,
            "output_height": new_h,
            "scale_x": sx,
            "scale_y": sy,
        },
    )


def choose_sample_render_plan(cfg: PipelineConfig) -> dict:
    """Choose coupled render controls before Stage 0 (grade, spacing, dots, low-res target)."""
    grade = 2 if coin(cfg.p_grade2) else 1
    char_spacing_scale = rand(cfg.char_spacing_min, cfg.char_spacing_max)
    word_spacing_scale = rand(cfg.word_spacing_min, cfg.word_spacing_max)

    resolution_target = None
    if cfg.enable_random_resolution and coin(cfg.p_random_resolution):
        resolution_target = {
            "width": randi(
                min(cfg.min_width, cfg.max_width), max(cfg.min_width, cfg.max_width)
            ),
            "height": randi(
                min(cfg.min_height, cfg.max_height), max(cfg.min_height, cfg.max_height)
            ),
        }

    # 1.0 is the old midpoint. If this sample is planned to be small, bias dots larger.
    if resolution_target is None:
        dot_scale = rand(max(cfg.dot_scale_min, 0.92), min(cfg.dot_scale_max, 1.08))
    else:
        wr = resolution_target["width"] / max(1, cfg.max_image_width)
        hr = resolution_target["height"] / max(1, cfg.max_image_height)
        lowres_ratio = min(1.0, max(0.0, math.sqrt(max(0.0, wr * hr))))
        midpoint = 1.0 + cfg.dot_scale_lowres_boost * (1.0 - lowres_ratio)
        dot_scale = midpoint + rand(-0.08, 0.08)
    dot_scale = float(np.clip(dot_scale, cfg.dot_scale_min, cfg.dot_scale_max))

    return {
        "braille_grade": grade,
        "char_spacing_scale": char_spacing_scale,
        "word_spacing_scale": word_spacing_scale,
        "dot_scale": dot_scale,
        "resolution_target": resolution_target,
    }
