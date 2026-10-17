"""Photometric effects: surface realism, camera optics, noise / degradation, lighting."""

import numpy as np
import cv2
from PIL import Image
import random
import io
import math
from .deps import HAS_NOISE, noise_lib
from .utils import _transform_quads_through_remap, coin, rand, randi


# ===========================================================================
# STAGE 2 — BACKGROUND & SURFACE REALISM
# ===========================================================================


def apply_paper_texture(img: np.ndarray) -> np.ndarray:
    """Overlay synthetic paper grain onto the image."""
    h, w = img.shape[:2]
    grain = np.random.normal(0, rand(4, 12), (h, w)).astype(np.float32)
    if img.ndim == 3:
        grain = grain[:, :, np.newaxis]
    result = np.clip(img.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    return result


def apply_perlin_illumination(img: np.ndarray) -> np.ndarray:
    """Spatially varying brightness map (Perlin noise or fallback cosine)."""
    h, w = img.shape[:2]
    illum = np.zeros((h, w), dtype=np.float32)

    if HAS_NOISE:
        scale = rand(0.005, 0.03)
        ox, oy = rand(0, 1000), rand(0, 1000)
        for y in range(h):
            for x in range(w):
                illum[y, x] = noise_lib.pnoise2(
                    ox + x * scale, oy + y * scale, octaves=4
                )
    else:
        # Fallback: sum of low-freq cosines
        for _ in range(3):
            fx = rand(0.5, 3) / w
            fy = rand(0.5, 3) / h
            px = rand(0, np.pi * 2)
            py = rand(0, np.pi * 2)
            xs = np.cos(2 * np.pi * fx * np.arange(w) + px)
            ys = np.cos(2 * np.pi * fy * np.arange(h) + py)
            illum += np.outer(ys, xs) / 3.0

    # Normalise to [0.75 .. 1.25] brightness range
    illum = (illum - illum.min()) / (illum.max() - illum.min() + 1e-7)
    illum = 0.75 + illum * 0.50  # [0.75, 1.25]

    if img.ndim == 3:
        illum = illum[:, :, np.newaxis]
    return np.clip(img.astype(np.float32) * illum, 0, 255).astype(np.uint8)


def apply_dot_shadows(img: np.ndarray, hmap: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Cast directional shadow from embossed dots (Stage 2). `scale` rescales the
    offsets when the base render is not at the reference font size (1.0 = legacy)."""
    sx = randi(1, 4) * random.choice([-1, 1])
    sy = randi(1, 4) * random.choice([-1, 1])
    if scale != 1.0:
        sx = int(math.copysign(max(1, round(abs(sx) * scale)), sx))
        sy = int(math.copysign(max(1, round(abs(sy) * scale)), sy))
    shadow = np.roll(np.roll(hmap, sy, axis=0), sx, axis=1).astype(np.float32)
    shadow = np.clip(shadow * rand(0.2, 0.5), 0, 1)
    if img.ndim == 3:
        shadow = shadow[:, :, np.newaxis]
    result = np.clip(img.astype(np.float32) - shadow * 80, 0, 255).astype(np.uint8)
    return result


def apply_stains(img: np.ndarray) -> np.ndarray:
    """Random semi-transparent stain blobs (coffee, fingerprints, aging)."""
    h, w = img.shape[:2]
    overlay = img.copy().astype(np.float32)
    n_stains = randi(1, 4)
    for _ in range(n_stains):
        cx, cy = randi(0, w), randi(0, h)
        rx = randi(10, 80)
        ry = randi(10, 60)
        stain_val = rand(100, 180)  # brownish-grey
        alpha = rand(0.03, 0.15)
        tmp = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(tmp, (cx, cy), (rx, ry), rand(0, 360), 0, 360, 1.0, -1)
        tmp = cv2.GaussianBlur(tmp, (0, 0), rand(5, 20))
        if overlay.ndim == 3:
            tmp3 = tmp[:, :, np.newaxis]
            overlay += tmp3 * (stain_val - overlay) * alpha
        else:
            overlay += tmp * (stain_val - overlay) * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ===========================================================================
# STAGE 4 — CAMERA & OPTICS SIMULATION
# ===========================================================================


def apply_defocus_blur(img: np.ndarray) -> np.ndarray:
    """Spatially varying Gaussian (sharp centre, blurred edges)."""
    h, w = img.shape[:2]
    max_sigma = rand(1.5, 4.5)
    cx, cy = w / 2, h / 2
    Y, X = np.mgrid[0:h, 0:w]
    r = np.sqrt(((X - cx) / (w / 2)) ** 2 + ((Y - cy) / (h / 2)) ** 2)
    r = np.clip(r, 0, 1)

    b1 = cv2.GaussianBlur(img, (0, 0), max_sigma * 0.33)
    b2 = cv2.GaussianBlur(img, (0, 0), max_sigma * 0.66)
    b3 = cv2.GaussianBlur(img, (0, 0), max_sigma)
    w1 = (
        np.clip(1 - r * 2, 0, 1)[:, :, np.newaxis]
        if img.ndim == 3
        else np.clip(1 - r * 2, 0, 1)
    )
    w2 = (
        np.clip(1 - np.abs(r - 0.5) * 2, 0, 1)[:, :, np.newaxis]
        if img.ndim == 3
        else np.clip(1 - np.abs(r - 0.5) * 2, 0, 1)
    )
    w3 = (
        np.clip((r - 0.5) * 2, 0, 1)[:, :, np.newaxis]
        if img.ndim == 3
        else np.clip((r - 0.5) * 2, 0, 1)
    )
    out = (
        b1.astype(np.float32) * w1
        + b2.astype(np.float32) * w2
        + b3.astype(np.float32) * w3
    )
    return out.astype(np.uint8)


def apply_motion_blur(img: np.ndarray) -> np.ndarray:
    """Small random linear kernel (camera shake)."""
    length = randi(2, 7)
    angle = rand(0, 360)
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    M = cv2.getRotationMatrix2D((length / 2, length / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    kernel /= kernel.sum() + 1e-7
    return cv2.filter2D(img, -1, kernel)


def apply_lens_distortion_tracked(
    img: np.ndarray, quads: list[np.ndarray]
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Barrel/pincushion distortion with bbox tracking."""
    h, w = img.shape[:2]
    k1 = rand(-0.3, 0.3)
    k2 = rand(-0.1, 0.1)
    cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, 0, 0, 0])
    map1, map2 = cv2.initUndistortRectifyMap(cam, dist, None, cam, (w, h), cv2.CV_32FC1)
    warped = cv2.remap(
        img, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    new_quads = _transform_quads_through_remap(quads, map1, map2)
    return warped, new_quads


def apply_lens_distortion(img: np.ndarray) -> np.ndarray:
    warped, _ = apply_lens_distortion_tracked(img, [])
    return warped


def apply_chromatic_aberration(img: np.ndarray) -> np.ndarray:
    """Shift R and B channels slightly in opposite directions."""
    if img.ndim == 2:
        return img
    shift = randi(1, 3)
    b, g, r = cv2.split(img)
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    return cv2.merge([b, g, r])


def apply_vignetting(img: np.ndarray) -> np.ndarray:
    """Dark radial gradient toward corners."""
    h, w = img.shape[:2]
    Y, X = np.mgrid[0:h, 0:w]
    cx, cy = w / 2, h / 2
    r = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    strength = rand(0.15, 0.5)
    vignette = 1 - strength * r**2
    vignette = np.clip(vignette, 0, 1)
    if img.ndim == 3:
        vignette = vignette[:, :, np.newaxis]
    return np.clip(img.astype(np.float32) * vignette, 0, 255).astype(np.uint8)


# ===========================================================================
# STAGE 5 — NOISE & DEGRADATION
# ===========================================================================


def apply_gaussian_noise(img: np.ndarray) -> np.ndarray:
    sigma = rand(2, 18)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_salt_pepper(img: np.ndarray) -> np.ndarray:
    # Fraction of *pixels* (H*W) hit by each of salt and pepper. Previously
    # rand(0.001, 0.008) * img.size, which counted channels too (3x as many
    # pixels, i.e. 0.3%-2.4% each); now 0.02%-0.15%.
    amount = rand(0.0002, 0.0015)
    out = img.copy()
    n = int(amount * img.shape[0] * img.shape[1])
    coords = [np.random.randint(0, i, n) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 255  # salt
    coords = [np.random.randint(0, i, n) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 0  # pepper
    return out


def apply_downscale_upscale(img: np.ndarray) -> np.ndarray:
    """Simulate cheap camera / scanner by downscaling then upscaling."""
    h, w = img.shape[:2]
    factor = rand(0.4, 0.75)
    small = cv2.resize(
        img,
        (max(1, int(w * factor)), max(1, int(h * factor))),
        interpolation=cv2.INTER_LINEAR,
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def apply_jpeg_compression(img: np.ndarray) -> np.ndarray:
    quality = randi(45, 88)
    pil_img = Image.fromarray(
        img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    )
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    result = np.array(Image.open(buf))
    if img.ndim == 2:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    else:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    return result


def apply_eraser_patches(img: np.ndarray) -> np.ndarray:
    """Occlude small rectangular regions (tape, torn edges, occlusion)."""
    h, w = img.shape[:2]
    n = randi(1, 4)
    out = img.copy()
    for _ in range(n):
        ph = randi(5, h // 6)
        pw = randi(5, w // 6)
        py = randi(0, h - ph)
        px = randi(0, w - pw)
        fill_val = randi(200, 255) if coin(0.7) else randi(0, 30)
        if out.ndim == 3:
            out[py : py + ph, px : px + pw] = fill_val
        else:
            out[py : py + ph, px : px + pw] = fill_val
    return out


# ===========================================================================
# STAGE 6 — LIGHTING CONDITIONS
# ===========================================================================


def apply_color_temperature(img: np.ndarray) -> np.ndarray:
    """Shift white balance toward warm (yellow) or cool (blue)."""
    if img.ndim == 2:
        return img
    b, g, r = cv2.split(img.astype(np.float32))
    mode = random.choice(["warm", "cool", "neutral"])
    if mode == "warm":
        r = np.clip(r * rand(1.0, 1.15), 0, 255)
        b = np.clip(b * rand(0.85, 1.0), 0, 255)
    elif mode == "cool":
        b = np.clip(b * rand(1.0, 1.15), 0, 255)
        r = np.clip(r * rand(0.85, 1.0), 0, 255)
    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])


def apply_low_light(img: np.ndarray) -> np.ndarray:
    """Reduce brightness and amplify noise to mimic indoor low-light shot."""
    factor = rand(0.35, 0.70)
    dark = np.clip(img.astype(np.float32) * factor, 0, 255)
    extra_noise = np.random.normal(0, rand(8, 20), img.shape).astype(np.float32)
    return np.clip(dark + extra_noise, 0, 255).astype(np.uint8)


def apply_overexposure(img: np.ndarray) -> np.ndarray:
    """Clip highlights in random edge patches (flash / window glare)."""
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)
    edge = random.choice(["top", "bottom", "left", "right"])
    span = randi(h // 8, h // 3)
    if edge == "top":
        region = out[:span, :]
    elif edge == "bottom":
        region = out[h - span :, :]
    elif edge == "left":
        region = out[:, :span]
    else:
        region = out[:, w - span :]
    glow = rand(1.2, 1.8)
    region[:] = np.clip(region * glow + rand(20, 80), 0, 255)
    return out.astype(np.uint8)


def apply_shadow_bands(img: np.ndarray) -> np.ndarray:
    """One or two soft linear gradient shadows across the image."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    n = randi(1, 2)
    for _ in range(n):
        axis = random.choice(["horizontal", "vertical"])
        strength = rand(0.2, 0.5)
        pos = rand(0.2, 0.8)
        width = rand(0.05, 0.25)
        if axis == "horizontal":
            ramp = np.abs(np.linspace(0, 1, h) - pos) / width
        else:
            ramp = np.abs(np.linspace(0, 1, w) - pos) / width
        band = np.clip(1 - (1 - ramp) * strength, 0.4, 1.0)
        if axis == "horizontal":
            band = band[:, np.newaxis]
            if out.ndim == 3:
                band = band[:, :, np.newaxis]
        else:
            band = band[np.newaxis, :]
            if out.ndim == 3:
                band = band[:, :, np.newaxis]
        out *= band
    return np.clip(out, 0, 255).astype(np.uint8)
