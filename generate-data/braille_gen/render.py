"""Text layout (character / word spacing) and the base height-map render."""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import math
from typing import Optional
from .braille_text import BRAILLE_SPACE, text_to_braille
from .config import PipelineConfig
from .utils import _clip_quad_to_image, _fit_within_limits, rand, randi


# ===========================================================================
# TEXT LAYOUT (dynamic character/word spacing + per-word pixel offsets)
# ===========================================================================


def _measure_braille_word(
    braille_word: str,
    font: ImageFont.FreeTypeFont,
    char_spacing_px: float,
) -> float:
    if not braille_word:
        return 0.0
    advances = [float(font.getlength(ch)) for ch in braille_word]
    return sum(advances) + char_spacing_px * max(0, len(advances) - 1)


def layout_paragraph(
    text: str,
    font: ImageFont.FreeTypeFont,
    max_width: int,
    grade: int,
    char_spacing_scale: float = 1.0,
    word_spacing_scale: float = 1.0,
) -> tuple[list[list[dict]], str]:
    """Greedy wrap using exact custom cell advances; returns layout and translation backend."""
    # A fully dotted cell gives a stable reference advance for spacing controls.
    cell_advance = max(1.0, float(font.getlength("⠿")))
    char_spacing_px = (char_spacing_scale - 1.0) * cell_advance
    word_space_px = max(1.0, float(font.getlength(BRAILLE_SPACE)) * word_spacing_scale)

    lines_out: list[list[dict]] = []
    backend_used = "unknown"
    for paragraph in text.split("\n"):
        words_en = [w for w in paragraph.split() if w]
        if not words_en:
            lines_out.append([])
            continue

        translated = []
        for w in words_en:
            br, backend = text_to_braille(w, grade)
            backend_used = backend
            translated.append((w, br, _measure_braille_word(br, font, char_spacing_px)))

        current = []
        current_w = 0.0
        for en, br, bw in translated:
            candidate_w = bw if not current else current_w + word_space_px + bw
            if current and candidate_w > max_width:
                lines_out.append(current)
                current = []
                current_w = 0.0
            x0 = 0.0 if not current else current_w + word_space_px
            current.append(
                {
                    "word_en": en,
                    "word_br": br,
                    "x0": x0,
                    "x1": x0 + bw,
                    "char_spacing_px": char_spacing_px,
                }
            )
            current_w = x0 + bw
        lines_out.append(current)
    return lines_out, backend_used


# ===========================================================================
# STAGE 0 — BASE RENDER (height-map → Lambertian shading)
# ===========================================================================


def render_base(
    text: str,
    cfg: PipelineConfig,
    grade: int = 1,
    char_spacing_scale: float = 1.0,
    word_spacing_scale: float = 1.0,
    dot_scale: float = 1.0,
    ghost: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, list[dict], str]:
    """Render braille with dynamic spacing/dot size and return word-level quads.

    ghost (opt-in, None = legacy): flat-print style where every non-blank cell shows all six
    positions as tiny light dots and the cell's real dots are drawn large and dark on top.
    Keys: scale (ghost dot size vs real), paper / ghost_col / ink (BGR tuples), ghost_alpha."""

    def load_font(font_size: int) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype(cfg.font_path, font_size)
        except IOError:
            raise FileNotFoundError(f"Font not found: {cfg.font_path}")

    def measure_layout(font_size: int, line_spacing: int, pad: int):
        font = load_font(font_size)
        max_text_width = max(1, cfg.max_image_width - pad * 2)
        lines, backend = layout_paragraph(
            text,
            font,
            max_text_width,
            grade,
            char_spacing_scale,
            word_spacing_scale,
        )
        line_widths = [max((e["x1"] for e in line), default=0.0) for line in lines]
        ink_bbox = font.getbbox("⠿", anchor="la")
        ink_top, ink_bottom = float(ink_bbox[1]), float(ink_bbox[3])
        line_advance = max(1.0, float(font.getbbox("A")[3]) + line_spacing)
        text_w = max(line_widths, default=1.0)
        text_h = max(
            1.0, (max(1, len(lines)) - 1) * line_advance + (ink_bottom - ink_top)
        )
        w = max(1, int(math.ceil(text_w)) + pad * 2)
        h = max(1, int(math.ceil(text_h)) + pad * 2)
        origin = (float(pad), float(pad) - ink_top)
        return font, lines, backend, ink_bbox, line_advance, w, h, origin

    font_size = cfg.font_size
    line_spacing = cfg.line_spacing
    pad = cfg.padding
    font, layout_lines, backend, ink_bbox, line_advance, w, h, draw_origin = (
        measure_layout(font_size, line_spacing, pad)
    )

    fitted_w, fitted_h, scale = _fit_within_limits(
        w, h, cfg.max_image_width, cfg.max_image_height
    )
    if scale < 1.0:
        font_size = max(1, int(math.floor(font_size * scale)))
        line_spacing = max(0, int(math.floor(line_spacing * scale)))
        pad = max(1, int(math.floor(pad * scale)))
        font, layout_lines, backend, ink_bbox, line_advance, w, h, draw_origin = (
            measure_layout(font_size, line_spacing, pad)
        )
        fitted_w, fitted_h, _ = _fit_within_limits(
            w, h, cfg.max_image_width, cfg.max_image_height
        )

    font_scale = font_size / max(1, cfg.font_ref or cfg.font_size)
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    origin_x, origin_y = draw_origin

    # Draw each braille cell separately so character spacing is genuinely variable.
    for line_idx, line in enumerate(layout_lines):
        y = origin_y + line_idx * line_advance
        for entry in line:
            x = origin_x + entry["x0"]
            for ci, ch in enumerate(entry["word_br"]):
                draw.text((x, y), ch, fill=255, font=font, anchor="la")
                x += float(font.getlength(ch))
                if ci + 1 < len(entry["word_br"]):
                    x += entry["char_spacing_px"]

    if cfg.enable_dot_jitter:
        mask = _apply_dot_jitter(mask, font_scale=font_scale, dot_scale=dot_scale)

    blur_r = max(0.6, random.uniform(3, 6) * font_scale * cfg.dot_blur_scale)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_r))
    hmap = np.array(mask).astype(np.float32) / 255.0

    ink_left, ink_top, ink_right, ink_bottom = map(float, ink_bbox)
    jitter_margin = (
        max(1, int(round(2 * font_scale * max(1.0, dot_scale))))
        if cfg.enable_dot_jitter
        else 0
    )
    bbox_margin = float(math.ceil(jitter_margin + 1.25 * blur_r))

    word_records: list[dict] = []
    for line_idx, line in enumerate(layout_lines):
        line_origin_y = origin_y + line_idx * line_advance
        for entry in line:
            x0 = origin_x + entry["x0"] + ink_left - bbox_margin
            x1 = origin_x + entry["x1"] + bbox_margin
            y0 = line_origin_y + ink_top - bbox_margin
            y1 = line_origin_y + ink_bottom + bbox_margin
            quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
            word_records.append(
                {
                    "text": entry["word_en"],
                    "braille_text": entry["word_br"],
                    "line_idx": line_idx,
                    "quad": quad,
                    # extra (ignored by build_bbox_record): px the quad extends
                    # beyond the ink; the real-matching crop uses it
                    "bbox_margin": bbox_margin,
                }
            )

    if ghost is None:
        dzdx = cv2.Sobel(hmap, cv2.CV_64F, 1, 0, ksize=3)
        dzdy = cv2.Sobel(hmap, cv2.CV_64F, 0, 1, ksize=3)
        depth_mul = rand(2.5, 5.0)
        normals = np.dstack((-dzdx * depth_mul, -dzdy * depth_mul, np.ones_like(hmap)))
        norm = np.linalg.norm(normals, axis=2, keepdims=True)
        normals /= np.maximum(norm, 1e-7)

        light_dir = np.array([rand(-1.5, 1.5), rand(-1.5, 1.5), rand(0.3, 1.0)])
        light_dir /= np.linalg.norm(light_dir)
        diffuse = np.clip(np.sum(normals * light_dir, axis=2), 0, 1)
        surface_noise = np.random.normal(1.0, 0.08, hmap.shape)
        base_col = cfg.base_surface_color * np.clip(surface_noise, 0.75, 1.25)
        illumination = rand(0.3, 0.55) + diffuse * rand(0.6, 1.0)
        image = np.clip(base_col * illumination, 0, 255).astype(np.uint8)
    else:
        # all six positions of every non-blank cell, drawn small, under the real dots
        gmask = Image.new("L", (w, h), 0)
        gdraw = ImageDraw.Draw(gmask) if ghost.get("show_ghost", True) else None
        for line_idx, line in enumerate(layout_lines):
            y = origin_y + line_idx * line_advance
            for entry in line:
                x = origin_x + entry["x0"]
                for ci, ch in enumerate(entry["word_br"]):
                    if gdraw is not None and (ord(ch) - 0x2800) & 0x3F:
                        gdraw.text((x, y), "\u283f", fill=255, font=font, anchor="la")
                    x += float(font.getlength(ch))
                    if ci + 1 < len(entry["word_br"]):
                        x += entry["char_spacing_px"]
        gmask = _apply_dot_jitter(gmask, font_scale=font_scale, dot_scale=dot_scale * ghost["scale"])
        gmask = gmask.filter(ImageFilter.GaussianBlur(radius=max(0.5, 0.6 * blur_r)))
        g = (np.array(gmask).astype(np.float32) / 255.0)[..., None] * float(ghost["ghost_alpha"])
        r = np.clip(hmap * 1.15, 0, 1)
        if ghost.get("ring"):
            # hollow dots: keep only an outer ring of each dot (printed circle outlines)
            dot_r = 0.5 * 18.0 * font_scale * dot_scale
            k = max(3, int(round(2 * ghost["ring"] * dot_r)) | 1)
            inner = cv2.erode((r > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            inner = cv2.GaussianBlur(inner.astype(np.float32), (0, 0), max(0.5, 0.15 * dot_r))
            r = np.clip(r - inner, 0, 1)
        r = r[..., None]
        paper = np.array(ghost["paper"], np.float32) * np.clip(np.random.normal(1.0, 0.02, (h, w, 1)), 0.9, 1.1)
        out = paper * (1 - g) + np.array(ghost["ghost_col"], np.float32) * g
        out = out * (1 - r) + np.array(ghost["ink"], np.float32) * r
        image = np.clip(out, 0, 255).astype(np.uint8)
        hmap = np.zeros_like(hmap)   # flat print: no embossing shadows, no hmap-alpha compositing

    if image.shape[1] != fitted_w or image.shape[0] != fitted_h:
        sx = fitted_w / image.shape[1]
        sy = fitted_h / image.shape[0]
        image = cv2.resize(image, (fitted_w, fitted_h), interpolation=cv2.INTER_AREA)
        hmap = cv2.resize(hmap, (fitted_w, fitted_h), interpolation=cv2.INTER_AREA)
        for rec in word_records:
            rec["quad"][:, 0] *= sx
            rec["quad"][:, 1] *= sy

    final_h, final_w = image.shape[:2]
    for rec in word_records:
        rec["quad"] = _clip_quad_to_image(rec["quad"], final_w, final_h)
    return image, hmap, word_records, backend


def _apply_dot_jitter(
    mask: Image.Image,
    font_scale: float = 1.0,
    dot_scale: float = 1.0,
) -> Image.Image:
    """Jitter/reshape connected braille dots; dot_scale=1.0 is the old midpoint."""
    arr = np.array(mask)
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (arr > 30).astype(np.uint8), connectivity=8
    )
    jittered = np.zeros_like(arr)
    out_img = Image.fromarray(jittered)
    draw = ImageDraw.Draw(out_img)
    max_jitter = max(1, int(round(2 * font_scale)))

    for i in range(1, len(stats)):
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        w2, h2 = stats[i][2] / 2.0, stats[i][3] / 2.0
        if w2 < 1.5 or h2 < 1.5:
            continue
        # 1.0 reproduces approximately the prior dot size; random anisotropy remains.
        rx = max(2, int(round(w2 * dot_scale * rand(0.88, 1.12) * rand(0.90, 1.05))))
        ry = max(2, int(round(h2 * dot_scale * rand(0.88, 1.12) * rand(0.90, 1.05))))
        dx, dy = randi(-max_jitter, max_jitter), randi(-max_jitter, max_jitter)
        draw.ellipse(
            [(cx + dx - rx, cy + dy - ry), (cx + dx + rx, cy + dy + ry)],
            fill=randi(200, 255),
        )
    return out_img
