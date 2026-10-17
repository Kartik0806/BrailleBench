"""Master per-sample pipeline (legacy mode)."""

import numpy as np
from typing import Optional
from .background import _has_backgrounds, compose_on_background
from .braille_text import _normalize_source_text
from .config import PipelineConfig
from .effects import (
    apply_chromatic_aberration,
    apply_color_temperature,
    apply_defocus_blur,
    apply_dot_shadows,
    apply_downscale_upscale,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_lens_distortion_tracked,
    apply_low_light,
    apply_motion_blur,
    apply_overexposure,
    apply_paper_texture,
    apply_perlin_illumination,
    apply_salt_pepper,
    apply_shadow_bands,
    apply_stains,
    apply_vignetting,
)
from .geometry import (
    apply_elastic_warp_tracked,
    apply_page_curl_tracked,
    apply_perspective_warp_tracked,
    apply_random_boundary_tracked,
    apply_random_resolution_tracked,
    apply_right_angle_rotation_tracked,
    apply_skew_tracked,
    apply_vertical_flip_tracked,
    choose_sample_render_plan,
)
from .render import render_base
from .utils import coin, ensure_3ch


# ===========================================================================
# MASTER PIPELINE
# ===========================================================================


def generate_sample(
    text: str, cfg: PipelineConfig, sample_idx: int = 0
) -> tuple[np.ndarray, list[dict], bool, Optional[str], dict]:
    """Run the full pipeline and keep every geometry-changing augmentation bbox-safe."""
    text = _normalize_source_text(text)
    if not text:
        raise ValueError("Text is empty after braille-safe normalization")

    plan = choose_sample_render_plan(cfg)
    img, hmap, word_records, braille_backend = render_base(
        text,
        cfg,
        grade=plan["braille_grade"],
        char_spacing_scale=plan["char_spacing_scale"],
        word_spacing_scale=plan["word_spacing_scale"],
        dot_scale=plan["dot_scale"],
    )
    quads = [rec["quad"] for rec in word_records]
    meta = dict(plan)
    meta["braille_backend"] = braille_backend
    meta["vertical_flip"] = False
    meta["rotation_angle"] = 0
    meta["random_boundary"] = None
    meta["random_resolution"] = None

    img, quads, bg_used, bg_source = _augment_rendered(
        img, hmap, quads, cfg, plan, meta
    )
    for rec, q in zip(word_records, quads):
        rec["quad"] = q
    return img, word_records, bg_used, bg_source, meta


def _augment_rendered(img, hmap, quads, cfg, plan, meta):
    """Stages 2-7 on a rendered base image (shared by the legacy and the
    real-matching paths; pure code motion out of generate_sample, same RNG order)."""
    if cfg.enable_paper_texture and (
        cfg.p_paper_texture >= 1 or coin(cfg.p_paper_texture)
    ):
        img = apply_paper_texture(img)
    img = apply_perlin_illumination(img)
    img = apply_dot_shadows(img, hmap, cfg.font_size / max(1, cfg.font_ref or cfg.font_size))
    if coin(cfg.p_stain):
        img = apply_stains(img)

    bg_used = False
    bg_source = None
    if cfg.enable_background_compose and _has_backgrounds(cfg.background_dir):
        if coin(cfg.p_background_compose):
            img, bg_source = compose_on_background(img, hmap, cfg.background_dir)
            bg_used = bg_source is not None
    img = ensure_3ch(img)

    if cfg.enable_perspective_warp and coin(cfg.p_perspective):
        img, quads = apply_perspective_warp_tracked(img, quads)
    if cfg.enable_page_curl and coin(cfg.p_page_curl):
        img, quads = apply_page_curl_tracked(img, quads)
    if cfg.enable_elastic_warp and coin(cfg.p_elastic):
        img, quads = apply_elastic_warp_tracked(img, quads)
    if cfg.enable_skew and coin(cfg.p_skew):
        # This is an in-plane rotation, i.e. rotation about the camera/page z-axis.
        img, quads = apply_skew_tracked(
            img, quads, cfg.max_skew_deg, boundary_mode=cfg.skew_boundary_mode
        )

    if cfg.enable_vertical_flip and coin(cfg.p_vertical_flip):
        img, quads = apply_vertical_flip_tracked(img, quads)
        meta["vertical_flip"] = True

    if (
        cfg.enable_right_angle_rotation
        and cfg.rotation_angle != 0
        and coin(cfg.p_rotation)
    ):
        img, quads = apply_right_angle_rotation_tracked(img, quads, cfg.rotation_angle)
        meta["rotation_angle"] = int(cfg.rotation_angle) % 360

    if cfg.enable_camera_effects:
        if coin(cfg.p_defocus):
            img = apply_defocus_blur(img)
        if coin(cfg.p_motion_blur):
            img = apply_motion_blur(img)
        if coin():
            img, quads = apply_lens_distortion_tracked(img, quads)
        if coin(cfg.p_chromatic):
            img = apply_chromatic_aberration(img)
        if coin(cfg.p_vignette):
            img = apply_vignetting(img)

    if cfg.enable_lighting and (cfg.p_lighting >= 1 or coin(cfg.p_lighting)):
        if coin(0.6):
            img = apply_color_temperature(img)
        if coin(cfg.p_low_light):
            img = apply_low_light(img)
        elif coin(cfg.p_overexposure):
            img = apply_overexposure(img)
        if coin(cfg.p_shadow_band):
            img = apply_shadow_bands(img)

    if cfg.enable_noise and (cfg.p_noise >= 1 or coin(cfg.p_noise)):
        if coin(0.5):
            img = apply_downscale_upscale(img)
        if coin(cfg.p_salt_pepper):
            img = apply_salt_pepper(img)
        img = apply_gaussian_noise(img)
        img = apply_jpeg_compression(img)

    # Colored outer boundary is padding, not an overlay, so it never hides text.
    if cfg.enable_random_boundary and coin(cfg.p_random_boundary):
        img, quads, boundary_meta = apply_random_boundary_tracked(
            img, quads, cfg.boundary_min_px, cfg.boundary_max_px
        )
        meta["random_boundary"] = boundary_meta

    # About 30% by default. The target was chosen before rendering so dot size
    # could be increased for planned low-resolution samples.
    if plan["resolution_target"] is not None:
        target = plan["resolution_target"]
        img, quads, res_meta = apply_random_resolution_tracked(
            img, quads, target["width"], target["height"]
        )
        meta["random_resolution"] = res_meta

    return img, quads, bg_used, bg_source
