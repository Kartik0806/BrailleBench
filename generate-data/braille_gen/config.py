"""Pipeline configuration and the real-matching defaults."""

import os
from dataclasses import dataclass, field
from typing import Optional


# ===========================================================================
# CONFIG
# ===========================================================================


@dataclass
class PipelineConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    font_path: str = os.environ.get("BRAILLE_FONT_PATH", "/usr/share/fonts/truetype/freefont/FreeMono.ttf")
    background_dir: str = "data/background"
    output_dir: str = "data/synthetic"
    label_file: str = "labels.jsonl"

    # ── Render ─────────────────────────────────────────────────────────────
    font_size: int = 140
    line_spacing: int = 40
    padding: int = 100
    max_image_width: int = 1920
    max_image_height: int = 1080
    base_surface_color: int = 200

    # Randomly choose Grade-1 vs Grade-2 per generated sample.
    p_grade2: float = 0.5

    # Dynamic cell/word spacing. 1.0 means the original font advance.
    char_spacing_min: float = 0.90
    char_spacing_max: float = 1.18
    word_spacing_min: float = 0.85
    word_spacing_max: float = 1.55

    # Dot-size scale. 1.0 is the old/current midpoint. Low-resolution samples
    # are biased toward the larger end so dots remain visible after resizing.
    dot_scale_min: float = 0.78
    dot_scale_max: float = 1.38
    dot_scale_lowres_boost: float = 0.55

    # ── Augmentation toggle flags ──────────────────────────────────────────
    enable_dot_jitter: bool = True
    enable_paper_texture: bool = True
    enable_perspective_warp: bool = True
    enable_page_curl: bool = True
    enable_elastic_warp: bool = True
    enable_skew: bool = True
    enable_vertical_flip: bool = True
    enable_right_angle_rotation: bool = True
    enable_random_boundary: bool = True
    enable_random_resolution: bool = True
    enable_camera_effects: bool = False
    enable_noise: bool = True
    enable_lighting: bool = True
    enable_background_compose: bool = True

    # ── Per-stage probabilities (applied per sample) ───────────────────────
    p_perspective: float = 0.8
    p_page_curl: float = 0.5
    p_elastic: float = 0.6
    p_skew: float = 0.5
    max_skew_deg: float = 8.0  # in-plane rotation = rotation about camera z-axis
    p_vertical_flip: float = 0.15
    # Exact 0/90/180/270-degree image rotation. This is separate from a
    # vertical flip: 180° rotation flips BOTH x and y axes.
    p_rotation: float = 0.0
    rotation_angle: int = 0
    p_random_boundary: float = 0.45
    p_random_resolution: float = 0.30  # about 30 / 100 images
    p_defocus: float = 0.6
    p_motion_blur: float = 0.5
    p_chromatic: float = 0.5
    p_vignette: float = 0.7
    p_salt_pepper: float = 0.4
    p_eraser: float = 0.1
    p_stain: float = 0.4
    p_shadow_band: float = 0.4
    p_overexposure: float = 0.3
    p_low_light: float = 0.35
    p_background_compose: float = 0.5
    p_paper_texture: float = 1.0
    p_lighting: float = 1.0
    p_noise: float = 1.0
    p_shuffle_words: float = 0.0
    p_shuffle_chars: float = 0.0
    shuffle_char_word_fraction: float = 0.35

    # Actual low-resolution output bounds when p_random_resolution fires.
    min_width: int = 320
    max_width: int = 1280
    min_height: int = 180
    max_height: int = 720

    # Random colored outer boundary (padding, so text is never occluded).
    boundary_min_px: int = 4
    boundary_max_px: int = 36

    # ── Skew boundary handling ─────────────────────────────────────────────
    skew_boundary_mode: str = "expand"

    # ── Real-image matching (opt-in). None = legacy behaviour, byte for byte.
    # Reference font size for the size-relative effects in render_base (dot jitter,
    # blur, bbox margin). 0 = cfg.font_size, i.e. the legacy behaviour.
    font_ref: int = 0
    # Multiplier on the height-map blur radius in render_base (1.0 = legacy).
    dot_blur_scale: float = 1.0
    real_profile: Optional[dict] = None
    real_pool_weights: dict = field(default_factory=lambda: dict(REAL_POOL_WEIGHTS))
    real_opts: dict = field(default_factory=lambda: dict(REAL_DEFAULTS))
    real_seed: int = 0
    # opt-in text cleaning (see strip_punct_edge_tokens); False = legacy texts
    strip_punct_tokens: bool = False


# Tunables of the real-matching mode (all overridable with --real_opt key=value).
REAL_POOL_WEIGHTS = {"eval_combined": 0.4, "new_data": 0.4, "book_pages": 0.2}
REAL_DEFAULTS = {
    # in-word cell tightness R = cell pitch / dot pitch (measured physical value)
    # (the estimator reads ~0.15-0.2 below the true R on warped synthetic crops, so the
    #  standard mode sits above the 2.5 of standard braille: measured R then matches real)
    "p_R_standard": 0.60, "R_standard_mean": 2.65, "R_standard_sd": 0.15,
    "R_min": 2.15, "R_max": 3.3,
    # x/y squeeze of the whole image (oblique photo / non-uniform resize), lognormal sigma
    "aniso_sigma": 0.10, "aniso_clip": 1.45,
    # word gap b (blank advance in cell pitches): bank of real values, else uniform
    # gap_calib: bank values are *measured* gaps; the estimator reads ~+0.28 above the
    # true b on synthetic crops, so the bank is shifted before use
    "p_gap_bank": 0.85, "gap_min": 0.32, "gap_max": 2.2, "gap_calib": -0.28, "gap_floor_lo": 0.15,
    # text structure
    "max_lines": 8, "p_single_line": 0.4, "max_single_cells": 48,
    # jitter on resampled real records and independent "exploration" draws
    "jitter_sigma": 0.08, "p_explore": 0.10,
    # exploration = a real record of the same line-count class with much wider
    # jitter, so the tails are covered without impossible combinations
    "explore_sigma_P": 0.45, "explore_sigma_m": 0.5, "explore_sigma_dr": 0.25,
    "P_min": 5.0, "P_max": 110.0,
    # post-resize degradation at the final (low) resolution
    "p_blur": 0.5, "blur_max_dp": 0.22, "p_noise": 0.4, "noise_max": 6.0,
    "p_jpeg": 0.5, "jpeg_q_min": 30, "jpeg_q_max": 90,
    # smallest allowed base font (px) and padding (cell pitches) of the working canvas
    "base_font_min": 72, "canvas_pad": 3.8,
    # dot size: target ratio multiplier and height-map blur multiplier (calibration knobs)
    "dot_ratio_calib": 0.9, "dot_blur_scale": 0.3,
    # a failing sample is re-drawn with a fresh seed up to this many times (exact label counts)
    "retries": 6,
    # no clipped dots: every margin >= margin_floor cells (re-drawn in [floor, floor+spread]),
    # measured from the dot edges padded by max(bbox_margin, crop_safety * cell pitch)
    "margin_floor": 0.08, "margin_floor_spread": 0.15, "crop_safety": 0.06,
    # readable long texts: (min cells in the text, minimum cell pitch in px), ascending
    "long_P_floor": ((16, 18.0), (32, 22.0)),
    # size budget of the FINAL image (applied last, after the long-text pitch floor):
    # longest side <= max_side px and area <= max_pixels px. Qwen3.5 spends one visual token
    # per 32x32 px, so 1024*1024 px ~ 1024 visual tokens (real eval: p99 1216, max 1680).
    "max_side": 2048, "max_pixels": 1024 * 1024, "max_visual_tokens": 1024,
    # ---- optional styles (all 0 = off; the 200k realmatch set was generated with them off) ----
    # ghost cells: flat print, six tiny light dots per cell + large dark real dots (texts <= ghost_max_words)
    "p_ghost": 0.0, "ghost_max_words": 4, "ghost_scale": (0.22, 0.45),
    # hollow dots: flat print with ring-shaped dots (ring keeps the outer (1 - ring_inner) of the radius)
    "p_hollow": 0.0, "hollow_inner": (0.45, 0.75), "p_hollow_ghost": 0.3,
    # dynamic spacing: instead of the real-image bank, draw cell tightness R, word gap b and line
    # pitch L uniformly from wide ranges (per sample, with probability p_dynamic_spacing)
    "p_dynamic_spacing": 0.0, "dyn_R": (2.1, 3.8), "dyn_b": (0.3, 2.5), "dyn_L": (1.5, 3.5),
    # tape strips: every line on a coloured (Dymo-like) tape strip over a page / photo background
    "p_strip": 0.0,
    # close-ups: big cells for short texts, often as a blurry low-res upscale (phone crops)
    "p_closeup": 0.0, "closeup_max_cells": 10, "closeup_P": (60.0, 140.0), "closeup_lowres_p": 0.5,
    # reject (and retry) near-uniform images: nothing readable is left in them
    "min_image_std": 4.0,
    # additive correction of the framing margins (measured margins run slightly low)
    "margin_offset": (0.04, 0.0, 0.10, 0.075),
}


CFG = PipelineConfig()
