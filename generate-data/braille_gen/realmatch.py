"""Real-image matching mode (opt-in): sampling from a measured profile and rendering to match it."""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import json
import math
from typing import Optional
from .background import _load_backgrounds
from .braille_text import _normalize_source_text
from .config import PipelineConfig
from .pipeline import _augment_rendered
from .render import layout_paragraph, render_base
from .utils import _clip_quad_to_image, _order_quad, coin, rand, randi


# ===========================================================================
# REAL-IMAGE MATCHING MODE (opt-in, see module docstring)
# ===========================================================================
#
# Physical parameterisation (all measured on real photos, see src/realstats):
#   P   cell pitch in the FINAL image, px          (scale)
#   R   cell pitch / dot pitch inside a word        (cell tightness) -> char spacing
#   b   blank advance across a word boundary, in cell pitches        -> word spacing
#   dr  dot diameter / dot pitch                                     -> dot size
#   m   margins from the outermost dots to the image edge, in cell pitches
#   a   x/y squeeze of the final image
# Generator controls are derived exactly from the font geometry:
#   char_spacing_scale  cs = R * dot_pitch / cell_advance
#   word_spacing_scale  ws = cs * (1 + b) - 1     (start-to-start across a boundary
#                                                  is advance*(1+ws); in-word pitch is advance*cs)

_FONT_GEOM_CACHE: dict = {}


def font_geometry(font_path: str, size: int = 140) -> dict:
    """Measure the braille glyph geometry of `font_path` at `size` px by rendering it."""
    key = (font_path, size)
    if key in _FONT_GEOM_CACHE:
        return _FONT_GEOM_CACHE[key]
    font = ImageFont.truetype(font_path, size)
    adv = float(font.getlength("⠿"))
    ox, oy = size, size  # glyph origin ('la' anchor) inside a scratch canvas
    canvas = Image.new("L", (int(adv + 2 * size), int(3 * size)), 0)
    ImageDraw.Draw(canvas).text((ox, oy), "⠿", fill=255, font=font, anchor="la")
    arr = np.array(canvas)
    n, _, st, cen = cv2.connectedComponentsWithStats((arr > 30).astype(np.uint8), connectivity=8)
    dots = [(cen[i][0] - ox, cen[i][1] - oy, 0.5 * (st[i][2] + st[i][3]) / 2.0) for i in range(1, n)]
    if len(dots) != 6:
        raise RuntimeError(f"expected 6 dots in the full cell glyph of {font_path}, found {len(dots)}")
    xs = sorted({round(d[0], 1) for d in dots})
    ys = sorted({round(d[1], 1) for d in dots})
    col_x = [float(np.mean([d[0] for d in dots if abs(d[0] - x) < 2])) for x in xs[:1] + xs[-1:]]
    row_y = [float(np.mean([d[1] for d in dots if abs(d[1] - y) < 2])) for y in (ys[0], ys[len(ys) // 2], ys[-1])]
    geom = {
        "size": size, "adv": adv,
        "dot_pitch": float(np.mean([col_x[1] - col_x[0], row_y[1] - row_y[0], row_y[2] - row_y[1]])),
        "dot_diam": float(2 * np.mean([d[2] for d in dots])),
        "col_x": col_x, "row_y": row_y,
        "ink_left": float(font.getbbox("⠿", anchor="la")[0]),
        "ink_top": float(font.getbbox("⠿", anchor="la")[1]),
    }
    _FONT_GEOM_CACHE[key] = geom
    return geom


_REAL_SAMPLER_CACHE: dict = {}


class RealSampler:
    """Resamples the real style bank (profile JSON) with jitter / exploration."""

    def __init__(self, profile: dict, pool_weights: dict, opts: dict):
        self.o = opts
        self.pools = {}
        for name, recs in profile["pools"].items():
            w = float(pool_weights.get(name, 0.0))
            if w <= 0 or not recs:
                continue
            by: dict = {}
            for r in recs:
                cls = 1 if r["nl"] == 1 else 2 if r["nl"] <= 3 else 4
                cells = r.get("c", r["w"] * r["nl"])
                by.setdefault((self.cell_bin(cells), cls), []).append(r)
            self.pools[name] = (w, by)
        if not self.pools:
            raise ValueError("real profile has no usable pool for the given --real_pool_weights")
        gap = []
        for name, vals in profile.get("gap_b", {}).items():
            w = float(pool_weights.get(name, 0.0))
            gap.extend([(w, v) for v in vals if w > 0])
        self.gap = gap
        self.gap_w = [w for w, _ in gap]
        self.wrap = []
        for name, (w, by) in self.pools.items():
            for (bi, cls), lst in by.items():
                if cls != 1:
                    self.wrap.extend([r["w"] for r in lst])

    CELL_BIN_EDGES = (8, 16, 32, 64, 128)

    @classmethod
    def cell_bin(cls, cells: float) -> int:
        return sum(cells >= e for e in cls.CELL_BIN_EDGES)

    def pick_record(self, cls: int, n_cells: float = 8.0) -> dict:
        """A real style record with a similar line-count class AND cell count (real pitch
        depends on how much text is in the image: tight crops of one word vs dense pages)."""
        nb = len(self.CELL_BIN_EDGES) + 1
        b0 = self.cell_bin(n_cells)
        cls_order = {1: (1, 2, 4), 2: (2, 4, 1), 4: (4, 2, 1)}[cls]
        bin_order = sorted(range(nb), key=lambda b: (abs(b - b0), b))
        for bi in bin_order:
            for c in cls_order:
                cands = [(w, by[(bi, c)]) for (w, by) in self.pools.values() if by.get((bi, c))]
                if cands:
                    ws = [w for w, _ in cands]
                    return random.choice(random.choices([l for _, l in cands], weights=ws, k=1)[0])
        raise ValueError("empty style bank")

    def pick_gap(self) -> float:
        o = self.o
        if self.gap and random.random() < o["p_gap_bank"]:
            v = random.choices([v for _, v in self.gap], weights=self.gap_w, k=1)[0]
            v = v * math.exp(random.gauss(0, 0.10)) + o["gap_calib"]
        else:
            v = rand(o["gap_min"], o["gap_max"])
        if v < o["gap_min"]:
            # real data has many near-touching word gaps; spread them over a range instead of
            # piling every one of them on the floor value
            v = rand(o["gap_floor_lo"], o["gap_min"])
        return float(min(v, o["gap_max"]))

    def pick_R(self) -> float:
        o = self.o
        if random.random() < o["p_R_standard"]:
            r = random.gauss(o["R_standard_mean"], o["R_standard_sd"])
        else:
            r = rand(o["R_min"], o["R_max"])
        return float(min(max(r, o["R_min"]), o["R_max"]))

    def pick_wrap_cells(self) -> float:
        return float(random.choice(self.wrap)) * rand(0.9, 1.25) if self.wrap else 24.0


def _get_real_sampler(cfg: PipelineConfig) -> RealSampler:
    key = (id(cfg.real_profile), json.dumps(cfg.real_pool_weights, sort_keys=True), json.dumps(cfg.real_opts, sort_keys=True, default=str))
    if key not in _REAL_SAMPLER_CACHE:
        _REAL_SAMPLER_CACHE[key] = RealSampler(cfg.real_profile, cfg.real_pool_weights, cfg.real_opts)
    return _REAL_SAMPLER_CACHE[key]


def _ink_quad(word_records: list[dict], geom: dict, font_size: int, cs: float, dot_scale: float,
              full_cells: bool = False):
    """Axis-aligned quad around the actual dots of the rendered base image (pre-warp)."""
    sf = font_size / geom["size"]
    adv = geom["adv"] * sf
    pitch = adv * cs
    r_eff = 0.5 * geom["dot_diam"] * sf * dot_scale
    cx = [x * sf for x in geom["col_x"]]
    ry = [y * sf for y in geom["row_y"]]
    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    for rec in word_records:
        bm = float(rec["bbox_margin"])
        q = rec["quad"]
        wx = float(q[0][0]) + bm - geom["ink_left"] * sf
        ly = float(q[0][1]) + bm - geom["ink_top"] * sf
        for k, ch in enumerate(rec["braille_text"]):
            bits = ord(ch) - 0x2800
            if not bits & 0x3F:
                continue
            if full_cells:
                bits = 0x3F
            left = bits & 0x07
            right = (bits >> 3) & 0x07
            cols = [c for c, m in ((0, left), (1, right)) if m]
            rows = [r for r in range(3) if (left >> r) & 1 or (right >> r) & 1]
            cell_x = wx + k * pitch
            x0 = min(x0, cell_x + cx[min(cols)] - r_eff)
            x1 = max(x1, cell_x + cx[max(cols)] + r_eff)
            y0 = min(y0, ly + ry[min(rows)] - r_eff)
            y1 = max(y1, ly + ry[max(rows)] + r_eff)
    if not np.isfinite([x0, y0, x1, y1]).all():
        pts = np.concatenate([np.asarray(r["quad"], dtype=np.float32) for r in word_records], axis=0)
        x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x1, y1 = float(pts[:, 0].max()), float(pts[:, 1].max())
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def _crop_window_tracked(img, quads, x0, y0, x1, y1):
    """Crop [x0,x1)x[y0,y1) (float, may exceed the image: replicate-padded) and shift the quads."""
    ix0, iy0 = int(math.floor(x0)), int(math.floor(y0))
    ix1, iy1 = max(ix0 + 2, int(math.ceil(x1))), max(iy0 + 2, int(math.ceil(y1)))
    h, w = img.shape[:2]
    pl, pt = max(0, -ix0), max(0, -iy0)
    pr, pb = max(0, ix1 - w), max(0, iy1 - h)
    if pl or pt or pr or pb:
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, borderType=cv2.BORDER_REPLICATE)
        ix0, iy0, ix1, iy1 = ix0 + pl, iy0 + pt, ix1 + pl, iy1 + pt
        shift_x, shift_y = ix0 - pl, iy0 - pt
    else:
        shift_x, shift_y = ix0, iy0
    out = img[iy0:iy1, ix0:ix1]
    new_h, new_w = out.shape[:2]
    new_quads = []
    for q in quads:
        pts = np.asarray(q, dtype=np.float32).copy()
        pts[:, 0] -= shift_x
        pts[:, 1] -= shift_y
        new_quads.append(_clip_quad_to_image(_order_quad(pts), new_w, new_h))
    return np.ascontiguousarray(out), new_quads


def _resize_tracked(img, quads, sx, sy):
    h, w = img.shape[:2]
    nw, nh = max(2, int(round(w * sx))), max(2, int(round(h * sy)))
    interp = cv2.INTER_AREA if (nw <= w and nh <= h) else cv2.INTER_CUBIC
    out = cv2.resize(img, (nw, nh), interpolation=interp)
    fx, fy = nw / w, nh / h
    new_quads = []
    for q in quads:
        pts = np.asarray(q, dtype=np.float32).copy()
        pts[:, 0] *= fx
        pts[:, 1] *= fy
        new_quads.append(_clip_quad_to_image(_order_quad(pts), nw, nh))
    return out, new_quads, fx, fy


def qwen_visual_tokens(w: int, h: int, factor: int = 32, min_pixels: int = 65536, max_pixels: int = 16777216) -> int:
    """Visual tokens Qwen3-VL/3.5 spends on a w x h image (smart_resize to multiples of 32, 1 token per 32x32)."""
    hb, wb = max(factor, round(h / factor) * factor), max(factor, round(w / factor) * factor)
    if hb * wb > max_pixels:
        b = math.sqrt(h * w / max_pixels)
        hb, wb = math.floor(h / b / factor) * factor, math.floor(w / b / factor) * factor
    elif hb * wb < min_pixels:
        b = math.sqrt(min_pixels / (h * w))
        hb, wb = math.ceil(h * b / factor) * factor, math.ceil(w * b / factor) * factor
    return (hb // factor) * (wb // factor)


def _post_degrade(img, dot_pitch_px, o, meta):
    """Blur / noise / JPEG applied at the FINAL (low) resolution, like a real small crop."""
    info = {}
    if random.random() < o["p_blur"]:
        sigma = rand(0.0, o["blur_max_dp"]) * dot_pitch_px
        if sigma > 0.15:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
            info["blur_sigma"] = round(sigma, 3)
    if random.random() < o["p_noise"]:
        sigma = rand(0.5, o["noise_max"])
        noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        info["noise_sigma"] = round(sigma, 2)
    if random.random() < o["p_jpeg"]:
        q = randi(int(o["jpeg_q_min"]), int(o["jpeg_q_max"]))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            info["jpeg_q"] = q
    meta["real_style"]["post"] = info
    return img


TAPE_COLORS = [(20, 20, 20), (35, 30, 30), (90, 40, 10), (40, 70, 20), (20, 20, 140), (20, 110, 230),
               (30, 170, 230), (110, 40, 40), (60, 60, 60), (0, 90, 0), (130, 20, 90)]   # BGR
DOT_LIGHT = [(245, 245, 245), (230, 240, 250), (210, 225, 235), (200, 230, 255)]


def _tape_strip_image(img_gray: np.ndarray, hmap: np.ndarray, word_records: list[dict], P: float,
                      bg_dir: str) -> np.ndarray:
    """Re-composite a rendered braille canvas as light embossed dots on coloured tape strips
    (one strip per text line) laid on a page / photo background. Geometry is unchanged."""
    h, w = hmap.shape
    paths = _load_backgrounds(bg_dir)
    if paths and random.random() < 0.6:
        page = cv2.imread(random.choice(paths))
        page = cv2.resize(page, (w, h)) if page is not None else None
    else:
        page = None
    if page is None:
        c0 = np.array([randi(120, 255) for _ in range(3)], np.float32)
        c1 = np.clip(c0 + np.array([rand(-60, 60) for _ in range(3)], np.float32), 0, 255)
        t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        page = (c0 * (1 - t) + c1 * t) * np.ones((1, w, 1), np.float32)
    out = page.astype(np.float32)
    lines: dict = {}
    for rec in word_records:
        q = np.asarray(rec["quad"], np.float32)
        x0, y0, x1, y1 = q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
        L = lines.setdefault(rec["line_idx"], [x0, y0, x1, y1])
        L[:] = [min(L[0], x0), min(L[1], y0), max(L[2], x1), max(L[3], y1)]
    one_color = random.random() < 0.6
    color0 = np.array(random.choice(TAPE_COLORS), np.float32)
    strip_mask = np.zeros((h, w), np.float32)
    for li in sorted(lines):
        x0, y0, x1, y1 = lines[li]
        x0 -= rand(0.2, 1.2) * P; x1 += rand(0.2, 1.2) * P
        y0 -= rand(0.1, 0.45) * P; y1 += rand(-0.1, 0.25) * P
        x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        col = color0 if one_color else np.array(random.choice(TAPE_COLORS), np.float32)
        d = max(1, int(rand(0.02, 0.1) * P))                               # drop shadow on the page
        sy0, sy1, sx0, sx1 = min(h, y0 + d), min(h, y1 + d), min(w, x0 + d), min(w, x1 + d)
        out[sy0:sy1, sx0:sx1] *= rand(0.55, 0.8)
        gloss = 1.0 + rand(-0.15, 0.15) * np.linspace(-1, 1, y1 - y0, dtype=np.float32)[:, None, None]
        out[y0:y1, x0:x1] = np.clip(col * gloss * np.random.normal(1.0, 0.03, (y1 - y0, x1 - x0, 1)), 0, 255)
        strip_mask[y0:y1, x0:x1] = 1.0
    shade = img_gray.astype(np.float32) / max(1.0, float(np.percentile(img_gray, 99)))
    dot = np.array(random.choice(DOT_LIGHT), np.float32) * np.clip(0.55 + 0.6 * shade, 0.4, 1.15)[..., None]
    a = (np.clip(hmap * rand(1.0, 1.4), 0, 1) * np.maximum(strip_mask, 0.0))[..., None]
    out = out * (1 - a) + dot * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _ghost_style() -> dict:
    """Random flat-print palette: light paper + dark ink (or the inverse), ghost dots in between."""
    if random.random() < 0.8:
        paper = tuple(randi(215, 255) for _ in range(3))
        ink = tuple(randi(0, 110) for _ in range(3))
    else:
        paper = tuple(randi(0, 70) for _ in range(3))
        ink = tuple(randi(200, 255) for _ in range(3))
    k = rand(0.25, 0.6)                                       # ghost colour: ink faded toward paper
    ghost_col = tuple(int(i * k + p * (1 - k)) for i, p in zip(ink, paper))
    return {"paper": paper, "ink": ink, "ghost_col": ghost_col, "ghost_alpha": rand(0.6, 1.0)}


def generate_sample_realmatch(
    text: str, cfg: PipelineConfig, sample_idx: int = 0
) -> tuple[np.ndarray, list[dict], bool, Optional[str], dict]:
    """Same outputs as generate_sample, but scale/spacing/tightness follow real images."""
    from dataclasses import replace

    text = _normalize_source_text(text)
    if not text:
        raise ValueError("Text is empty after braille-safe normalization")
    o = cfg.real_opts
    sampler = _get_real_sampler(cfg)
    geom = font_geometry(cfg.font_path)
    adv_em = geom["adv"] / geom["size"]
    dp_em = geom["dot_pitch"] / geom["size"]
    diam_over_dp = geom["dot_diam"] / geom["dot_pitch"]

    grade = 2 if coin(cfg.p_grade2) else 1

    # ---- spacing: physical R and b -> generator controls --------------------
    dynamic = o["p_dynamic_spacing"] > 0 and random.random() < o["p_dynamic_spacing"]
    R = rand(*o["dyn_R"]) if dynamic else sampler.pick_R()
    cs = R * dp_em / adv_em
    b = rand(*o["dyn_b"]) if dynamic else sampler.pick_gap()
    ws = max(0.02, cs * (1.0 + b) - 1.0)

    # ---- text structure: cells per line --------------------------------------
    small = 72
    f_small = ImageFont.truetype(cfg.font_path, small)
    pitch_small = f_small.getlength("⠿") * cs
    one_line, _ = layout_paragraph(text, f_small, 10**7, grade, cs, ws)
    n_cells = max(1.0, max((e["x1"] for ln in one_line for e in ln), default=1.0) / pitch_small)
    if n_cells <= o["max_single_cells"] and random.random() < o["p_single_line"]:
        wrap_cells = float("inf")
    else:
        wrap_cells = max(sampler.pick_wrap_cells(), n_cells / max(1, o["max_lines"]), 4.0)
    wrap_px_small = 10**7 if wrap_cells == float("inf") else wrap_cells * pitch_small
    lines_small, _ = layout_paragraph(text, f_small, wrap_px_small, grade, cs, ws)
    n_lines = max(1, len(lines_small))
    cls = 1 if n_lines == 1 else 2 if n_lines <= 3 else 4

    # ---- style record: scale, margins, dot size ------------------------------
    explore = random.random() < o["p_explore"]
    rec = sampler.pick_record(cls, n_cells)
    if explore:
        P_t = rec["P"] * math.exp(random.gauss(0, o["explore_sigma_P"]))
        m = [min(max(v + random.gauss(0, o["explore_sigma_m"]), -0.35), 3.6) for v in rec["m"]]
        dr = rec["dr"] * math.exp(random.gauss(0, o["explore_sigma_dr"]))
        pool = "explore"
    else:
        js = o["jitter_sigma"]
        P_t = rec["P"] * math.exp(random.gauss(0, js))
        m = [min(max(v + random.gauss(0, 0.06), -0.35), 3.6) for v in rec["m"]]
        dr = rec["dr"] * math.exp(random.gauss(0, js))
        pool = "bank"
    P_t = float(min(max(P_t, o["P_min"]), o["P_max"]))
    # longer texts must stay readable: per text-size floor on the cell pitch (real long photos
    # go down to ~9 px/cell, which taught the model to guess); ascending, larger floors win
    for min_cells, floor_px in o["long_P_floor"]:
        if n_cells >= min_cells and P_t < floor_px:
            P_t = float(floor_px) * math.exp(rand(0.0, 0.25))
    closeup = False
    if o["p_closeup"] > 0 and n_cells <= o["closeup_max_cells"] and random.random() < o["p_closeup"]:
        lo, hi = o["closeup_P"]
        P_t = math.exp(rand(math.log(lo), math.log(hi)))
        closeup, pool = True, "closeup"
    m = [min(max(v + off, -0.35), 3.7) for v, off in zip(m, o["margin_offset"])]
    # no dot may be cut by the crop (a clipped first cell with a full label teaches the model to
    # invent the start of the text): margins below the floor are re-drawn just above it
    m = [v if v >= o["margin_floor"] else rand(o["margin_floor"], o["margin_floor"] + o["margin_floor_spread"]) for v in m]
    dr = min(max(dr * o["dot_ratio_calib"], 0.3), 1.1)
    dot_scale = float(np.clip(dr / diam_over_dp, 0.4, 1.7))
    aniso = float(np.clip(math.exp(random.gauss(0, o["aniso_sigma"])), 1 / o["aniso_clip"], o["aniso_clip"]))

    # ---- render the working canvas at a comfortable base size ---------------
    font_size = int(round(max(o["base_font_min"], P_t / (adv_em * cs))))
    sf = font_size / geom["size"]
    P_base = geom["adv"] * sf * cs
    # line pitch in cell pitches (real value if the record has one); the layout's own line
    # advance is bbox("A")[3] + line_spacing, so line_spacing is whatever reaches the target
    if dynamic:
        L = rand(*o["dyn_L"])
    else:
        L = rec.get("lp") or 1.75
        L = float(min(max(L * math.exp(random.gauss(0, 0.06)), 1.45), 6.0))
    a_bottom = float(ImageFont.truetype(cfg.font_path, font_size).getbbox("A")[3])
    line_spacing = int(round(L * P_base - a_bottom))
    pad = int(math.ceil(o["canvas_pad"] * P_base))
    # a word longer than the wrap width must not trigger render_base's shrink-to-fit
    # (it would silently change the scale), so the canvas allows the widest line
    widest_cells = max((e["x1"] for ln in lines_small for e in ln), default=1.0) / pitch_small
    wrap_px = 10**6 if wrap_cells == float("inf") else int(max(wrap_cells, 1.03 * widest_cells) * P_base)
    kbase = P_base / 84.0
    cfg_r = replace(
        cfg,
        font_size=font_size,
        font_ref=int(geom["size"]),
        line_spacing=line_spacing,
        padding=pad,
        max_image_width=wrap_px + 2 * pad,
        max_image_height=10**6,
        boundary_min_px=max(1, int(round(cfg.boundary_min_px * kbase))),
        boundary_max_px=max(2, int(round(cfg.boundary_max_px * kbase))),
        enable_random_resolution=False,
        dot_blur_scale=float(o["dot_blur_scale"]),
    )
    style = "emboss"
    ghost = None
    if (o["p_ghost"] > 0 and len(text.split()) <= o["ghost_max_words"] and random.random() < o["p_ghost"]):
        ghost = _ghost_style()
        ghost["scale"] = rand(*o["ghost_scale"])
        style = "ghost"
    elif o["p_hollow"] > 0 and random.random() < o["p_hollow"]:
        ghost = _ghost_style()
        ghost["scale"] = rand(*o["ghost_scale"])
        ghost["ring"] = 1.0 - rand(*o["hollow_inner"])
        ghost["show_ghost"] = len(text.split()) <= o["ghost_max_words"] and random.random() < o["p_hollow_ghost"]
        style = "hollow"
    elif o["p_strip"] > 0 and random.random() < o["p_strip"]:
        style = "strip"
    if style != "emboss":
        cfg_r = replace(cfg_r, p_background_compose=0.0)   # these styles composite their own page
    img, hmap, word_records, backend = render_base(
        text, cfg_r, grade=grade, char_spacing_scale=cs,
        word_spacing_scale=ws, dot_scale=dot_scale, ghost=ghost,
    )
    if style == "strip":
        img = _tape_strip_image(img, hmap, word_records, P_base, cfg.background_dir)
    quads = [rec["quad"] for rec in word_records]
    ink_q = _ink_quad(word_records, geom, font_size, cs, dot_scale, full_cells=(style == "ghost" or (style == "hollow" and ghost.get("show_ghost"))))

    plan = {
        "braille_grade": grade, "char_spacing_scale": cs, "word_spacing_scale": ws,
        "dot_scale": dot_scale, "resolution_target": None,
    }
    meta = dict(plan)
    meta["braille_backend"] = backend
    meta["vertical_flip"] = False
    meta["rotation_angle"] = 0
    meta["random_boundary"] = None
    meta["random_resolution"] = None

    img, quads_ext, bg_used, bg_source = _augment_rendered(
        img, hmap, quads + [ink_q], cfg_r, plan, meta
    )
    ink_after = np.asarray(quads_ext[-1], dtype=np.float32)
    quads = quads_ext[:-1]

    # ---- framing: crop to the real margin distribution (uses the tracked ink box)
    ix0, iy0 = float(ink_after[:, 0].min()), float(ink_after[:, 1].min())
    ix1, iy1 = float(ink_after[:, 0].max()), float(ink_after[:, 1].max())
    # the ink box is dot centres +- radius; pad it for dot jitter, blur halo and warp-tracking
    # error so that, with margins >= margin_floor, no dot is cut
    safety = max(float(word_records[0]["bbox_margin"]) if word_records else 0.0, o["crop_safety"] * P_base)
    raw_ink = (ix0, iy0, ix1, iy1)          # dot edges, before the safety pad (for the audit below)
    ix0, iy0, ix1, iy1 = ix0 - safety, iy0 - safety, ix1 + safety, iy1 + safety
    wx0, wy0 = ix0 - m[0] * P_base, iy0 - m[2] * P_base
    wx1, wy1 = ix1 + m[1] * P_base, iy1 + m[3] * P_base
    ink_area = max(1e-6, (ix1 - ix0) * (iy1 - iy0))
    kept = max(0.0, min(ix1, wx1) - max(ix0, wx0)) * max(0.0, min(iy1, wy1) - max(iy0, wy0)) / ink_area
    if not np.isfinite(kept) or kept < 0.3:
        raise ValueError(f"ink box not inside the crop window (kept {kept:.2f}); sample is retried")
    img, quads = _crop_window_tracked(img, quads, wx0, wy0, wx1, wy1)

    # ---- final scale to the target cell pitch (+ x/y squeeze, size limits) ---
    ch, cw = img.shape[:2]
    s = P_t / P_base
    s = max(s, 28.0 / max(1, cw), 28.0 * aniso / max(1, ch))   # real crops are >= ~28 px (after the squeeze)
    s_wanted = s
    s = min(s, o["max_side"] / max(cw, ch / aniso, 1))           # longest side (after the squeeze)
    s = min(s, math.sqrt(o["max_pixels"] * aniso / max(1, cw * ch)))   # area = visual-token budget
    capped = s < s_wanted - 1e-9
    sx, sy = s, s / aniso
    # exact visual-token budget: the processor rounds each side to a multiple of 32 px, which can
    # push a capped area slightly over; shrink until the processor's token count fits
    for _ in range(6):
        t = qwen_visual_tokens(max(2, round(cw * sx)), max(2, round(ch * sy)))
        if t <= o["max_visual_tokens"]:
            break
        f = math.sqrt(o["max_visual_tokens"] / t) * 0.99
        sx, sy, capped = sx * f, sy * f, True
    img, quads, fx, fy = _resize_tracked(img, quads, sx, sy)
    dp_native = (P_base * fx) / R
    # audit: the tracked dot box in FINAL image pixels and its smallest distance to the border
    # (>= 0 means no dot is cut; a negative value would be a clipped dot)
    lowres = None
    if closeup and random.random() < o["closeup_lowres_p"]:
        lowres = rand(0.2, 0.5)                            # phone crop of a small region, upscaled
        hh, ww = img.shape[:2]
        small_img = cv2.resize(img, (max(2, int(ww * lowres)), max(2, int(hh * lowres))), interpolation=cv2.INTER_AREA)
        img = cv2.resize(small_img, (ww, hh), interpolation=cv2.INTER_CUBIC)
    fh, fw = img.shape[:2]
    ink_final = [(raw_ink[0] - wx0) * fx, (raw_ink[1] - wy0) * fy, (raw_ink[2] - wx0) * fx, (raw_ink[3] - wy0) * fy]
    ink_border_px = min(ink_final[0], ink_final[1], fw - ink_final[2], fh - ink_final[3])

    meta["real_style"] = {
        "pool": pool, "class": cls, "n_lines": n_lines, "n_cells_est": round(n_cells, 1),
        "wrap_cells": None if wrap_cells == float("inf") else round(wrap_cells, 1),
        "P_target": round(P_t, 3), "P_base": round(P_base, 3), "P_final_x": round(P_base * fx, 3),
        "R": round(R, 4), "b": round(b, 4), "dr": round(dr, 4), "aniso": round(aniso, 4),
        "margins": [round(v, 3) for v in m], "font_size": font_size, "line_pitch": round(L, 3),
        "scale_x": round(fx, 5), "scale_y": round(fy, 5), "ink_kept": round(kept, 3),
        "ink_box": [round(v, 2) for v in ink_final], "ink_border_px": round(ink_border_px, 3),
        "size_capped": bool(capped), "style": style, "closeup": closeup, "dynamic_spacing": dynamic,
        "closeup_lowres": None if lowres is None else round(lowres, 3), "visual_tokens": qwen_visual_tokens(img.shape[1], img.shape[0]),
    }
    img = _post_degrade(img, dp_native, o, meta)
    if float(img.std()) < o["min_image_std"]:
        # e.g. an extreme low-light draw or a faint background blend: no readable dots left
        raise ValueError(f"near-uniform image (std {float(img.std()):.2f}); sample is retried")

    for rec, q in zip(word_records, quads):
        rec["quad"] = q
    return img, word_records, bg_used, bg_source, meta
