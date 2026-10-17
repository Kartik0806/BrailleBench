"""
Braille Synthetic-to-Real Domain Adaptation Dataset Generator  (v2)
=============================================================
v2 = braille_gen_fineweb.py + an OPT-IN "real-image matching" mode.
Without --real_profile every code path, random draw and output byte is the same
as braille_gen_fineweb.py (checked by src/realstats/check_legacy_equivalence.py).
With --real_profile <json> each sample is rendered so that its scale and
spacing follow the distribution measured on real braille photos
(src/realstats/measure_real.py -> build_profile.py):
  * cell pitch in native pixels  (real crops are 10-40 px/cell, legacy was 60-90)
  * margins / tightness           (real crops often touch the dots, legacy >= 1.2 cells)
  * in-word cell tightness R = cell pitch / dot pitch   (real ~2.2-2.8, legacy 2.9-3.7)
  * word gap b = blank advance in cell pitches            (real 0.3-2.0, legacy 0.57-1.83)
  * dot diameter / dot pitch, x/y aspect squeeze, lines per image, cells per line
  * post-resize blur / noise / JPEG at the final (low) resolution
Every geometric step stays bbox-tracked, and the realised values are written to
each label's `augmentation.real_style` so they can be audited from the labels.

Original docstring follows.
-------------------------------------------------------------
Implements all 7 augmentation stages from the adaptation plan, plus:
  Stage 3b — Random skew (small bounded-angle rotation, random direction)
  Bounding-box tracking — every geometric stage that moves pixels around
    (perspective warp, page curl, elastic warp, skew) also updates word-level
    quadrilaterals, so every generated sample gets a label record in the
    same nested word/line/sentence/paragraph bbox schema as the reference
    format, plus optional bbox-overlay visualizations.

CHANGES vs. the "dynamic word-wrap" revision:
  * render_base() now returns per-word layout info (`word_records`) in
    addition to the image/height-map. Word boxes are computed from the
    actual greedy word-wrap layout (via `layout_paragraph`), not
    reconstructed after the fact, so they line up with what was drawn.
  * All four geometric augmentation stages (perspective, page curl, elastic
    warp, and the new skew stage) have "_tracked" variants that transform
    the word quads alongside the pixels:
      - Perspective warp & skew are affine/projective, so quad points are
        transformed exactly via cv2.perspectiveTransform / cv2.transform.
      - Page curl, elastic warp, and lens distortion use cv2.remap, which
        is a *backward* mapping (dst(x,y) = src(map(x,y))). Word-box perimeter
        points are therefore inverted through the exact remap field with an
        iterative solver, then enclosed by a minimum-area quadrilateral.
        This avoids the drift and corner-only under-coverage of the previous
        first-order approximation.
  * Camera-optics stage remains OFF by default. Lens distortion now has a
    tracked variant; defocus, motion blur, chromatic aberration, and
    vignetting do not reposition the underlying text geometry.
  * Stage 7 (background composition) is now skipped automatically (with a
    one-time warning) if `background_dir` contains no images, instead of
    silently doing nothing every single call. The probability of using a
    background image *when present* is configurable via `--bg_prob`
    (default 0.5) instead of being hardcoded.
  * New Stage 3b: `apply_skew` / `apply_skew_tracked` — small bounded-angle
    rotation in a random direction (± `max_skew_deg`), probability
    `p_skew`, both configurable via CLI.
  * Output layout is now a proper dataset directory:
        <output_dir>/images/sample_000000.jpg
        <output_dir>/labels/labels.jsonl
        <output_dir>/visualization/sample_000000.jpg   (only with --visualize)
  * Fixed a multiprocessing bug where every worker was called with
    `offset=0`, so parallel chunks all wrote to the same sample_idx range
    and clobbered each other's images/labels. Offsets are now computed
    per-chunk from cumulative sample counts.

The code is split by pipeline stage into the modules of this package (config, utils, braille_text,
render, effects, geometry, background, labels, pipeline, realmatch, dataset, texts, cli).
src/braille_gen_fineweb_v2.py is a thin entry point that re-exports this namespace.
"""

import importlib as _importlib

# Re-export every module-level name (including private helpers) so that
# `import braille_gen` exposes the same namespace the single-file generator did.
for _name in (
    "deps", "config", "utils", "braille_text", "render", "effects", "geometry",
    "background", "labels", "pipeline", "realmatch", "dataset", "texts", "cli",
):
    _module = _importlib.import_module(f".{_name}", __name__)
    globals().update({k: v for k, v in vars(_module).items() if not k.startswith("__")})
