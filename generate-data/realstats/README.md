# realstats — measure real braille photos, match the generator to them

The synthetic data (`braille_gen_fineweb.py`) renders every sample at ~60-90 px per
braille cell, with >=1.2 cells of margin, cells spaced ~3.2 dot-pitches apart and
scale tied to text length. Real photos (the eval set, `eval/new_data`, the book
photos) are 10-60 px per cell, crops often touch the dots, cells are ~2.5 dot-pitches
apart, and scale depends on what the photo is. This folder measures those
properties on real images and drives `braille_gen_fineweb_v2.py --real_profile`.

```
measure_real.py            per-image physical measurements (dot pitch, cell pitch, R, margins,
                           lines, line pitch, word gaps) — the SAME estimator is used on real
                           and generated images, so they are comparable
run_measure_pools.py       measure the real pools  -> out/real_<pool>.jsonl
check_pitch_vs_labels.py   sanity check: ink width / P must equal the label's braille cell count
build_profile.py           out/real_*.jsonl -> configs/realmatch_profile_v1[_noeval].json
validate_generation.py     generate a bucket-mixed set, measure it, print real-vs-synthetic quantiles
check_legacy_equivalence.py  proves v2 without --real_profile == the original generator (bytes; original is in backup/src/)
montage_synth.py           contact sheet of generated samples with their sampled parameters
louis_env.sh               `source` it to use the in-repo liblouis build (UEB g1/g2) with any python
```

The generator itself is the `src/braille_gen/` package (config, braille_text, render, effects, geometry,
labels, pipeline, realmatch, dataset, texts, cli); `src/braille_gen_fineweb_v2.py` is a thin entry point
for it. Paths are not hard-coded; override with env vars: `LIBLOUIS_DIR` (louis_env.sh), `EVAL_DIR`
(run_measure_pools.py, default `../eval` next to the repo), `BRAILLE_FONT_PATH` (generator font),
`STRESS_TEXTS` / `VAL_TEXTS` / `STYLE_TEXTS` (text file, one text per line, for the stress / validate / style scripts).

## Pools

| pool | what | labelled | used for |
|---|---|---|---|
| `eval_combined` | `eval/data/combined/` (322 imgs; 265 are the test set) | yes | profile (optional) + validation |
| `new_data` | `eval/new_data/` (4023 tight crops, mostly 60 px tall) | yes | profile |
| `book_pages` | `eval/data/Braille book dataset/` (190 small photos) | no | profile |

`configs/realmatch_profile_v1.json` uses all three; `..._noeval.json` leaves the test pool
out so the test set is not used to set anything (compare both with `validate_generation.py`).

## Reproduce

```bash
cd src/realstats && source louis_env.sh
python run_measure_pools.py                      # ~2 min
python check_pitch_vs_labels.py                  # detector sanity vs. labels
python build_profile.py --out ../../configs/realmatch_profile_v1.json
python build_profile.py --exclude eval_combined --out ../../configs/realmatch_profile_v1_noeval.json
python check_legacy_equivalence.py               # v2 (no profile) == original, byte for byte
python validate_generation.py legacy
python validate_generation.py gen --tag rm --profile ../../configs/realmatch_profile_v1.json
python validate_generation.py table --tags legacy rm
```

Generate data: `bash generate_realmatch.sh` (same buckets/args as `generate_balanced_shuffled.sh`).
Every generated label records the realised parameters in `augmentation.real_style`.

## Physical parameterisation used by the generator

| symbol | meaning | how it is set |
|---|---|---|
| P | cell pitch in the final image (px) | real record of the same cell-count class, jittered; image is rendered large and area-downscaled |
| R | cell pitch / dot pitch inside a word | `char_spacing_scale = R * dot_pitch / cell_advance` |
| b | blank advance across a word boundary, in cell pitches | `word_spacing_scale = cs*(1+b) - 1` |
| dr | dot diameter / dot pitch | `dot_scale` + `dot_blur_scale` (calibrated) |
| m | margins to the outermost dots (cell pitches, negative = clipped) | tight crop around the bbox-tracked ink box |
| L | line pitch (cell pitches) | `line_spacing` |
| a | x/y squeeze | anisotropic final resize |

Estimator biases were calibrated on noise-free generated crops (measured vs intended value);
`REAL_DEFAULTS` in the generator documents each calibration constant.

## Changes after the first real-matched training run (2026-10-04)

The first model trained on real-matched data (`qwen2b-realmatch-2`) got worse and invented words at
the start of sentences. Two generator properties explained it and were changed (REAL_DEFAULTS):

* `margin_floor` / `margin_floor_spread` / `crop_safety`: no dot is clipped any more. Before, 14% of
  images had their first cells cut on the left while the label kept the full first word.
  Every label now records `real_style.ink_border_px` (tracked dot box to image border, final px; >= 0).
* `long_P_floor`: cell pitch floor for longer texts ((16 cells, 18 px), (32 cells, 22 px)); before,
  64+ cell texts had median 17 px and a quarter under 10 px. `max_side` raised to 3200 px.

Checks: `check_retry_behavior.py` (failure injection), `stress_realmatch.py` (replays the generator's
per-sample seeds, reports exceptions / blank images).

## Text mix and cleaning (2026-10-04, late)

* `--strip_punct_tokens` (v2, opt-in; `STRIP_PUNCT_TOKENS=1` in the launchers): no punctuation-only texts and
  no leading/trailing punctuation-only tokens. 2.7% of FineWeb windows started with one ("- and"); the real
  eval has none, and the realmatch-2 model learned to output "- am".
* `generate_realmatch.sh` bucket shares / word ranges are configurable (defaults = original; verified that the
  default generator command lines are byte-identical). `run_realmatch_data.sh` uses 35/15/20/30% with
  g1 1-2, g1 3-6, g2 1-3, g2 4-50 words, which fills the 4-5 word gap (0% -> 10%).
  `text_mix_preview.py` previews the word-count mix of a bucket config from FineWeb without rendering.
* `train_realmatch_200k.sh`: original train.sh hyper-parameters on the 200k data, so only the data differs
  from the legacy-data model.

## Image size / visual-token budget

`REAL_DEFAULTS`: `max_side` 2048 px, `max_pixels` 1024*1024, `max_visual_tokens` 1024, applied at the final
resize (last, so it wins over the long-text pitch floor). The token count uses the processor's own
rounding (`qwen_visual_tokens`, multiples of 32 px, 1 token per 32x32), so no generated image exceeds
1024 visual tokens. Measured on 6000 samples of the run_realmatch_data.sh mix: p50 80, p99 1008, max 1024
tokens; 2.15% of samples are scaled down by the budget; 0.11% of >=32-cell texts end below the 22 px floor
(min 19.7 px). Each label records `real_style.visual_tokens` and `real_style.size_capped`.
Override e.g. `REAL_OPTS="max_visual_tokens=1536 max_pixels=1572864"`.

## Add-on styles (2026-10-05)

After the 200k model (81.9% exact on the eval), `run_realmatch_addon.sh` generates an add-on set (default 80k,
55% single words with heavy letter scrambling, lighter noise) and merges it with the 200k set via symlinks
(`merge_realmatch_datasets.py`) into `data_realmatch_combined`; train with `train_realmatch_combined.sh`.
New REAL_DEFAULTS styles, all 0 by default (the 200k set was generated with them off):
`p_ghost` (flat print, six tiny light dots per cell + large dark real dots, texts <= 4 words),
`p_strip` (lines on coloured tape strips over a page / photo), `p_closeup` (60-140 px per cell for
texts <= 10 cells, half as a blurry low-res upscale). Samples: `style_samples.py` -> `addon_samples/`.
