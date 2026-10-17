# generate-data

Synthetic braille photographs with labels. Text (Unicode braille, UEB Grade 1 or Grade 2 via liblouis)
is rendered as an embossed height map and pushed through geometric, optical, lighting and noise
augmentations. Every geometric step tracks the word boxes, so each image gets a label with word, line,
sentence and paragraph boxes.

An optional **real-matching mode** (`--real_profile`) renders each sample so that scale, margins, cell
and word spacing, dot size and lines per image follow distributions measured on real braille photos.
Without that flag the generator is the plain synthetic one.

## Layout

```
braille_gen_fineweb_v2.py   entry point (python braille_gen_fineweb_v2.py ...)
braille_gen/                the generator, split by pipeline stage:
    config  utils  braille_text  render  effects  geometry
    background  labels  pipeline  realmatch  dataset  texts  cli
realstats/                  measure real photos, build the real-match profile, validate generated data
braille_text_plan.py        stratified text plans             }  optional staged pipeline
make_llava_json.py          labels.jsonl -> llava.json        }  (pretrain / midtrain / posttrain);
migrate_final_data.py       merge stage datasets              }  not needed to generate data
```

## Setup

- Python packages: `numpy`, `opencv-python`, `pillow`, `tqdm`, and `datasets` (to stream FineWeb).
  Optional: `scipy`, `noise`.
- **liblouis with Python bindings.** Without it a built-in translator is used and the braille differs.
  To use a local build: `LIBLOUIS_DIR=/path/to/liblouis-<version> source realstats/louis_env.sh`.
- A monospace font (default `/usr/share/fonts/truetype/freefont/FreeMono.ttf`; override with
  `BRAILLE_FONT_PATH` or `--font_path`).
- Optional background photos for compositing: `--background_dir <dir>` (skipped if empty).

## Generate

```bash
# from a text file (one text per line)
python braille_gen_fineweb_v2.py --texts_file texts.txt --num_samples 1000 \
    --output_dir data/train --background_dir backgrounds/ --n_workers 8

# from FineWeb (streamed from the Hugging Face Hub)
python braille_gen_fineweb_v2.py --dataset HuggingFaceFW/fineweb --dataset_config sample-10BT \
    --num_samples 1000 --min_text_words 1 --max_text_words 10 --output_dir data/train

# real-matching mode
python braille_gen_fineweb_v2.py ... --real_profile realmatch_profile.json
```

`python braille_gen_fineweb_v2.py --help` lists every flag (augmentation probabilities, grade mix
`--p_grade2`, text length limits, `--start_index` to append, `--visualize` for box overlays).
Generate a validation set the same way into `data/val`; [`train`](../train/) looks for it next to
`data/train`. Existing output is never overwritten unless you pass `--reset_output`.

## Output

```
<output_dir>/images/sample_000000.jpg
<output_dir>/labels/labels.jsonl
```

Each label has `source_text`, `braille_text`, `braille_grade`, the word / line / sentence / paragraph
boxes, image size and the augmentations that were applied.

## Real-matching profile

`realstats/` builds the profile from real photos: `run_measure_pools.py` measures the image pools
(set `EVAL_DIR` to the folder holding them), `build_profile.py` turns the measurements into the
profile JSON, and `validate_generation.py` compares generated data against the real statistics. See
`realstats/README.md` for details and the measured differences from the plain generator.

## Reproducibility

In real-matching mode every sample is seeded from its index, so output is reproducible and independent of
the number of workers. In plain mode the workers' `random` state is reseeded by Python after each fork,
so `--seed` fixes the texts but not the exact images.

## Environment variables

| Variable | Used by | Meaning |
|---|---|---|
| `LIBLOUIS_DIR` | `realstats/louis_env.sh` | liblouis build to use |
| `BRAILLE_FONT_PATH` | generator | font file |
| `EVAL_DIR` | `realstats/run_measure_pools.py` | folder with the real image pools |
| `STRESS_TEXTS`, `VAL_TEXTS`, `STYLE_TEXTS` | `realstats/` scripts | text file, one text per line |
