"""Command-line entry point: argument parsing and the multi-iteration generation driver."""

import numpy as np
import random
import os
import json
import glob
import argparse
import multiprocessing as mp
from .background import _has_backgrounds
from .config import PipelineConfig
from .dataset import generate_dataset, write_labels_to_file
from .deps import HAS_LIBLOUIS
from .realmatch import RealSampler
from .texts import get_data, get_texts_from_file


def main():
    parser = argparse.ArgumentParser(
        description="FineWeb -> synthetic braille dataset generator"
    )
    parser.add_argument("--n_per_text", type=int, default=1)
    parser.add_argument("--output_dir", default="data/train")
    parser.add_argument("--background_dir", default="backgrounds/")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb")
    parser.add_argument("--dataset_config", default="sample-10BT")
    parser.add_argument(
        "--min_text_chars",
        type=int,
        default=1,
        help="Minimum sampled character budget per example (default: 1)",
    )
    parser.add_argument(
        "--max_text_chars",
        type=int,
        default=420,
        help="Maximum character count / sampled character budget (default: 420)",
    )
    parser.add_argument(
        "--min_text_words",
        type=int,
        default=1,
        help="Minimum sampled number of whole words per example (default: 1)",
    )
    parser.add_argument(
        "--max_text_words",
        type=int,
        default=10,
        help="Maximum sampled number of whole words per example (default: 10)",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Run multiple sequential generation batches automatically",
    )
    parser.add_argument(
        "--seed_step",
        type=int,
        default=1,
        help="Amount added to --seed after each iteration (default: 1)",
    )
    parser.add_argument(
        "--font_path", default=os.environ.get("BRAILLE_FONT_PATH", "/usr/share/fonts/truetype/freefont/FreeMono.ttf")
    )
    parser.add_argument("--bg_prob", type=float, default=0.5)
    parser.add_argument("--p_perspective", type=float, default=0.8)
    parser.add_argument("--p_page_curl", type=float, default=0.5)
    parser.add_argument("--p_elastic", type=float, default=0.6)
    parser.add_argument("--p_stain", type=float, default=0.4)
    parser.add_argument("--p_salt_pepper", type=float, default=0.4)
    parser.add_argument("--p_low_light", type=float, default=0.35)
    parser.add_argument("--p_shadow_band", type=float, default=0.4)
    parser.add_argument("--p_paper_texture", type=float, default=1.0)
    parser.add_argument("--p_lighting", type=float, default=1.0)
    parser.add_argument("--p_noise", type=float, default=1.0)
    parser.add_argument(
        "--p_shuffle_words", type=float, default=0.0,
        help="Chance of rearranging whole words before rendering (off by default)",
    )
    parser.add_argument(
        "--p_shuffle_chars", type=float, default=0.0,
        help="Chance of scrambling letters in selected words (off by default)",
    )
    parser.add_argument(
        "--shuffle_char_word_fraction", type=float, default=0.35,
        help="Fraction of eligible words scrambled when character shuffle fires",
    )
    parser.add_argument(
        "--real_profile", default=None,
        help="JSON from src/realstats/build_profile.py. Enables real-image matching "
        "(scale / spacing / margins follow real photos). Omit for legacy behaviour.",
    )
    parser.add_argument(
        "--real_pool_weights", default=None,
        help="e.g. eval_combined=0.4,new_data=0.4,book_pages=0.2 (pools missing from the profile are ignored)",
    )
    parser.add_argument(
        "--real_opt", action="append", default=[], metavar="KEY=VALUE",
        help="override an entry of REAL_DEFAULTS, e.g. --real_opt p_R_standard=0.8 (repeatable)",
    )
    parser.add_argument(
        "--strip_punct_tokens", action="store_true",
        help="drop punctuation-only texts and strip punctuation-only tokens (e.g. a leading '-') "
        "from both ends of every text, also after word shuffling",
    )
    parser.add_argument(
        "--texts_file", default=None,
        help="UTF-8 file with one text per line; used instead of streaming --dataset (offline runs)",
    )
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--p_grade2", type=float, default=0.5)
    parser.add_argument("--p_vertical_flip", type=float, default=0.15)
    parser.add_argument(
        "--p_rotation",
        type=float,
        default=0.0,
        help="Probability of applying --rotation_angle to each sample",
    )
    parser.add_argument(
        "--rotation_angle",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
        help="Exact clockwise image rotation in degrees: 0, 90, 180, or 270",
    )
    parser.add_argument("--p_random_resolution", type=float, default=0.30)
    parser.add_argument("--min_width", type=int, default=320)
    parser.add_argument("--max_width", type=int, default=1280)
    parser.add_argument("--min_height", type=int, default=180)
    parser.add_argument("--max_height", type=int, default=720)
    parser.add_argument("--p_random_boundary", type=float, default=0.45)
    parser.add_argument("--boundary_min_px", type=int, default=4)
    parser.add_argument("--boundary_max_px", type=int, default=36)
    parser.add_argument("--char_spacing_min", type=float, default=0.90)
    parser.add_argument("--char_spacing_max", type=float, default=1.18)
    parser.add_argument("--word_spacing_min", type=float, default=0.85)
    parser.add_argument("--word_spacing_max", type=float, default=1.55)
    parser.add_argument("--dot_scale_min", type=float, default=0.78)
    parser.add_argument("--dot_scale_max", type=float, default=1.38)
    parser.add_argument("--p_skew", type=float, default=0.7)
    parser.add_argument(
        "--max_skew_deg",
        type=float,
        default=12.0,
        help="Max in-plane rotation about the page/camera z-axis",
    )
    parser.add_argument(
        "--skew_boundary_mode", choices=["expand", "crop"], default="expand"
    )
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting sample index for filenames/labels. Useful for repeated runs "
        "with different augmentation settings without overwriting earlier images.",
    )
    parser.add_argument(
        "--reset_output",
        action="store_true",
        help="Delete existing generated sample_*.jpg files and labels.jsonl before starting",
    )
    args = parser.parse_args()

    if args.start_index < 0:
        parser.error("--start_index must be >= 0")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.num_samples < 1:
        parser.error("--num_samples must be >= 1")
    if args.n_per_text < 1:
        parser.error("--n_per_text must be >= 1")
    if args.min_text_chars < 1 or args.max_text_chars < args.min_text_chars:
        parser.error("Require 1 <= --min_text_chars <= --max_text_chars")
    if args.min_text_words < 1 or args.max_text_words < args.min_text_words:
        parser.error("Require 1 <= --min_text_words <= --max_text_words")
    for name in (
        "bg_prob", "p_grade2", "p_perspective", "p_page_curl", "p_elastic",
        "p_stain", "p_salt_pepper", "p_low_light", "p_shadow_band",
        "p_paper_texture", "p_lighting", "p_noise", "p_shuffle_words",
        "p_shuffle_chars", "shuffle_char_word_fraction", "p_vertical_flip",
        "p_rotation", "p_random_resolution", "p_random_boundary", "p_skew",
    ):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            parser.error(f"--{name} must be in [0, 1], got {value}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    images_dir = os.path.join(args.output_dir, "images")
    labels_dir = os.path.join(args.output_dir, "labels")
    vis_dir = os.path.join(args.output_dir, "visualization")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(vis_dir, exist_ok=True)
    label_file = os.path.join(labels_dir, "labels.jsonl")

    def reset_generated_output() -> None:
        removed = 0
        for pattern in (
            os.path.join(images_dir, "sample_*.jpg"),
            os.path.join(vis_dir, "sample_*.jpg"),
        ):
            for path in glob.glob(pattern):
                os.remove(path)
                removed += 1
        if os.path.exists(label_file):
            os.remove(label_file)
        print(f"🧹 Reset output: removed {removed} generated image/visualization files")

    def existing_label_images() -> tuple[set[str], int]:
        names: set[str] = set()
        line_count = 0
        if not os.path.exists(label_file):
            return names, line_count
        with open(label_file, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                line_count += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON in {label_file} at line {line_no}: {exc}"
                    ) from exc
                image_name = rec.get("image", "")
                if image_name:
                    base = os.path.basename(image_name)
                    if base in names:
                        raise RuntimeError(
                            f"Duplicate label record for {base} in {label_file}. "
                            "Clean/fix the dataset before appending more samples."
                        )
                    names.add(base)
        return names, line_count

    def validate_existing_output() -> set[str]:
        label_names, label_count = existing_label_images()
        disk_names = {
            os.path.basename(path)
            for path in glob.glob(os.path.join(images_dir, "sample_*.jpg"))
        }
        missing_images = label_names - disk_names
        missing_labels = disk_names - label_names
        if missing_images or missing_labels:
            details = []
            if missing_images:
                details.append(f"{len(missing_images)} label(s) have no image")
            if missing_labels:
                details.append(f"{len(missing_labels)} image(s) have no label")
            raise RuntimeError(
                "Existing output is inconsistent (" + ", ".join(details) + "). "
                "Use a clean output directory or --reset_output before generating."
            )
        if label_count != len(label_names):
            raise RuntimeError(
                f"Existing label file has {label_count} records but only "
                f"{len(label_names)} unique image names."
            )
        if label_count:
            print(f"✓ Existing dataset is consistent: {label_count} images + labels")
        return label_names

    if args.reset_output:
        reset_generated_output()
    existing_names = validate_existing_output()

    cfg = PipelineConfig(
        output_dir=images_dir,
        background_dir=args.background_dir,
        font_path=args.font_path,
        p_background_compose=args.bg_prob,
        p_perspective=args.p_perspective,
        p_page_curl=args.p_page_curl,
        p_elastic=args.p_elastic,
        p_stain=args.p_stain,
        p_salt_pepper=args.p_salt_pepper,
        p_low_light=args.p_low_light,
        p_shadow_band=args.p_shadow_band,
        p_paper_texture=args.p_paper_texture,
        p_lighting=args.p_lighting,
        p_noise=args.p_noise,
        p_shuffle_words=args.p_shuffle_words,
        p_shuffle_chars=args.p_shuffle_chars,
        shuffle_char_word_fraction=args.shuffle_char_word_fraction,
        p_grade2=args.p_grade2,
        p_vertical_flip=args.p_vertical_flip,
        p_rotation=args.p_rotation,
        rotation_angle=args.rotation_angle,
        p_random_resolution=args.p_random_resolution,
        min_width=args.min_width,
        max_width=args.max_width,
        min_height=args.min_height,
        max_height=args.max_height,
        p_random_boundary=args.p_random_boundary,
        boundary_min_px=args.boundary_min_px,
        boundary_max_px=args.boundary_max_px,
        char_spacing_min=args.char_spacing_min,
        char_spacing_max=args.char_spacing_max,
        word_spacing_min=args.word_spacing_min,
        word_spacing_max=args.word_spacing_max,
        dot_scale_min=args.dot_scale_min,
        dot_scale_max=args.dot_scale_max,
        p_skew=args.p_skew,
        max_skew_deg=args.max_skew_deg,
        skew_boundary_mode=args.skew_boundary_mode,
    )

    cfg.strip_punct_tokens = bool(args.strip_punct_tokens)
    if args.real_profile:
        with open(args.real_profile, "r", encoding="utf-8") as fh:
            cfg.real_profile = json.load(fh)
        cfg.real_seed = int(args.seed)
        if args.real_pool_weights:
            cfg.real_pool_weights = {
                k.strip(): float(v) for k, v in (kv.split("=") for kv in args.real_pool_weights.split(","))
            }
        for kv in args.real_opt:
            k, _, v = kv.partition("=")
            if k not in cfg.real_opts:
                parser.error(f"--real_opt: unknown key {k!r}; valid: {sorted(cfg.real_opts)}")
            try:
                cfg.real_opts[k] = json.loads(v)
            except json.JSONDecodeError:
                parser.error(f"--real_opt {kv}: value must be JSON (number, list, ...)")
        _RealSampler = RealSampler(cfg.real_profile, cfg.real_pool_weights, cfg.real_opts)  # fail fast
        print(
            "ℹ Real-image matching ON: profile=%s pools=%s opts-overrides=%s"
            % (args.real_profile, sorted(_RealSampler.pools), args.real_opt or "none")
        )
    elif args.real_pool_weights or args.real_opt:
        parser.error("--real_pool_weights / --real_opt need --real_profile")

    if _has_backgrounds(cfg.background_dir):
        print(f"ℹ Found background images in '{cfg.background_dir}'.")
    if not HAS_LIBLOUIS:
        print(
            "ℹ liblouis not found: Grade-1 uses the built-in uncontracted fallback and "
            "Grade-2 uses the built-in contraction subset. Install liblouis Python "
            "bindings to use liblouis for both grades."
        )
    else:
        print(
            "ℹ liblouis found: using it for both UEB Grade-1 and Grade-2 when the tables are available."
        )

    next_start_index = args.start_index

    for iteration in range(args.iterations):
        iter_seed = args.seed + iteration * args.seed_step
        random.seed(iter_seed)
        np.random.seed(iter_seed)

        print("\n" + "=" * 72)
        print(
            f"Iteration {iteration + 1}/{args.iterations} | "
            f"seed={iter_seed} | start_index={next_start_index}"
        )
        print("=" * 72)

        if args.texts_file:
            dataset_texts = get_texts_from_file(
                args.texts_file, args.num_samples,
                args.min_text_words, args.max_text_words, iter_seed,
                args.min_text_chars, args.max_text_chars, args.strip_punct_tokens,
            )
        else:
            dataset_texts = get_data(
                args.dataset,
                args.dataset_config,
                args.num_samples,
                args.min_text_chars,
                args.max_text_chars,
                args.min_text_words,
                args.max_text_words,
                iter_seed,
                args.strip_punct_tokens,
            )
        total_outputs = len(dataset_texts) * args.n_per_text
        if total_outputs == 0:
            raise RuntimeError(
                "FineWeb filtering produced zero usable samples. Relax the text "
                "length constraints or inspect the source dataset."
            )

        end_index = next_start_index + total_outputs - 1
        intended_names = {
            f"sample_{idx:06d}.jpg"
            for idx in range(next_start_index, end_index + 1)
        }
        collisions = sorted(intended_names & existing_names)
        if not collisions:
            collisions = sorted(
                name
                for name in intended_names
                if os.path.exists(os.path.join(images_dir, name))
            )
        if collisions:
            preview = ", ".join(collisions[:5])
            raise RuntimeError(
                f"Refusing to overwrite {len(collisions)} existing sample(s) in "
                f"index range {next_start_index}..{end_index}. Examples: {preview}. "
                "Choose a non-overlapping --start_index or use --reset_output "
                "for a brand-new dataset."
            )

        print(
            f"⚙ Generating {len(dataset_texts)} texts × {args.n_per_text} variants "
            f"with sample indices {next_start_index}..{end_index}"
        )
        print(
            f"   Text sampling: words={args.min_text_words}..{args.max_text_words}, "
            f"chars={args.min_text_chars}..{args.max_text_chars}"
        )

        manager = mp.Manager()
        output_q = manager.Queue()
        writer_process = mp.Process(
            target=write_labels_to_file,
            args=(output_q, label_file, total_outputs),
        )
        writer_process.start()

        def chunk(lst, n):
            k = max(1, len(lst) // n + 1)
            return [lst[i : i + k] for i in range(0, len(lst), k)]

        n_workers = max(1, args.n_workers)
        chunks = chunk(dataset_texts, n_workers)
        offsets, running = [], next_start_index
        for c in chunks:
            offsets.append(running)
            running += len(c) * args.n_per_text

        with mp.Pool(processes=n_workers) as pool:
            pool.starmap(
                generate_dataset,
                [
                    (
                        c,
                        args.n_per_text,
                        cfg,
                        False,
                        output_q,
                        off,
                        images_dir,
                        vis_dir,
                        args.visualize,
                    )
                    for c, off in zip(chunks, offsets)
                ],
            )
        output_q.put(None)
        writer_process.join()
        manager.shutdown()

        # Re-validate after every batch so an overnight run stops immediately if
        # an iteration somehow leaves images and labels out of sync.
        existing_names = validate_existing_output()
        next_start_index = end_index + 1

    print(
        f"\n✅ All {args.iterations} iteration(s) complete. "
        f"Next free start index: {next_start_index}"
    )
