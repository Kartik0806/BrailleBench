"""Dataset generation loop (per-worker sample generation and label writer)."""

import numpy as np
import cv2
import random
import os
import json
import time
from typing import Optional
import multiprocessing as mp
from .braille_text import shuffle_source_text, strip_punct_edge_tokens
from .config import PipelineConfig
from .labels import build_bbox_record, draw_visualization
from .pipeline import generate_sample
from .realmatch import generate_sample_realmatch


# ===========================================================================
# DATASET GENERATION LOOP
# ===========================================================================

from tqdm import tqdm


def write_labels_to_file(records: mp.Queue, label_file: str, total: int) -> None:
    """Write until a None sentinel arrives; avoids deadlock if one sample fails."""
    written = 0
    with (
        open(label_file, "a", encoding="utf-8") as label_fh,
        tqdm(total=total, desc="Writing labels") as bar,
    ):
        while True:
            record = records.get()
            if record is None:
                break
            label_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            bar.update(1)
    if written != total:
        print(f"⚠ Wrote {written}/{total} labels because some samples failed.")


def generate_dataset(
    texts: list[str],
    n_per_text: int = 1,
    cfg: Optional[PipelineConfig] = None,
    verbose: bool = True,
    output_q: Optional[mp.Queue] = None,
    offset: int = 0,
    images_dir: Optional[str] = None,
    vis_dir: Optional[str] = None,
    visualize: bool = False,
) -> None:
    if cfg is None:
        cfg = PipelineConfig()
    images_dir = images_dir or cfg.output_dir

    total = len(texts) * n_per_text
    done = 0
    t0 = time.time()

    for ti, text in enumerate(texts):
        for i in range(n_per_text):
            # `offset` is the number of samples already produced by earlier
            # chunks (see main below) so sample indices stay globally
            # unique across worker processes.
            sample_idx = offset + ti * n_per_text + i
            fname = f"sample_{sample_idx:06d}.jpg"
            fpath = os.path.join(images_dir, fname)

            # legacy mode: one attempt, original order of operations (byte-identical output).
            # real-matching mode: a failed sample is retried with a fresh seed (so the label
            # count is exact) and the image is written only after its label has been built
            # (so a failure can never leave an unlabelled image behind).
            real_mode = cfg.real_profile is not None
            attempts = max(1, int(cfg.real_opts.get("retries", 6))) if real_mode else 1
            for attempt in range(attempts):
                try:
                    text_rng = random.Random(sample_idx + 0x9E3779B97F4A7C15)
                    rendered_text, text_shuffle = shuffle_source_text(text, cfg, text_rng)
                    if cfg.strip_punct_tokens:
                        # a word shuffle can move an inner "-" to the front
                        rendered_text = strip_punct_edge_tokens(rendered_text) or rendered_text
                    if real_mode:
                        # reproducible per sample, independent of how samples are chunked
                        # over workers; attempt 0 is the seed used before retries existed
                        _seed = (cfg.real_seed * 1000003 + sample_idx * 7919 + 17 + attempt * 104729) % (2**32)
                        random.seed(_seed)
                        np.random.seed(_seed)
                        _gen = generate_sample_realmatch
                    else:
                        _gen = generate_sample
                    img, word_records, bg_used, bg_source, sample_meta = _gen(
                        rendered_text, cfg, sample_idx
                    )
                    if text_shuffle:
                        sample_meta["text_shuffle"] = text_shuffle
                    if not real_mode:
                        cv2.imwrite(fpath, img)

                    h, w = img.shape[:2]
                    record = build_bbox_record(
                        rendered_text,
                        word_records,
                        w,
                        h,
                        sample_idx,
                        fname,
                        bg_used,
                        bg_source,
                        sample_meta=sample_meta,
                    )
                    if text_shuffle:
                        record["source_text_before_shuffle"] = text
                    if real_mode:
                        record["augmentation"]["real_style"]["attempt"] = attempt
                        json.dumps(record, ensure_ascii=False)  # must be serialisable before we commit
                        if not cv2.imwrite(fpath, img):
                            raise IOError(f"cv2.imwrite failed for {fpath}")

                    if output_q is not None:
                        output_q.put(record)

                    if visualize and vis_dir is not None:
                        vis = draw_visualization(img, record)
                        cv2.imwrite(os.path.join(vis_dir, fname), vis)

                    done += 1
                    break
                except Exception as e:
                    if attempt + 1 < attempts:
                        print(f"  ⚠ sample {sample_idx} attempt {attempt} failed ({e}); retrying")
                    else:
                        print(f"  ⚠ Error on sample {sample_idx}: {e}")

    elapsed = time.time() - t0
    print(f"\n✅ Generated {done}/{total} samples in {elapsed:.1f}s → {images_dir}")
