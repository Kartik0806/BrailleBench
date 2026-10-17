"""
Generate a bucket-mixed validation set, measure it with the SAME estimator used on
the real pools (measure_real.py), and compare distributions.

    source louis_env.sh
    python validate_generation.py gen   --tag rm1 [--profile ../../configs/realmatch_profile_v1.json] [--n 1600] [--clean] [--real_opt k=v ...]
    python validate_generation.py legacy                    # measure a sample of the legacy dataset
    python validate_generation.py table --tags rm1 legacy   # quantile table vs real pools

Buckets mirror generate_balanced_shuffled.sh (40% g1 1-2 words, 10% g1 6-10, 20% g2 1-3, 30% g2 6-50).
--clean switches photometric noise / background / lighting off so the estimator sees the
geometry only (checks that the generator's parameters land where intended).
"""
import argparse, json, os, subprocess, sys, shutil
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(HERE, "out")
sys.path.insert(0, HERE)
from measure_real import measure_path  # noqa: E402

TEXTS = os.environ.get("VAL_TEXTS") or sys.exit("set VAL_TEXTS to a text file with one text per line")
BUCKETS = [  # (share, p_grade2, min_words, max_words, max_chars)
    (0.40, 0.0, 1, 2, 120), (0.10, 0.0, 6, 10, 180), (0.20, 1.0, 1, 3, 120), (0.30, 1.0, 6, 50, 450),
]
COMMON = ("--bg_prob 0.30 --p_perspective 0.40 --p_page_curl 0.10 --p_elastic 0.15 --p_skew 0.20 --max_skew_deg 8 "
          "--p_vertical_flip 0 --p_rotation 0 --p_random_boundary 0.05 --p_paper_texture 0.70 --p_stain 0.10 "
          "--p_lighting 0.60 --p_low_light 0.15 --p_shadow_band 0.15 --p_noise 0.40 --p_salt_pepper 0.10").split()
CLEAN = ("--bg_prob 0 --p_perspective 0.40 --p_page_curl 0.10 --p_elastic 0.15 --p_skew 0.20 --max_skew_deg 8 "
         "--p_vertical_flip 0 --p_rotation 0 --p_random_boundary 0 --p_paper_texture 0 --p_stain 0 "
         "--p_lighting 0 --p_low_light 0 --p_shadow_band 0 --p_noise 0 --p_salt_pepper 0").split()


def gen(a):
    d = os.path.join(OUT, f"gen_{a.tag}")
    if os.path.exists(d):
        shutil.rmtree(d)
    start = 0
    caps = [int(c) for c in a.caps.split(",")] if a.caps else None
    for bi, (share, pg2, mnw, mxw, mxc) in enumerate(BUCKETS):
        if caps:
            mxc = caps[bi]
        n = int(round(a.n * share))
        cmd = [sys.executable, os.path.join(SRC, "braille_gen_fineweb_v2.py"),
               "--output_dir", d, "--background_dir", os.path.join(ROOT, "background"),
               "--texts_file", TEXTS, "--num_samples", str(n), "--start_index", str(start),
               "--seed", str(1000 + bi + a.seed), "--n_workers", "24", "--p_grade2", str(pg2),
               "--min_text_words", str(mnw), "--max_text_words", str(mxw), "--max_text_chars", str(mxc),
               "--p_random_resolution", "0.10", "--min_width", "640", "--max_width", "1280",
               "--min_height", "360", "--max_height", "720"] + (CLEAN if a.clean else COMMON)
        if a.profile:
            cmd += ["--real_profile", a.profile]
            for kv in a.real_opt:
                cmd += ["--real_opt", kv]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout[-1500:], r.stderr[-3000:]); raise SystemExit("generation failed")
        start += n
    measure_dir(d, a.tag)


def measure_dir(d, tag, limit=None):
    recs = [json.loads(l) for l in open(os.path.join(d, "labels", "labels.jsonl"))]
    if limit and len(recs) > limit:
        rng = np.random.RandomState(0)
        recs = [recs[i] for i in rng.choice(len(recs), limit, replace=False)]
    items = [(os.path.join(d, r["image"]), len(r["source_text"].split())) for r in recs]
    with Pool(32) as p:
        res = p.map(measure_path, items, chunksize=4)
    with open(os.path.join(OUT, f"synth_{tag}.jsonl"), "w") as fh:
        for r, m in zip(recs, res):
            m["pool"] = tag
            m["n_words"] = len(r["source_text"].split())
            m["truth"] = r["source_text"]
            m["real_style"] = (r.get("augmentation") or {}).get("real_style")
            fh.write(json.dumps(m) + "\n")
    print(tag, "measured", sum("error" not in m for m in res), "of", len(res))


def remeasure(a):
    measure_dir(os.path.join(OUT, f"gen_{a.tag}"), a.tag)


def legacy(a):
    measure_dir(os.path.join(ROOT, "data_new_balanced", "train"), "legacy", limit=a.n)


def _q(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray([x for x in a if x is not None and np.isfinite(x)], float)
    return None if len(a) < 5 else [float(np.percentile(a, p)) for p in ps]


def rel(rows):
    return [r for r in rows if "error" not in r and r["n"] >= 8 and r["q"] >= 0.5]


def table(a):
    srcs = {"EVAL": "real_eval_combined", "NEW": "real_new_data", "BOOK": "real_book_pages"}
    srcs.update({t: f"synth_{t}" for t in a.tags})
    data = {k: rel([json.loads(l) for l in open(os.path.join(OUT, v + ".jsonl"))]) for k, v in srcs.items()}
    full = {k: [json.loads(l) for l in open(os.path.join(OUT, v + ".jsonl"))] for k, v in srcs.items()}
    metrics = [
        ("native W", lambda r: r["W"]), ("native H", lambda r: r["H"]),
        ("dot pitch dp (native)", lambda r: r["dp"]),
        ("dp in MODEL input (x qwen)", lambda r: r["dp"] * r["qwen_scale"]),
        ("cell pitch P (native)", lambda r: r.get("P")), ("R = P/dp", lambda r: r.get("R")),
        ("diam/dp", lambda r: r["diam_over_dp"]), ("nlines", lambda r: r["nlines"]),
        ("ink width (cells)", lambda r: r["ink_w"]),
        ("margin left (P)", lambda r: r["margin_l"]), ("margin right (P)", lambda r: r["margin_r"]),
        ("margin top (P)", lambda r: r["margin_t"]), ("margin bottom (P)", lambda r: r["margin_b"]),
        ("word gap b (mean)", lambda r: r.get("gap_b_mean")),
        ("line pitch / P (multi-line)", lambda r: r.get("line_pitch_P")),
    ]
    print("quantiles p5 / p25 / p50 / p75 / p95   (reliable detections: n>=8 dots, q>=0.5)")
    for name, fn in metrics:
        print(f"\n{name}")
        for k, rows in data.items():
            qs = _q([fn(r) for r in rows])
            print(f"   {k:8s} n={len(rows):5d}  " + ("   ".join(f"{v:8.2f}" for v in qs) if qs else "(too few)"))
    print("\nshare single-line:", {k: round(float(np.mean([r['nlines'] == 1 for r in rows])), 3) for k, rows in data.items()})
    print("share with a dot touching/clipped by the border (any margin < 0.05 P):",
          {k: round(float(np.mean([min(r['margin_l'], r['margin_r'], r['margin_t'], r['margin_b']) < 0.05 for r in rows])), 3) for k, rows in data.items()})
    print("detection success (reliable / all):", {k: f"{len(data[k])}/{len(full[k])}" for k in data})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "legacy", "table", "remeasure"])
    ap.add_argument("--tag", default="rm1")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--caps", default=None, help="per-bucket max_text_chars, e.g. 24,60,24,120")
    ap.add_argument("--real_opt", action="append", default=[])
    ap.add_argument("--tags", nargs="*", default=[])
    a = ap.parse_args()
    {"gen": gen, "legacy": legacy, "table": table, "remeasure": remeasure}[a.cmd](a)
