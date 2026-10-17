"""Replay the run script's bucket mix through generate_sample_realmatch and report image size /
visual tokens / cap hits / long-text pitch.   python size_budget_check.py [n_total]"""
import json, math, os, random, sys
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import braille_gen_fineweb_v2 as g
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEXTS = os.environ.get("STRESS_TEXTS") or sys.exit("set STRESS_TEXTS to a text file with one text per line")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
BUCKETS = [(0.35, 0.0, 1, 2, 16), (0.15, 0.0, 3, 6, 48), (0.20, 1.0, 1, 3, 16), (0.30, 1.0, 4, 50, 120)]
PROF = json.load(open(os.path.join(ROOT, "configs", "realmatch_profile_v1_noeval.json")))
jobs = []
for bi, (share, pg2, mnw, mxw, mxc) in enumerate(BUCKETS):
    n = int(N * share)
    texts = g.get_texts_from_file(TEXTS, n, mnw, mxw, 77 + bi, 1, mxc, True)
    jobs += [(bi, pg2, i, t) for i, t in enumerate(texts)]

def tok(w, h):
    f = 32; hb = max(f, round(h / f) * f); wb = max(f, round(w / f) * f)
    if hb * wb < 65536:
        b = math.sqrt(65536 / (h * w)); hb = math.ceil(h * b / f) * f; wb = math.ceil(w * b / f) * f
    return (hb // f) * (wb // f)

def one(job):
    bi, pg2, i, text = job
    cfg = g.PipelineConfig(background_dir=os.path.join(ROOT, "background"), p_grade2=pg2, real_profile=PROF,
                           real_seed=900 + bi, strip_punct_tokens=True, p_background_compose=0.30, p_perspective=0.40,
                           p_page_curl=0.10, p_elastic=0.15, p_skew=0.20, max_skew_deg=8, p_vertical_flip=0,
                           p_random_boundary=0.05, p_paper_texture=0.70, p_stain=0.10, p_lighting=0.60,
                           p_low_light=0.15, p_shadow_band=0.15, p_noise=0.40, p_salt_pepper=0.10)
    for attempt in range(6):
        seed = (cfg.real_seed * 1000003 + i * 7919 + 17 + attempt * 104729) % (2**32)
        random.seed(seed); np.random.seed(seed)
        try:
            img, wr, bg, bgs, meta = g.generate_sample_realmatch(text, cfg, i)
            h, w = img.shape[:2]; rs = meta["real_style"]
            return dict(w=w, h=h, tok=tok(w, h), capped=rs["size_capped"], P=rs["P_final_x"], cells=rs["n_cells_est"], border=rs["ink_border_px"])
        except Exception:
            continue
    return None

if __name__ == "__main__":
    with Pool(48) as p:
        res = [r for r in p.map(one, jobs, chunksize=20) if r]
    t = np.array([r["tok"] for r in res]); W = np.array([r["w"] for r in res]); H = np.array([r["h"] for r in res])
    print(f"samples {len(res)}/{len(jobs)}")
    print(f"visual tokens p50/p95/p99/max = {np.percentile(t,[50,95,99,100]).astype(int)}   share > 1024 tokens: {np.mean(t>1024):.2%}")
    print(f"longest side max = {max(W.max(), H.max())} px   area max = {(W*H).max()/1e6:.2f} MP   share size-capped = {np.mean([r['capped'] for r in res]):.2%}")
    long = [r for r in res if r["cells"] >= 32]
    Pl = np.array([r["P"] for r in long])
    print(f">=32-cell texts: n={len(long)}  P min/p1/p50 = {Pl.min():.1f}/{np.percentile(Pl,1):.1f}/{np.median(Pl):.1f}  share below the 22 px floor: {np.mean(Pl<21.99):.2%}  (capped ones: {sum(r['capped'] for r in long)})")
    print(f"min ink_border_px = {min(r['border'] for r in res):.2f}")
