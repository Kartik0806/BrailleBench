"""Replay generate_dataset's per-sample seeding for many samples and report exceptions / blank
images. Usage: python stress_realmatch.py <n> <seed> <profile> [max_chars] [min_words] [max_words] [grade2]"""
import json, os, random, sys, traceback
import numpy as np, cv2
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import braille_gen_fineweb_v2 as g

N, SEED, PROFILE = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
MAXC = int(sys.argv[4]) if len(sys.argv) > 4 else 16
MINW = int(sys.argv[5]) if len(sys.argv) > 5 else 1
MAXW = int(sys.argv[6]) if len(sys.argv) > 6 else 2
G2 = float(sys.argv[7]) if len(sys.argv) > 7 else 0.0
TEXTS = os.environ.get("STRESS_TEXTS") or sys.exit("set STRESS_TEXTS to a text file with one text per line")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cfg = g.PipelineConfig(background_dir=os.path.join(ROOT, "background"), p_grade2=G2, real_profile=json.load(open(PROFILE)),
                       real_seed=SEED, p_background_compose=0.30, p_perspective=0.40, p_page_curl=0.10, p_elastic=0.15,
                       p_skew=0.20, max_skew_deg=8, p_vertical_flip=0, p_rotation=0, p_random_boundary=0.05,
                       p_paper_texture=0.70, p_stain=0.10, p_lighting=0.60, p_low_light=0.15, p_shadow_band=0.15,
                       p_noise=0.40, p_salt_pepper=0.10)
texts = g.get_texts_from_file(TEXTS, N, MINW, MAXW, SEED, 1, MAXC)

def one(i):
    out = {"i": i}
    try:
        seed = (cfg.real_seed * 1000003 + i * 7919 + 17) % (2**32)
        random.seed(seed); np.random.seed(seed)
        img, wr, bg, bgs, meta = g.generate_sample_realmatch(texts[i], cfg, i)
        h, w = img.shape[:2]
        out.update(h=h, w=w, std=float(img.std()), text=texts[i], rs=meta["real_style"])
        try:
            g.build_bbox_record(texts[i], wr, w, h, i, "x.jpg", bg, bgs, sample_meta=meta)
        except Exception as e:
            out["bbox_error"] = repr(e)[:150]
    except Exception as e:
        out["error"] = repr(e)[:200]; out["tb"] = traceback.format_exc()[-600:]
    return out

if __name__ == "__main__":
    with Pool(48) as p:
        res = p.map(one, range(N), chunksize=50)
    err = [r for r in res if "error" in r]; be = [r for r in res if "bbox_error" in r]
    ok = [r for r in res if "error" not in r]
    flat = [r for r in ok if r["std"] < 4.0]
    kept = np.array([r["rs"].get("ink_kept", 1.0) for r in ok]) if ok else np.array([1.0])
    print(f"   ink_kept pct 0.1/1/5/50 = {np.round(np.percentile(kept,[0.1,1,5,50]),2)}  samples with ink_kept<0.6: {(kept<0.6).sum()}")
    print(f"N={N}: generate exceptions={len(err)}  bbox exceptions={len(be)}  near-uniform images (std<4)={len(flat)}")
    for r in (err + be)[:5]: print(json.dumps({k: r.get(k) for k in ("i", "error", "bbox_error", "text", "tb")}, indent=1)[:900])
    for r in flat[:6]: print("flat:", r["i"], r["w"], r["h"], r["text"], "style:", {k: r["rs"][k] for k in ("P_target", "margins", "aniso", "class", "n_lines", "pool")})
    json.dump([r for r in err + be + flat], open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", f"stress_{SEED}.json"), "w"), default=str)
