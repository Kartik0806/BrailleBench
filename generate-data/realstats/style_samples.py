"""Render example images of the optional styles (ghost / strip / closeup) for review.
   STYLE_TEXTS=texts.txt python style_samples.py <out_dir> [n_per_style]"""
import json, os, random, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import braille_gen_fineweb_v2 as g
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
os.makedirs(OUT, exist_ok=True)
TEXTS = os.environ.get("STYLE_TEXTS") or sys.exit("set STYLE_TEXTS to a text file with one text per line")
prof = json.load(open(os.path.join(ROOT, "configs", "realmatch_profile_v1_noeval.json")))
base = dict(background_dir=os.path.join(ROOT, "background"), real_profile=prof, strip_punct_tokens=True,
            p_background_compose=0.30, p_perspective=0.40, p_page_curl=0.10, p_elastic=0.15, p_skew=0.20,
            max_skew_deg=8, p_vertical_flip=0, p_random_boundary=0.05, p_paper_texture=0.70, p_stain=0.10,
            p_lighting=0.60, p_low_light=0.10, p_shadow_band=0.10, p_noise=0.30, p_salt_pepper=0.02)
styles = {"ghost": ({"p_ghost": 1.0}, 1, 4, 30), "strip": ({"p_strip": 1.0}, 2, 25, 140),
          "closeup": ({"p_closeup": 1.0}, 1, 1, 14),
          "hollow": ({"p_hollow": 1.0, "p_closeup": 0.3}, 1, 3, 24),
          "dynamic_spacing": ({"p_dynamic_spacing": 1.0}, 2, 8, 60)}
only = os.environ.get("STYLES")
if only: styles = {k: v for k, v in styles.items() if k in only.split(",")}
for name, (opts, mnw, mxw, mxc) in styles.items():
    cfg = g.PipelineConfig(**base); cfg.real_opts.update(opts); cfg.real_seed = 31
    texts = g.get_texts_from_file(TEXTS, N, mnw, mxw, 5, 1, mxc, True)
    tiles = []
    for i, t in enumerate(texts):
        for attempt in range(6):
            random.seed(1000 + i * 31 + attempt); np.random.seed(1000 + i * 31 + attempt)
            cfg.p_grade2 = 0.5
            try:
                img, wr, bg, bgs, meta = g.generate_sample_realmatch(t, cfg, i); break
            except Exception as e:
                img = None
        if img is None: continue
        rs = meta["real_style"]
        cv2.imwrite(os.path.join(OUT, f"{name}_{i:02d}.jpg"), img)
        s = min(560 / img.shape[1], 170 / img.shape[0]); v = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        tile = np.full((v.shape[0] + 34, 570, 3), 255, np.uint8); tile[:v.shape[0], :v.shape[1]] = v
        cv2.putText(tile, f"{img.shape[1]}x{img.shape[0]} P={rs['P_final_x']:.0f} {rs['style']}{' lowres' if rs.get('closeup_lowres') else ''} R={rs['R']:.2f} b={rs['b']:.2f}", (3, v.shape[0] + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 160), 1)
        cv2.putText(tile, t[:70], (3, v.shape[0] + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        tiles.append(tile)
    rows = [tiles[k:k + 2] for k in range(0, len(tiles), 2)]
    M = np.full((sum(max(t.shape[0] for t in r) + 6 for r in rows), 1150, 3), 210, np.uint8); y = 0
    for r in rows:
        for j, t in enumerate(r): M[y:y + t.shape[0], j * 580:j * 580 + 570] = t
        y += max(t.shape[0] for t in r) + 6
    cv2.imwrite(os.path.join(OUT, f"montage_{name}.png"), M)
    print(name, len(tiles), "->", os.path.join(OUT, f"montage_{name}.png"))
