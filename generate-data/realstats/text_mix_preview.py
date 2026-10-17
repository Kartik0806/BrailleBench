"""Preview the word-count mix a bucket configuration produces (FineWeb texts only, no rendering).
   source louis_env.sh; python text_mix_preview.py <name> '<share,p_grade2,minw,maxw,maxc;...>' [n_per_bucket] [--strip]"""
import json, os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import braille_gen_fineweb_v2 as g

name, spec = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 1500
strip = "--strip" in sys.argv
buckets = [tuple(float(x) for x in b.split(",")) for b in spec.split(";")]
rows = []
for bi, (share, pg2, mnw, mxw, mxc) in enumerate(buckets):
    texts = g.get_data("HuggingFaceFW/fineweb", "sample-10BT", n, 1, int(mxc), int(mnw), int(mxw), 4242 + bi, strip)
    rows.append((share, texts))
def hist(ws_counts_weights):
    bins = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3), ("4-5", 4, 5), ("6-9", 6, 9), ("10-15", 10, 15), ("16+", 16, 10**6)]
    out = {}
    for lab, lo, hi in bins:
        out[lab] = sum(w * np.mean([(lo <= len(t.split()) <= hi) for t in texts]) for w, texts in ws_counts_weights)
    return out
h = hist(rows)
tot = sum(w for w, _ in rows)
res = {k: round(100 * v / tot, 1) for k, v in h.items()}
lead = sum(w * np.mean([bool(t.split()) and not re.search(r"[A-Za-z0-9]", t.split()[0]) for t in texts]) for w, texts in rows) / tot
print(json.dumps({"config": name, "strip": strip, "word_count_pct": res, "leading_punct_token_pct": round(100 * lead, 2),
                  "chars_median_by_bucket": [float(np.median([len(t) for t in texts])) for _, texts in rows]}))
