"""Sanity-check measured cell pitch against the known text: for single-line images with a
ground-truth label, ink width / P should equal the braille cell count (grade 1 or 2).
    source louis_env.sh; python check_pitch_vs_labels.py [pool ...]"""
import json, os, re, sys
import numpy as np
import louis
HERE = os.path.dirname(os.path.abspath(__file__))

def cells(text, grade):
    tbl = ["braille-patterns.cti", "en-ueb-g1.ctb" if grade == 1 else "en-ueb-g2.ctb"]
    try:
        return len(louis.translateString(tbl, text))
    except Exception:
        return None

def check(pool, edges=(0, 6, 9, 13, 20, 40, 999), prefix="real_"):
    R = [json.loads(l) for l in open(os.path.join(HERE, "out", f"{prefix}{pool}.jsonl"))]
    out = []
    for r in R:
        if "error" in r or r["n"] < 8 or r["q"] < 0.5 or "P" not in r or r["nlines"] != 1 or not r.get("truth"):
            continue
        t = r["truth"]
        if re.search(r"[<>]", t) or len(t) > 60:
            continue
        n1, n2 = cells(t, 1), cells(t, 2)
        if not n1:
            continue
        est = r["ink_w"] + 0.45
        N = min((abs(est - n), n) for n in (n1, n2))[1]
        out.append((r["P"], r["diam_over_dp"], est / N))
    a = np.array(out)
    ok = np.abs(a[:, 2] - 1) <= 0.2
    print(f"\n{pool}: single-line labelled reliable records {len(a)}; consistent (|ink_w/N-1|<=0.2): {ok.mean():.3f}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.sum() >= 8:
            print(f"  P in [{lo:>3},{hi:>3}) n={m.sum():4d} consistent={ok[m].mean():.2f}  median ink/N={np.median(a[m, 2]):.2f}")

if __name__ == "__main__":
    for p in (sys.argv[1:] or ["eval_combined", "new_data"]):
        check(p)
