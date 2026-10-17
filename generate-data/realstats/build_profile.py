"""
Turn out/real_<pool>.jsonl (measure_real.py output) into a generator profile.

    python build_profile.py --out ../../configs/realmatch_profile_v1.json
    python build_profile.py --exclude eval_combined --out ../../configs/realmatch_profile_v1_noeval.json

The profile is a bank of real "style records" (cell pitch, margins, dot size,
line structure) plus a bank of measured word gaps. braille_gen_fineweb_v2.py
resamples these records (with jitter) so generated images cover the real
joint distribution instead of independent hand-set ranges.

Record fields (all per real image):
  P     cell pitch in native px           (dot-column spacing a+b; falls back to 2.5*dp)
  dr    dot diameter / dot pitch
  m     [left, right, top, bottom] margins in cell pitches (negative = dots clipped)
  nl    number of text lines
  c     total cells in the image (ink width x lines)
  lp    line pitch in cell pitches (multi-line records only)
  w     ink width in cell pitches (~cells per line)
  H, W  native image size (for reference/validation only)
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
POOLS = ("eval_combined", "new_data", "book_pages")


def load(pool):
    with open(os.path.join(HERE, "out", f"real_{pool}.jsonl")) as fh:
        return [json.loads(l) for l in fh]


try:  # label-consistency filter needs liblouis (source louis_env.sh); skipped otherwise
    import louis
    import re
except ImportError:  # pragma: no cover
    louis = None


def _cells(text, grade):
    tbl = ["braille-patterns.cti", "en-ueb-g1.ctb" if grade == 1 else "en-ueb-g2.ctb"]
    try:
        return len(louis.translateString(tbl, text))
    except Exception:
        return None


def consistent(r, tol=0.3):
    """Single-line images with a label: ink width / P must equal the braille cell count.
    Returns True when it cannot be checked (unlabelled, multi-line, untranslatable)."""
    t = r.get("truth")
    if louis is None or not t or r.get("nlines") != 1 or "P" not in r or len(t) > 60 or re.search(r"[<>]", t):
        return True
    n1, n2 = _cells(t, 1), _cells(t, 2)
    if not n1:
        return True
    est = r["ink_w"] + 0.45
    N = min((abs(est - n), n) for n in (n1, n2))[1]
    return abs(est / N - 1.0) <= tol


def clean(r):
    """Reliable detections only; returns a bank record or None."""
    if "error" in r or r["n"] < 8 or r["q"] < 0.5 or not consistent(r):
        return None
    P = r.get("P") or 2.5 * r["dp"]
    if not np.isfinite(P) or P < 8.0 or P > 400:
        return None
    m = [r["margin_l"], r["margin_r"], r["margin_t"], r["margin_b"]]
    if not all(np.isfinite(m)):
        return None
    return dict(
        P=round(float(P), 2),
        dr=round(float(np.clip(r["diam_over_dp"], 0.3, 1.1)), 3),
        m=[round(float(np.clip(x, -0.35, 3.5)), 3) for x in m],
        nl=int(r["nlines"]),
        c=round(float(np.clip(r["ink_w"] * r["nlines"], 1.0, 3000.0)), 1),   # ~ total cells
        lp=(round(float(np.clip(r["line_pitch_P"], 1.4, 8.0)), 2) if r.get("line_pitch_P") is not None else None),
        w=round(float(np.clip(r["ink_w"], 1.0, 60.0)), 2),
        H=int(r["H"]), W=int(r["W"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="pools to leave out (e.g. the test pool)")
    args = ap.parse_args()

    prof = {"version": 1, "pools": {}, "gap_b": {}, "source_pools": [], "excluded_pools": args.exclude}
    for pool in POOLS:
        if pool in args.exclude:
            continue
        rows = load(pool)
        bank = [c for c in (clean(r) for r in rows) if c]
        prof["pools"][pool] = bank
        prof["source_pools"].append(pool)
        gb = [r["gap_b_mean"] for r in rows
              if "error" not in r and r["n"] >= 8 and r["q"] >= 0.5 and "gap_b_mean" in r
              and np.isfinite(r["gap_b_mean"]) and consistent(r)]
        prof["gap_b"][pool] = [round(float(np.clip(g, 0.15, 3.0)), 3) for g in gb]
        print(f"{pool:14s} bank={len(bank):5d}  gap_b={len(gb):4d}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(prof, fh, separators=(",", ":"))
    print("wrote", args.out, os.path.getsize(args.out) // 1024, "KiB")


if __name__ == "__main__":
    main()
