"""
Measure scale / spacing statistics of braille IMAGES, in physical units.

Used to (a) characterise real-world pools and (b) check generated data with the
SAME estimator, so the two are comparable.

Per image it reports (all in native pixels unless noted):
  dp      dot pitch = median nearest-neighbour distance between detected dots
  diam    dot diameter (from the blob scale)
  P       cell pitch (dot-column spacing inside a cell + between cells)
  R       P / dp          (cell tightness; standard braille is ~2.4-2.6)
  a_dp    within-cell column gap / dp
  nlines  number of text lines (dot rows clustered)
  line_pitch_P  top-row to top-row distance of consecutive lines, in cell pitches
  margin_{l,r,t,b}   distance from the outermost dot edge to the image edge, in cell pitches
  word gaps (only when n_words is supplied): for each of the n_words-1 widest
      column gaps, the "blank advance" b = (gap + a)/P - 1, i.e. start-to-start
      distance across the word boundary minus one cell pitch, in cell pitches.
      b ~= 1 for standard braille (one blank cell).

Dot detection is a scale-space LoG (polarity chosen automatically), so it works
for bright-on-dark, dark-on-light and ring dots. It is a measurement tool, not
a perfect detector: every record carries `n` (dots) and `q` (fraction of dots
whose nearest neighbour is within 0.7-1.4x the median) so callers can filter.
"""
import math
import numpy as np
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

# Qwen3-VL image processor (patch 16, merge 2): the model sees the image after
# this resize, so small crops are up-scaled to >= 65536 px.
QWEN_FACTOR, QWEN_MIN_PIXELS, QWEN_MAX_PIXELS = 32, 65536, 16777216


def qwen_resize_scale(h, w):
    f = QWEN_FACTOR
    hb = max(f, round(h / f) * f)
    wb = max(f, round(w / f) * f)
    if hb * wb > QWEN_MAX_PIXELS:
        b = math.sqrt(h * w / QWEN_MAX_PIXELS)
        hb, wb = math.floor(h / b / f) * f, math.floor(w / b / f) * f
    elif hb * wb < QWEN_MIN_PIXELS:
        b = math.sqrt(QWEN_MIN_PIXELS / (h * w))
        hb, wb = math.ceil(h * b / f) * f, math.ceil(w * b / f) * f
    return math.sqrt(hb * wb / (h * w))


def detect_dots(gray, work=800, sig_min=0.9, sig_max=26, nsig=16):
    """Return (pts[x, y, sigma] in native px, work scale).

    Works at native resolution (never up-scales: up-scaling a small crop pushes big
    blurry dots outside the sigma range and the detector then latches onto grain);
    only images larger than `work` px are down-scaled."""
    h, w = gray.shape
    r = min(1.0, work / max(h, w))
    if r < 1.0:
        g = cv2.resize(gray, (max(8, int(w * r)), max(8, int(h * r))), interpolation=cv2.INTER_AREA).astype(np.float32)
    else:
        g = gray.astype(np.float32)
    g = g - cv2.GaussianBlur(g, (0, 0), max(g.shape) / 6)
    sigs = np.geomspace(sig_min, sig_max, nsig)
    best = None
    for pol in (+1, -1):
        stack = np.stack(
            [(-pol) * s * s * cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), s), cv2.CV_32F) for s in sigs]
        )
        mx = ndi.maximum_filter(stack, size=(3, 5, 5))
        pk = (stack == mx) & (stack > 0)
        vals = stack[pk]
        if vals.size == 0:
            continue
        thr = 0.35 * np.percentile(vals, 99)
        sel = pk & (stack > thr)
        idx = np.argwhere(sel)
        resp = stack[sel]
        score = np.sort(resp)[::-1][:25].mean() if len(resp) else 0
        if best is None or score > best[0]:
            best = (score, idx, resp)
    if best is None:
        return np.zeros((0, 3)), r
    _, idx, resp = best
    pts = np.stack([idx[:, 2], idx[:, 1], sigs[idx[:, 0]]], 1).astype(np.float64)
    order = np.argsort(-resp)
    ms = np.median(pts[order[:30], 2])
    keep = (pts[:, 2] > ms / 1.7) & (pts[:, 2] < ms * 1.7)
    pts, resp = pts[keep], resp[keep]
    order = np.argsort(-resp)
    chosen = []
    for i in order:
        if all(math.hypot(pts[i, 0] - pts[j, 0], pts[i, 1] - pts[j, 1])
               > 1.2 * math.sqrt(2) * max(pts[i, 2], pts[j, 2]) for j in chosen):
            chosen.append(i)
    pts = pts[chosen]
    pts[:, :3] /= r
    return pts, r


def _cluster_1d(vals, thr):
    vals = np.sort(vals)
    groups = [[vals[0]]]
    for v in vals[1:]:
        if v - np.mean(groups[-1]) > thr:
            groups.append([v])
        else:
            groups[-1].append(v)
    return groups


def measure_image(gray, n_words=None):
    """Measure one grayscale image. Returns a dict (None if no dots found)."""
    H, W = gray.shape
    pts, _ = detect_dots(gray)
    if len(pts) < 4:
        return None
    d, _ = cKDTree(pts[:, :2]).query(pts[:, :2], k=2)
    nn = d[:, 1]
    dp = float(np.median(nn))
    q = float(np.mean((nn > 0.7 * dp) & (nn < 1.4 * dp)))
    diam = float(np.median(pts[:, 2]) * 2 * math.sqrt(2))
    rec = dict(W=W, H=H, n=int(len(pts)), q=q, dp=dp, diam=diam, diam_over_dp=diam / dp,
               qwen_scale=qwen_resize_scale(H, W))

    # ---- text lines: cluster dot rows --------------------------------------
    row_groups = _cluster_1d(pts[:, 1], 0.45 * dp)
    row_y = np.array([np.mean(g) for g in row_groups])
    line_id = np.zeros(len(row_y), dtype=int)
    for i in range(1, len(row_y)):
        line_id[i] = line_id[i - 1] + (1 if row_y[i] - row_y[i - 1] > 1.75 * dp else 0)
    rec["nlines"] = int(line_id[-1] + 1) if len(row_y) else 1
    # assign every dot to its line
    pt_line = np.array([line_id[int(np.argmin(np.abs(row_y - y)))] for y in pts[:, 1]])

    # ---- column structure of the longest line ------------------------------
    best_line = int(np.bincount(pt_line).argmax())
    lp = pts[pt_line == best_line]
    if len(lp) >= 6:
        cols = _cluster_1d(lp[:, 0], 0.5 * dp)
        cx = np.array([np.mean(g) for g in cols])
        u = np.diff(cx) / dp
        rec["ncols"] = int(len(cx))
        small = u[u < 2.05]
        if len(small) >= 6:
            # two modes: within-cell (a ~ 1) and between-cell (b ~ 1.5)
            thr = np.median(small)
            lo, hi = small[small <= thr], small[small > thr]
            if len(lo) >= 3 and len(hi) >= 3:
                a, b = float(np.median(lo)), float(np.median(hi))
                if b - a > 0.25:
                    rec["a_dp"], rec["b_dp"] = a, b
                    rec["R"] = a + b
                    rec["P"] = (a + b) * dp
        if n_words and n_words >= 2 and "P" in rec:
            Pdp = rec["R"]
            gaps = np.sort(u)[::-1][: n_words - 1]
            rec["gap_b"] = [float((g + rec["a_dp"]) / Pdp - 1.0) for g in gaps]
            rec["gap_b_min"] = min(rec["gap_b"])
            rec["gap_b_mean"] = float(np.mean(rec["gap_b"]))

    # ---- line pitch (top row to top row of consecutive lines), in cell pitches ----
    P = rec.get("P", 2.5 * dp)
    if rec["nlines"] >= 2:
        tops = np.array([row_y[line_id == k].min() for k in range(rec["nlines"])])
        rec["line_pitch_P"] = float(np.median(np.diff(tops)) / P)

    # ---- margins in cell pitches ------------------------------------------
    rad = diam / 2
    x0, x1 = float((pts[:, 0] - rad).min()), float((pts[:, 0] + rad).max())
    y0, y1 = float((pts[:, 1] - rad).min()), float((pts[:, 1] + rad).max())
    rec.update(margin_l=x0 / P, margin_r=(W - x1) / P, margin_t=y0 / P, margin_b=(H - y1) / P,
               ink_w=(x1 - x0) / P, ink_h=(y1 - y0) / P, P_used=float(P))
    return rec


def measure_path(args):
    path, n_words = args
    try:
        im = cv2.imread(path)
        if im is None:
            return dict(path=path, error="unreadable")
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        r = measure_image(g, n_words)
        if r is None:
            return dict(path=path, error="no_dots", W=g.shape[1], H=g.shape[0])
        r["path"] = path
        return r
    except Exception as e:  # keep batch jobs alive
        return dict(path=path, error=repr(e)[:120])
