"""Run measure_real over the real image pools -> out/real_<pool>.jsonl"""
import json, os, sys, glob
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from measure_real import measure_path

EVAL = os.environ.get("EVAL_DIR") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "eval"))  # default: ../eval next to the repo
OUT = os.path.join(os.path.dirname(__file__), "out")

def nwords(t):
    return len((t or "").split())

def pools():
    p = {}
    # R3: held-out test images (labelled) + the unlabelled remainder of data/combined
    lab = {}
    for l in open(f"{EVAL}/data/combined.jsonl"):
        l = l.strip()
        if l:
            d = json.loads(l); lab[os.path.basename(d["image_path"])] = d.get("ground_truth")
    items = []
    for f in sorted(glob.glob(f"{EVAL}/data/combined/*")):
        b = os.path.basename(f); items.append((f, nwords(lab[b]) if b in lab else None, lab.get(b)))
    p["eval_combined"] = items
    # R2: book pages (unlabelled, disjoint from the test set)
    p["book_pages"] = [(f, None, None) for f in sorted(glob.glob(f"{EVAL}/data/Braille book dataset/*"))]
    # R1: new_data (labelled single words / short phrases, disjoint from the test set)
    items = []
    for l in open(f"{EVAL}/new_data/labels.jsonl"):
        d = json.loads(l); items.append((f"{EVAL}/new_data/{d['image_path']}", nwords(d["ground_truth"]), d["ground_truth"]))
    p["new_data"] = items
    return p

if __name__ == "__main__":
    which = sys.argv[1:] or list(pools())
    P = pools()
    for name in which:
        items = P[name]
        with Pool(40) as pool:
            res = pool.map(measure_path, [(f, nw) for f, nw, _ in items], chunksize=4)
        with open(os.path.join(OUT, f"real_{name}.jsonl"), "w") as fh:
            for (f, nw, t), r in zip(items, res):
                r["pool"] = name; r["n_words"] = nw; r["truth"] = t
                fh.write(json.dumps(r) + "\n")
        ok = sum("error" not in r for r in res)
        print(name, len(items), "measured ok:", ok, flush=True)
