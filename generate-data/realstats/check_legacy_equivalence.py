"""
Prove that braille_gen_fineweb_v2.py WITHOUT --real_profile is byte-identical to
braille_gen_fineweb.py: same seeds + same texts -> identical image bytes and labels.

    PYTHONPATH=<liblouis>/python LOUIS_TABLEPATH=<liblouis>/tables \
        python check_legacy_equivalence.py [N]
"""
import hashlib, importlib, json, os, queue, random, shutil, sys, tempfile
import numpy as np

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
sys.path.append(os.path.join(os.path.dirname(SRC), "backup", "src"))  # the legacy braille_gen_fineweb.py lives in backup/
N = int(sys.argv[1]) if len(sys.argv) > 1 else 24
TEXTS = [
    "to do", "Heritage Version", "the quick brown fox jumps over the lazy dog", "I put my",
    "illness, family emergencies, and more", "CryptoForge encryption software.", "a", "Hello, World!",
    "In 1998 the 3 teams scored 45 points.", "problematic first-born sons", "of the", "Where is it?",
    "A longer line of text that should wrap around when the page is narrow enough to need it, "
    "so that multi-line layout and line bounding boxes are exercised too.",
]

def run(modname, outdir, seed):
    mod = importlib.import_module(modname)
    cfg = mod.PipelineConfig(output_dir=outdir, background_dir=os.path.join(os.path.dirname(SRC), "background"),
                             p_grade2=0.5, p_rotation=0.0, p_vertical_flip=0.1, p_random_boundary=0.3,
                             p_random_resolution=0.3)
    os.makedirs(outdir, exist_ok=True)
    random.seed(seed); np.random.seed(seed)
    q = queue.Queue()
    texts = [TEXTS[i % len(TEXTS)] for i in range(N)]
    mod.generate_dataset(texts, 1, cfg, False, q, 0, outdir, None, False)
    recs = []
    while not q.empty():
        recs.append(q.get())
    recs.sort(key=lambda r: r["id"])
    return recs

def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()

if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="legacy_eq_")
    try:
        a = run("braille_gen_fineweb", os.path.join(tmp, "v1"), 123)
        b = run("braille_gen_fineweb_v2", os.path.join(tmp, "v2"), 123)
        assert len(a) == len(b) == N, (len(a), len(b))
        bad_img = [r["image"] for r in a
                   if md5(os.path.join(tmp, "v1", os.path.basename(r["image"]))) != md5(os.path.join(tmp, "v2", os.path.basename(r["image"])))]
        bad_lab = [x["id"] for x, y in zip(a, b) if json.dumps(x, sort_keys=True) != json.dumps(y, sort_keys=True)]
        print(f"samples={N}  image files differing: {len(bad_img)}  label records differing: {len(bad_lab)}")
        print("EQUIVALENT" if not bad_img and not bad_lab else "DIFFERENT: " + str((bad_img[:3], bad_lab[:3])))
        sys.exit(0 if not bad_img and not bad_lab else 1)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
