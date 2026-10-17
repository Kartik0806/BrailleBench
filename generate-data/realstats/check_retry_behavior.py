"""Failure injection: generate_dataset in real-matching mode must (a) retry a failing sample and still
emit exactly one label per sample, (b) never leave an image without a label, even if a sample always fails.
    source louis_env.sh; python check_retry_behavior.py"""
import json, os, queue, random, shutil, sys, tempfile
import numpy as np
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
import braille_gen_fineweb_v2 as g

prof = json.load(open(os.path.join(os.path.dirname(SRC), "configs", "realmatch_profile_v1_noeval.json")))
tmp = tempfile.mkdtemp(prefix="retry_")
orig = g.generate_sample_realmatch
calls = {}

def flaky(text, cfg, idx):
    calls[idx] = calls.get(idx, 0) + 1
    if idx % 5 == 0 and calls[idx] <= 2:          # fails twice, then works
        raise ValueError("injected failure")
    if idx == 7:                                   # never works
        raise ValueError("injected permanent failure")
    img, wr, bg, bgs, meta = orig(text, cfg, idx)
    if idx == 3 and calls[idx] == 1:               # fails AFTER rendering, inside build_bbox_record
        wr = [dict(r, quad=np.full((4, 2), np.nan, np.float32)) for r in wr]
        wr[0]["quad"] = "broken"
    return img, wr, bg, bgs, meta

import braille_gen.dataset as _dataset
_dataset.generate_sample_realmatch = flaky   # generate_dataset looks the function up in its own module
cfg = g.PipelineConfig(output_dir=tmp, background_dir=os.path.join(os.path.dirname(SRC), "background"), real_profile=prof, real_seed=1)
q = queue.Queue()
texts = ["hello world", "a", "the quick brown fox", "go", "it is"] * 4
g.generate_dataset(texts, 1, cfg, False, q, 0, tmp, None, False)
recs = []
while not q.empty():
    recs.append(q.get())
ids = sorted(int(r["id"]) for r in recs)
imgs = sorted(int(f[7:13]) for f in os.listdir(tmp) if f.endswith(".jpg"))
expect = [i for i in range(20) if i != 7]
print("labels:", len(recs), "images:", len(imgs), "| sample 7 (always fails) absent from both:", 7 not in ids and 7 not in imgs)
print("every label has an image and vice versa:", ids == imgs, "| all samples except 7 present:", ids == expect)
print("attempt used for idx 0 / 3 / 5:", {int(r['id']): r['augmentation']['real_style']['attempt'] for r in recs if int(r['id']) in (0, 3, 5)})
ok = ids == imgs == expect
print("PASS" if ok else "FAIL")
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if ok else 1)
