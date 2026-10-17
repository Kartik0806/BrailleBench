"""Real vs legacy vs real-matched samples at IDENTICAL pixel scale (no resizing, crop to 590x200).
   python compare_montage.py <real_dir> <legacy_dir> <synth_dir> <out.png> [n] [seed]"""
import glob, os, random, sys
from PIL import Image, ImageDraw, ImageFont
real_dir, leg_dir, syn_dir, out = sys.argv[1:5]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 10
random.seed(int(sys.argv[6]) if len(sys.argv) > 6 else 0)
W, H = 590, 200
F = ImageFont.load_default()
def pick(d, k, pat="*"):
    fs = sorted(glob.glob(os.path.join(d, pat)))
    return random.sample(fs, min(k, len(fs)))
cols = [("REAL (eval, native px)", pick(real_dir, n)), ("LEGACY synthetic (native px, cropped)", pick(leg_dir, n)), ("REAL-MATCHED synthetic (native px, cropped)", pick(syn_dir, n))]
M = Image.new("RGB", (3 * (W + 8), n * (H + 18) + 22), (200, 200, 200)); d = ImageDraw.Draw(M)
for ci, (title, fs) in enumerate(cols):
    d.text((ci * (W + 8) + 4, 4), title, fill=(0, 0, 0), font=F)
    for ri, f in enumerate(fs):
        im = Image.open(f).convert("RGB"); w, h = im.size
        tile = Image.new("RGB", (W, H), (255, 255, 255)); tile.paste(im.crop((0, 0, min(w, W), min(h, H))), (0, 0))
        M.paste(tile, (ci * (W + 8), 22 + ri * (H + 18)))
        d.text((ci * (W + 8) + 2, 22 + ri * (H + 18) + H + 3), f"{os.path.basename(f)[:22]} {w}x{h}", fill=(0, 0, 0), font=F)
M.save(out); print(out, M.size)
