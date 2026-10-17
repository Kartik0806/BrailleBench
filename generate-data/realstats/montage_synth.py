"""Montage of generated samples with their sampled real_style parameters (QA tool).
   python montage_synth.py <dataset_dir> <out.png> [start] [count]"""
import json, sys, textwrap
from PIL import Image, ImageDraw, ImageFont
d, out = sys.argv[1], sys.argv[2]
start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
count = int(sys.argv[4]) if len(sys.argv) > 4 else 16
recs = [json.loads(l) for l in open(f"{d}/labels/labels.jsonl")]
recs.sort(key=lambda r: r["id"]); recs = recs[start:start + count]
try: F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
except Exception: F = ImageFont.load_default()
tiles = []
for r in recs:
    im = Image.open(f"{d}/{r['image']}").convert("RGB")
    s = min(600 / im.width, 190 / im.height); disp = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS if s < 1 else Image.BICUBIC)
    rs = r["augmentation"].get("real_style", {})
    cap = [f'{r["id"]} {im.width}x{im.height} view x{s:.2f} | {r["source_text"][:48]!r}',
           f'P={rs.get("P_final_x")} R={rs.get("R")} b={rs.get("b")} dr={rs.get("dr")} aniso={rs.get("aniso")} lines={rs.get("n_lines")} m={rs.get("margins")}']
    t = Image.new("RGB", (610, disp.height + 40), (255, 255, 255)); t.paste(disp, (0, 0)); dr = ImageDraw.Draw(t)
    for i, c in enumerate(cap): dr.text((3, disp.height + 3 + 16 * i), c, fill=(0, 0, 0), font=F)
    tiles.append(t)
cols = 2
rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
H = sum(max(t.height for t in rr) + 8 for rr in rows)
M = Image.new("RGB", (cols * 620, H), (205, 205, 205)); y = 0
for rr in rows:
    for j, t in enumerate(rr): M.paste(t, (j * 620, y))
    y += max(t.height for t in rr) + 8
M.save(out); print(out, M.size)
