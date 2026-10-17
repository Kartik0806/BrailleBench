"""
Merge generated datasets into one train.py-compatible dataset WITHOUT copying images.

    python merge_realmatch_datasets.py --out data_realmatch_combined base=data_realmatch_200k addon=data_realmatch_addon

For every split present in all inputs (train, val) it writes
    <out>/<split>/images/<tag>_<original name>   relative symlink to the source image
    <out>/<split>/labels/labels.jsonl            all records, "image" and "id" prefixed with <tag>
and checks that every record's image exists, ids are unique, and the record count equals the sum of
the inputs. The labels file is written to a temp name and renamed last, so a present labels.jsonl
means the merge finished.
"""
import argparse, json, os, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="+", help="tag=dataset_root")
    a = ap.parse_args()
    inputs = [tuple(x.split("=", 1)) for x in a.inputs]
    if os.path.exists(a.out):
        sys.exit(f"{a.out} already exists")
    for split in ("train", "val"):
        srcs = [(t, os.path.join(r, split)) for t, r in inputs]
        if not all(os.path.isfile(os.path.join(d, "labels", "labels.jsonl")) for _, d in srcs):
            print(f"skip {split}: not present in every input")
            continue
        img_out = os.path.join(a.out, split, "images")
        lab_dir = os.path.join(a.out, split, "labels")
        os.makedirs(img_out); os.makedirs(lab_dir)
        tmp = os.path.join(lab_dir, "labels.jsonl.tmp")
        n_total, ids, counts = 0, set(), {}
        with open(tmp, "w", encoding="utf-8") as fo:
            for tag, d in srcs:
                n = 0
                with open(os.path.join(d, "labels", "labels.jsonl"), encoding="utf-8") as fi:
                    for line in fi:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        src_img = os.path.join(d, r["image"])
                        if not os.path.exists(src_img):
                            sys.exit(f"missing image {src_img}")
                        name = f"{tag}_{os.path.basename(r['image'])}"
                        os.symlink(os.path.relpath(src_img, img_out), os.path.join(img_out, name))
                        r["image"] = f"images/{name}"
                        r["id"] = f"{tag}_{r.get('id', n)}"
                        r["source_dataset"] = tag
                        if r["id"] in ids:
                            sys.exit(f"duplicate id {r['id']}")
                        ids.add(r["id"])
                        fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                        n += 1
                counts[tag] = n
                n_total += n
        n_links = len(os.listdir(img_out))
        if n_links != n_total:
            sys.exit(f"{split}: {n_links} image links for {n_total} records")
        os.replace(tmp, os.path.join(lab_dir, "labels.jsonl"))
        print(f"{split}: {n_total} records ({', '.join(f'{k} {v}' for k, v in counts.items())}) -> {a.out}/{split}")


if __name__ == "__main__":
    main()
