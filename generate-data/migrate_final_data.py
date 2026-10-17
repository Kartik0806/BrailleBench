#!/usr/bin/env python3
"""Merge the stage datasets into one numbered master dataset.

    python src/migrate_final_data.py --final_root final_data \
        --stages pretrain midtrain posttrain --out final_data/master --mode copy

Every stage (<final_root>/<stage>/{images/, labels/labels.jsonl, llava.json}) is validated
before anything is written. Samples are then renumbered globally (stage order, or a seeded
shuffle of the concatenation), images are transferred with --mode, the master labels.jsonl
and llava.json are written and re-read, and manifest.json is written last: it is the
completion marker of the master dataset.

The master follows each stage's labels.jsonl line order (workers write labels in completion
order, so that is not id order). --shuffle_seed orders by sha256(f"{seed}|{stage}|{orig_id}"),
a seeded permutation that does not depend on Python's RNG or on the stage's line order.

If anything fails after writing started (I/O error, verification mismatch, Ctrl-C, SIGTERM),
every file this run created is removed again; only SIGKILL/power loss can leave a partial
master, and it is recognisable by the missing manifest.json.
--mode move only removes source images after the master (manifest included) is complete.
hardlink/symlink share data with the stage images: rewriting a stage image in place changes
the master too.

Exit codes: 0 ok, 1 validation or runtime failure, 2 usage error, 130 interrupted (Ctrl-C/SIGTERM).
"""

import argparse
import errno
import fnmatch
import hashlib
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

DEFAULT_STAGES = ("pretrain", "midtrain", "posttrain")
MODES = ("copy", "hardlink", "symlink", "move")
IMAGE_GLOB = "sample_*.jpg"
MANIFEST_VERSION = 1
LENGTH_BUCKETS = (
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4", 4, 4),
    ("5", 5, 5),
    ("6-8", 6, 8),
    ("9-12", 9, 12),
    ("13-20", 13, 20),
    ("21-50", 21, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
    ("201-450", 201, 450),
    ("451+", 451, None),
)
LLAVA_KEYS = frozenset({"id", "image", "stage", "category", "conversations"})
MASTER_LLAVA_KEYS = frozenset({"id", "image", "stage", "category", "orig_id", "conversations"})
ADDED_LABEL_KEYS = ("stage", "orig_id", "orig_image")
TEXT_KIND_DEFINITIONS = {
    "digits_only": "source_text matches [0-9]+",
    "alnum_only": "source_text matches [A-Za-z0-9]+ (no space or punctuation; includes digits_only)",
    "single_char": "len(source_text) == 1",
    "multi_word": "len(source_text.split()) >= 2",
    "multi_line": "'\\n' in source_text",
}
DIGITS_RE = re.compile(r"[0-9]+")
ALNUM_RE = re.compile(r"[A-Za-z0-9]+")
LINE_SEPARATORS = tuple((chr(code), f"\\u{code:04x}") for code in (0x2028, 0x2029, 0x85))  # keep one JSON entry per splitlines() line
CHUNK = 1024
MAX_WORKERS = 64


class MigrateError(Exception):
    """A runtime failure after validation passed (I/O error, verification mismatch)."""


class UsageError(Exception):
    """Invalid argument combination."""


def log(message: str) -> None:
    print(f"[migrate] {message}", flush=True)


def md5_file(path: str, block: int = 1 << 20) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as fh:
        while True:
            data = fh.read(block)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def errno_name(exc: OSError) -> str:
    return errno.errorcode.get(exc.errno or 0, "OSERROR")


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def category_of(record: dict) -> str:
    source = record.get("source")
    stratum = source.get("stratum") if isinstance(source, dict) else None
    return stratum if isinstance(stratum, str) else ""


def valid_image_ref(image: str) -> bool:
    parts = image.split("/")
    return len(parts) == 2 and parts[0] == "images" and fnmatch.fnmatchcase(parts[1], IMAGE_GLOB)


def length_bucket(n: int) -> str:
    for label, lo, hi in LENGTH_BUCKETS:
        if n >= lo and (hi is None or n <= hi):
            return label
    raise ValueError(f"no length bucket for {n}")


def probe_image(path: str) -> tuple[int, Optional[str]]:
    """Return (size, problem): the file must be non-empty and start/end like a complete JPEG."""
    try:
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            if size == 0:
                return 0, "empty file"
            if fh.read(3) != b"\xff\xd8\xff":
                return size, "not a JPEG (no SOI marker at the start)"
            fh.seek(max(0, size - 64))
            if b"\xff\xd9" not in fh.read(64):
                return size, "truncated JPEG (no EOI marker at the end)"
    except OSError as exc:
        return -1, f"unreadable: {exc}"
    return size, None


class Problems:
    """Collects validation problems; reports a capped sample per (where, kind)."""

    def __init__(self, limit: int = 8):
        self.limit = limit
        self.groups: dict[tuple[str, str], list[str]] = {}

    def add(self, where: str, kind: str, detail: str) -> None:
        self.groups.setdefault((where, kind), []).append(detail)

    @property
    def count(self) -> int:
        return sum(len(items) for items in self.groups.values())

    def __bool__(self) -> bool:
        return bool(self.groups)

    def report(self) -> None:
        for (where, kind), items in self.groups.items():
            print(f"[migrate] ERROR {where}: {kind} ({len(items)})", file=sys.stderr)
            for detail in items[: self.limit]:
                print(f"[migrate]     {detail}", file=sys.stderr)
            if len(items) > self.limit:
                print(f"[migrate]     ... and {len(items) - self.limit} more", file=sys.stderr)


class Progress:
    """Prints one line per 10% (only for runs of at least 2000 items)."""

    def __init__(self, label: str, total: int):
        self.label = label
        self.total = total
        self.done = 0
        self.next_mark = 10
        self.start = time.time()
        self.lock = threading.Lock()

    def tick(self) -> None:
        if self.total < 2000:
            return
        with self.lock:
            self.done += 1
            pct = 100 * self.done // self.total
            if pct >= self.next_mark:
                log(f"{self.label}: {self.done}/{self.total} ({pct}%) {time.time() - self.start:.1f}s")
                self.next_mark = (pct // 10 + 1) * 10


def parallel_map(fn: Callable, items: list, workers: int, label: str) -> list:
    """Ordered map over a thread pool; the first failure (in item order) is re-raised after in-flight work ends."""
    progress = Progress(label, len(items))
    results: list = []
    if workers <= 1:
        for item in items:
            results.append(fn(item))
            progress.tick()
        return results
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for start in range(0, len(items), CHUNK):
            futures = [pool.submit(fn, item) for item in items[start : start + CHUNK]]
            for future in futures:
                results.append(future.result())
                progress.tick()
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return results


# ---------------------------------------------------------------------------
# Stage validation
# ---------------------------------------------------------------------------
@dataclass
class Stage:
    name: str
    root: str
    real_root: str
    records: list[dict]
    llava: list[dict]
    sizes: list[int]
    labels_md5: str
    llava_md5: str
    text_plan_md5: Optional[str]

    def image_path(self, index: int, resolved: bool = False) -> str:
        base = self.real_root if resolved else self.root
        return os.path.join(base, *self.records[index]["image"].split("/"))


def read_labels(path: str, name: str, problems: Problems) -> Optional[list[dict]]:
    """Strict JSONL parse: a blank or malformed line is a problem (line i + 1 is record i)."""
    records: list[dict] = []
    bad = 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    problems.add(name, "blank line in labels.jsonl", f"line {line_no}")
                    bad += 1
                    continue
                try:
                    rec = json.loads(line)
                except ValueError as exc:
                    problems.add(name, "invalid JSON in labels.jsonl", f"line {line_no}: {exc}")
                    bad += 1
                    continue
                if not isinstance(rec, dict):
                    problems.add(name, "labels.jsonl record is not an object", f"line {line_no}")
                    bad += 1
                    continue
                records.append(rec)
    except (OSError, UnicodeDecodeError) as exc:
        problems.add(name, "labels.jsonl unreadable", f"{path}: {exc}")
        return None
    if not records and not bad:
        problems.add(name, "labels.jsonl has no records", path)
    return None if bad or not records else records


def check_records(records: list[dict], name: str, problems: Problems) -> None:
    first_id: dict[str, int] = {}
    first_image: dict[str, int] = {}
    for index, rec in enumerate(records):
        line = index + 1
        rid, image, text = rec.get("id"), rec.get("image"), rec.get("source_text")
        if not isinstance(rid, str) or not rid:
            problems.add(name, "record without a non-empty string id", f"line {line}")
        elif rid in first_id:
            problems.add(name, "duplicate ids", f"{rid} (lines {first_id[rid]} and {line})")
        else:
            first_id[rid] = line
        if not isinstance(image, str) or not valid_image_ref(image):
            problems.add(name, f"image is not images/{IMAGE_GLOB}", f"line {line}: {image!r}")
        elif image in first_image:
            problems.add(name, "duplicate images", f"{image} (lines {first_image[image]} and {line})")
        else:
            first_image[image] = line
        if not isinstance(text, str) or not text:
            problems.add(name, "record without a non-empty source_text", f"{rid} (line {line})")
        for key in ADDED_LABEL_KEYS:
            if key in rec:
                problems.add(name, f"record already has '{key}' (already a master dataset?)", f"{rid} (line {line})")


def llava_mismatch(entry: object, rec: dict, stage_name: str) -> Optional[str]:
    """Why a stage llava entry does not belong to the label at the same position (None = consistent)."""
    if not isinstance(entry, dict):
        return "entry is not an object"
    if set(entry) != LLAVA_KEYS:
        return f"keys {sorted(entry)} instead of {sorted(LLAVA_KEYS)}"
    if entry["id"] != rec["id"]:
        return f"id {entry['id']!r} != label id {rec['id']!r}"
    if entry["image"] != rec["image"]:
        return f"image {entry['image']!r} != label image {rec['image']!r}"
    if entry["stage"] != stage_name:
        return f"stage {entry['stage']!r} != {stage_name!r}"
    if entry["category"] != category_of(rec):
        return f"category {entry['category']!r} != source.stratum {category_of(rec)!r}"
    conv = entry["conversations"]
    if not (
        isinstance(conv, list)
        and len(conv) == 2
        and all(isinstance(turn, dict) and set(turn) == {"from", "value"} for turn in conv)
        and conv[0]["from"] == "human"
        and conv[1]["from"] == "gpt"
        and isinstance(conv[0]["value"], str)
        and isinstance(conv[1]["value"], str)
    ):
        return "conversations are not one human turn followed by one gpt turn"
    if not conv[0]["value"].startswith("<image>\n") or not conv[0]["value"][len("<image>\n") :].strip():
        return "human turn is not '<image>\\n<prompt>'"
    if conv[1]["value"] != rec["source_text"]:
        return "gpt value differs from source_text"
    return None


def check_llava(llava_path: str, records: list[dict], name: str, problems: Problems) -> Optional[list[dict]]:
    try:
        with open(llava_path, "r", encoding="utf-8") as fh:
            llava = json.load(fh)
    except (OSError, ValueError) as exc:
        problems.add(name, "llava.json unreadable or not valid JSON", f"{llava_path}: {exc}")
        return None
    if not isinstance(llava, list):
        problems.add(name, "llava.json is not a JSON array", llava_path)
        return None
    if len(llava) != len(records):  # name the ids instead of a cascade of positional mismatches
        label_ids = {rec["id"] for rec in records if isinstance(rec.get("id"), str)}
        llava_ids = {entry["id"] for entry in llava if isinstance(entry, dict) and isinstance(entry.get("id"), str)}
        for rid in sorted(label_ids - llava_ids):
            problems.add(name, "labels without a llava entry (ids)", rid)
        for rid in sorted(llava_ids - label_ids):
            problems.add(name, "llava entries without a label (ids)", rid)
        return llava
    for index, (entry, rec) in enumerate(zip(llava, records)):
        if not isinstance(rec.get("id"), str) or not isinstance(rec.get("image"), str):
            continue  # already reported by check_records
        if not isinstance(rec.get("source_text"), str):
            continue
        why = llava_mismatch(entry, rec, name)
        if why:
            problems.add(name, "llava entry does not match the label at the same position", f"#{index} (label id {rec['id']}): {why}")
    return llava


def check_images(
    records: list[dict], images_dir: str, name: str, problems: Problems, workers: int
) -> tuple[Optional[list[int]], int]:
    """Existence, bijection with the labels, non-empty + complete JPEG.

    Returns (byte sizes aligned with records, number of images/sample_*.jpg files on disk).
    """
    try:
        listed = {
            entry.name
            for entry in os.scandir(images_dir)
            if fnmatch.fnmatchcase(entry.name, IMAGE_GLOB) and entry.is_file()
        }
    except OSError as exc:
        problems.add(name, "images/ unreadable", f"{images_dir}: {exc}")
        return None, -1
    referenced: set[str] = set()
    todo: list[tuple[int, str]] = []
    for index, rec in enumerate(records):
        image = rec.get("image")
        if not isinstance(image, str) or not valid_image_ref(image):
            continue
        base = image.split("/")[1]
        referenced.add(base)
        if base in listed:
            todo.append((index, os.path.join(images_dir, base)))
        else:
            problems.add(name, "label image missing on disk", f"{rec.get('id')} -> {image}")
    orphans = sorted(listed - referenced)
    for base in orphans:
        problems.add(name, "image file without a label", f"images/{base}")
    sizes = [-1] * len(records)
    results = parallel_map(lambda item: probe_image(item[1]), todo, workers, f"{name}: checking images")
    for (index, _path), (size, why) in zip(todo, results):
        sizes[index] = size
        if why:
            rec = records[index]
            kind = "empty image file" if size == 0 else "unusable image file"
            problems.add(name, kind, f"{rec.get('id')} -> {rec['image']}: {why}")
    return sizes, len(listed)


def load_stage(name: str, final_root: str, problems: Problems, workers: int) -> Optional[Stage]:
    before = problems.count
    root = os.path.join(final_root, name)
    if not os.path.isdir(root):
        problems.add(name, "stage directory missing", root)
        return None
    labels_path = os.path.join(root, "labels", "labels.jsonl")
    llava_path = os.path.join(root, "llava.json")
    images_dir = os.path.join(root, "images")
    if not os.path.isfile(llava_path):
        problems.add(name, "llava.json missing (it is the stage completion marker)", llava_path)
    if not os.path.isfile(labels_path):
        problems.add(name, "labels/labels.jsonl missing", labels_path)
    if not os.path.isdir(images_dir):
        problems.add(name, "images/ directory missing", images_dir)
    if problems.count > before:
        return None

    records = read_labels(labels_path, name, problems)
    if records is None:
        return None
    check_records(records, name, problems)
    llava = check_llava(llava_path, records, name, problems)
    sizes, n_listed = check_images(records, images_dir, name, problems, workers)
    counts = {"labels": len(records), f"images/{IMAGE_GLOB}": n_listed}
    if llava is not None:
        counts["llava.json"] = len(llava)
    if n_listed >= 0 and len(set(counts.values())) > 1:
        problems.add(name, "sample counts differ", " ".join(f"{key}={value}" for key, value in counts.items()))
    if problems.count > before or llava is None or sizes is None:
        return None

    plan_path = os.path.join(root, "text_plan.json")
    return Stage(
        name=name,
        root=root,
        real_root=os.path.realpath(root),
        records=records,
        llava=llava,
        sizes=sizes,
        labels_md5=md5_file(labels_path),
        llava_md5=md5_file(llava_path),
        text_plan_md5=md5_file(plan_path) if os.path.isfile(plan_path) else None,
    )


def check_out_dir(out: str, problems: Problems) -> None:
    if os.path.lexists(out) and not os.path.isdir(out):
        problems.add("master", "output path exists and is not a directory", out)
    elif os.path.isdir(out):
        entries = os.listdir(out)
        if entries:
            state = "a finished master (manifest.json present)" if "manifest.json" in entries else "an unfinished earlier run (no manifest.json)"
            problems.add(
                "master",
                "output directory is not empty (refusing to merge into it)",
                f"{out}: looks like {state}; {len(entries)} entries, e.g. {sorted(entries)[:3]}",
            )


def check_out_overlap(out: str, stage_roots: list[str]) -> None:
    real_out = os.path.realpath(out)
    for root in stage_roots:
        real_root = os.path.realpath(root)
        if real_out == real_root or real_out.startswith(real_root + os.sep) or real_root.startswith(real_out + os.sep):
            raise UsageError(f"--out {out} overlaps the stage directory {root}")


# ---------------------------------------------------------------------------
# Planning and transfer
# ---------------------------------------------------------------------------
def build_plan(stages: list[Stage], shuffle_seed: Optional[int]) -> list[tuple[int, int]]:
    """Master position g -> (stage index, record index)."""
    plan = [(si, ri) for si, stage in enumerate(stages) for ri in range(len(stage.records))]
    if shuffle_seed is not None:

        def key(item: tuple[int, int]) -> bytes:
            si, ri = item
            tag = f"{shuffle_seed}|{stages[si].name}|{stages[si].records[ri]['id']}"
            return hashlib.sha256(tag.encode("utf-8")).digest()

        plan.sort(key=key)
    return plan


def master_id(g: int) -> str:
    return f"{g:08d}"


def master_image(g: int) -> str:
    return f"images/sample_{g:06d}.jpg"


class Transferer:
    """Places one image at its master path and checks it (byte size, or md5 with --verify_hash)."""

    def __init__(self, mode: str, verify_hash: bool):
        self.mode = mode
        self.verify_hash = verify_hash
        self._lock = threading.Lock()
        self._warned = False

    def _note_fallback(self, src: str, dst: str, exc: OSError) -> None:
        with self._lock:
            if self._warned:
                return
            self._warned = True
        log(
            f"{self.mode}: cannot hard-link ({exc}); falling back to copy "
            f"(first case: {src} -> {dst}; the remaining ones are counted in the manifest)"
        )

    def place(self, src: str, dst: str) -> str:
        if self.mode == "copy":
            shutil.copyfile(src, dst)
            return "copy"
        if self.mode == "symlink":
            os.symlink(src, dst)
            return "symlink"
        try:  # hardlink, and move (which removes the source afterwards)
            os.link(src, dst)
            return "hardlink"
        except FileExistsError:
            raise
        except OSError as exc:
            self._note_fallback(src, dst, exc)
            shutil.copyfile(src, dst)
            return f"hardlink_fallback_copy:{errno_name(exc)}"

    def __call__(self, job: tuple[str, str, int]) -> str:
        src, dst, size = job
        method = self.place(src, dst)
        try:
            got = os.stat(dst).st_size
        except OSError as exc:
            raise MigrateError(f"{dst} is not readable after {self.mode}: {exc}") from exc
        if got != size:
            raise MigrateError(f"size mismatch after {self.mode}: {src} has {size} bytes, {dst} has {got}")
        if self.verify_hash and md5_file(src) != md5_file(dst):
            raise MigrateError(f"md5 mismatch after {self.mode}: {src} -> {dst}")
        return method


class Created:
    """Everything this run may create under out, so a failed run can be undone exactly."""

    def __init__(self, out: str, total: int):
        self.out = out
        self.images_dir = os.path.join(out, "images")
        self.labels_dir = os.path.join(out, "labels")
        self.labels_path = os.path.join(self.labels_dir, "labels.jsonl")
        self.llava_path = os.path.join(out, "llava.json")
        self.manifest_path = os.path.join(out, "manifest.json")
        self.total = total
        self.new_dirs: list[str] = []

    def make_dirs(self) -> None:
        missing = []
        path = self.out
        while path and not os.path.lexists(path):
            missing.append(path)
            path = os.path.dirname(path)
        self.new_dirs = list(reversed(missing))
        os.makedirs(self.images_dir)
        os.makedirs(self.labels_dir)

    def rollback(self) -> str:
        pid = os.getpid()
        paths = [os.path.join(self.images_dir, os.path.basename(master_image(g))) for g in range(self.total)]
        for path in (self.labels_path, self.llava_path, self.manifest_path):
            paths += [path, f"{path}.tmp-{pid}"]
        removed, left = 0, []
        for path in paths:
            if os.path.lexists(path):
                try:
                    os.unlink(path)
                    removed += 1
                except OSError as exc:
                    left.append(f"{path}: {exc}")
        for directory in [self.labels_dir, self.images_dir] + list(reversed(self.new_dirs)):
            try:
                os.rmdir(directory)
            except OSError:
                pass
        message = f"rolled back: removed {removed} file(s) created by this run"
        if left:
            message += f"; COULD NOT remove {len(left)}: {left[:3]}"
        elif os.path.isdir(self.out) and os.listdir(self.out):
            message += f"; {self.out} still holds other entries (not created by this run)"
        return message


def write_atomic(path: str, write: Callable) -> None:
    """Write to a temp file, fsync, then os.replace (readers never see a partial file)."""
    tmp_path = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def dump_line(obj: dict) -> str:
    text = json.dumps(obj, ensure_ascii=False)
    for raw, escaped in LINE_SEPARATORS:
        text = text.replace(raw, escaped)
    return text


def master_label(stage: Stage, index: int, g: int) -> dict:
    src = stage.records[index]
    rec = dict(src)
    rec["id"] = master_id(g)
    rec["image"] = master_image(g)
    rec["stage"] = stage.name
    rec["orig_id"] = src["id"]
    rec["orig_image"] = src["image"]
    return rec


def master_llava_entry(stage: Stage, index: int, g: int) -> dict:
    entry = stage.llava[index]
    return {
        "id": master_id(g),
        "image": master_image(g),
        "stage": stage.name,
        "category": entry["category"],
        "orig_id": stage.records[index]["id"],
        "conversations": entry["conversations"],
    }


def write_master_files(created: Created, stages: list[Stage], plan: list[tuple[int, int]]) -> None:
    def write_labels(fh) -> None:
        for g, (si, ri) in enumerate(plan):
            fh.write(dump_line(master_label(stages[si], ri, g)) + "\n")

    def write_llava(fh) -> None:
        entries = [dump_line(master_llava_entry(stages[si], ri, g)) for g, (si, ri) in enumerate(plan)]
        fh.write("[\n" + ",\n".join(entries) + "\n]\n")

    write_atomic(created.labels_path, write_labels)
    write_atomic(created.llava_path, write_llava)


# ---------------------------------------------------------------------------
# Re-verification of what was written, and statistics for the manifest
# ---------------------------------------------------------------------------
def verify_master(created: Created, stages: list[Stage], plan: list[tuple[int, int]], workers: int) -> tuple[dict, dict]:
    """Re-read the written master and compare it with the sources; returns (statistics, checks)."""
    n = len(plan)
    bad: list[str] = []

    def flag(message: str) -> None:
        bad.append(message)

    try:
        with open(created.labels_path, "r", encoding="utf-8") as fh:
            labels = [json.loads(line) for line in fh]
        with open(created.llava_path, "r", encoding="utf-8") as fh:
            llava = json.load(fh)
    except (OSError, ValueError) as exc:
        raise MigrateError(f"cannot re-read the master files just written: {exc}") from exc

    try:
        names_on_disk = set(os.listdir(created.images_dir))
    except OSError as exc:
        raise MigrateError(f"cannot list {created.images_dir}: {exc}") from exc
    total_sources = sum(len(stage.records) for stage in stages)
    counts_ok = len(labels) == len(llava) == len(names_on_disk) == n == total_sources
    if not counts_ok:
        flag(f"counts differ: labels={len(labels)} llava={len(llava)} images={len(names_on_disk)} plan={n} sources={total_sources}")
    if len(llava) != len(labels):
        raise MigrateError("; ".join(bad))

    stats: dict = {
        "stage_counts": {stage.name: 0 for stage in stages},
        "category_counts": {},
        "stage_category_counts": {stage.name: {} for stage in stages},
        "length_histogram": {label: 0 for label, _lo, _hi in LENGTH_BUCKETS},
        "text_kind_counts": {key: 0 for key in TEXT_KIND_DEFINITIONS},
        "shuffle_kind_counts": {},
        "grade_counts": {},
    }
    ids_ok = labels_ok = llava_ok = True
    seen_sources: set[tuple[str, str]] = set()
    for g, (rec, entry, (si, ri)) in enumerate(zip(labels, llava, plan)):
        stage, src = stages[si], stages[si].records[ri]
        if (
            rec.get("id") != master_id(g)
            or rec.get("image") != master_image(g)
            or rec.get("stage") != stage.name
            or rec.get("orig_id") != src["id"]
            or rec.get("orig_image") != src["image"]
        ):
            ids_ok = False
            flag(f"label #{g}: id/image/stage/orig_* differ from the plan ({rec.get('id')!r}, {rec.get('stage')!r}, {rec.get('orig_id')!r})")
        rest_master = {k: v for k, v in rec.items() if k not in ("id", "image", *ADDED_LABEL_KEYS)}
        rest_source = {k: v for k, v in src.items() if k not in ("id", "image")}
        if rest_master != rest_source:
            labels_ok = False
            flag(f"label #{g}: content differs from {stage.name} record {src['id']}")
        seen_sources.add((rec.get("stage"), rec.get("orig_id")))

        turns = entry.get("conversations") if isinstance(entry, dict) else None
        if (
            not isinstance(entry, dict)
            or set(entry) != MASTER_LLAVA_KEYS
            or entry["id"] != rec.get("id")
            or entry["image"] != rec.get("image")
            or entry["stage"] != rec.get("stage")
            or entry["orig_id"] != rec.get("orig_id")
            or entry["category"] != category_of(rec)
            or turns != stage.llava[ri]["conversations"]
            or turns[1]["value"] != rec.get("source_text")
        ):
            llava_ok = False
            flag(f"llava entry #{g} does not match label #{g} / its stage entry")

        text = src["source_text"]
        category = category_of(src)
        stats["stage_counts"][stage.name] += 1
        stats["category_counts"][category] = stats["category_counts"].get(category, 0) + 1
        per_stage = stats["stage_category_counts"][stage.name]
        per_stage[category] = per_stage.get(category, 0) + 1
        stats["length_histogram"][length_bucket(len(text))] += 1
        kinds = stats["text_kind_counts"]
        kinds["digits_only"] += bool(DIGITS_RE.fullmatch(text))
        kinds["alnum_only"] += bool(ALNUM_RE.fullmatch(text))
        kinds["single_char"] += len(text) == 1
        kinds["multi_word"] += len(text.split()) >= 2
        kinds["multi_line"] += "\n" in text
        source = src.get("source")
        shuffle = source.get("shuffle") if isinstance(source, dict) else None
        shuffle = shuffle if isinstance(shuffle, str) else "unknown"
        stats["shuffle_kind_counts"][shuffle] = stats["shuffle_kind_counts"].get(shuffle, 0) + 1
        grade = str(src["braille_grade"]) if "braille_grade" in src else "unknown"
        stats["grade_counts"][grade] = stats["grade_counts"].get(grade, 0) + 1

    expected_sources = {(stage.name, rec["id"]) for stage in stages for rec in stage.records}
    permutation_ok = len(seen_sources) == n and seen_sources == expected_sources
    if not permutation_ok:
        flag("master samples are not a permutation of the stage samples (lost or duplicated records)")

    def check_one(item: tuple[int, int]) -> Optional[str]:
        g, expected = item
        path = os.path.join(created.images_dir, os.path.basename(master_image(g)))
        try:
            size = os.stat(path).st_size
        except OSError as exc:
            return f"image #{g} {path}: {exc}"
        return None if size == expected and size > 0 else f"image #{g} {path}: {size} bytes, expected {expected}"

    jobs = [(g, stages[si].sizes[ri]) for g, (si, ri) in enumerate(plan)]
    image_problems = [p for p in parallel_map(check_one, jobs, workers, "verifying master images") if p]
    images_ok = not image_problems
    bad.extend(image_problems[:5])

    names_ok = names_on_disk == {os.path.basename(master_image(g)) for g in range(n)}
    if not names_ok:
        flag(f"images/ does not hold exactly sample_000000..sample_{n - 1:06d}")

    total = n
    sums_ok = (
        sum(stats["stage_counts"].values()) == total
        and sum(stats["category_counts"].values()) == total
        and sum(stats["length_histogram"].values()) == total
        and sum(stats["shuffle_kind_counts"].values()) == total
        and sum(stats["grade_counts"].values()) == total
    )
    if not sums_ok:
        flag("statistics do not sum to the total")

    checks = {
        "stage_inputs_validated": True,
        "counts_reconcile": counts_ok,
        "master_ids_and_images_follow_plan": ids_ok and names_ok,
        "master_labels_equal_stage_records": labels_ok,
        "master_llava_matches_labels": llava_ok,
        "master_is_permutation_of_stage_samples": permutation_ok,
        "master_images_exist_with_source_size": images_ok,
        "statistics_sum_to_total": sums_ok,
    }
    if bad or not all(checks.values()):
        raise MigrateError("verification of the written master failed: " + "; ".join(bad[:5]))
    return stats, checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge stage datasets into one renumbered master dataset (see the module docstring)."
    )
    parser.add_argument("--final_root", default="final_data", help="directory holding the stage directories")
    parser.add_argument("--stages", nargs="+", default=list(DEFAULT_STAGES), help="stage directory names, in master order")
    parser.add_argument("--out", default=None, help="master directory to create (default: <final_root>/master)")
    parser.add_argument("--mode", choices=MODES, default="copy", help="how images reach the master")
    parser.add_argument("--shuffle_seed", type=int, default=None, help="seeded shuffle of the concatenation (default: stage order)")
    parser.add_argument("--verify_hash", action="store_true", help="compare md5 of source and destination instead of byte size")
    parser.add_argument("--dry_run", action="store_true", help="validate and print the plan; write nothing")
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="threads for image checks and transfer (1 = serial). Measured on WekaFS: cold reads are ~7x faster with 8 "
        "threads and ~11x with 16 (no gain at 32); on a hot local disk with tiny files 1 can be faster",
    )
    args = parser.parse_args(argv)
    if len(set(args.stages)) != len(args.stages):
        parser.error("--stages contains duplicates")
    for name in args.stages:
        if not name or "/" in name or name in (".", ".."):
            parser.error(f"invalid stage name {name!r}")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    return args


def describe_plan(args: argparse.Namespace, stages: list[Stage], plan: list[tuple[int, int]], out: str) -> None:
    order = "stage order" if args.shuffle_seed is None else f"shuffled (seed {args.shuffle_seed})"
    total_bytes = sum(stage.sizes[ri] for stage in stages for ri in range(len(stage.records)))
    log(f"plan: {len(plan)} samples ({', '.join(f'{s.name} {len(s.records)}' for s in stages)}), {human_bytes(total_bytes)} of images")
    log(f"      mode={args.mode} order={order} verify={'md5' if args.verify_hash else 'size'} out={out}")
    if args.shuffle_seed is None:
        start = 0
        for stage in stages:
            log(f"      {stage.name}: master ids {master_id(start)}..{master_id(start + len(stage.records) - 1)}")
            start += len(stage.records)
    for g, (si, ri) in enumerate(plan[:3]):
        stage = stages[si]
        log(f"      e.g. {stage.name}/{stage.records[ri]['id']} ({stage.records[ri]['image']}) -> {master_id(g)} ({master_image(g)})")


def run(args: argparse.Namespace) -> int:
    started = time.time()
    final_root = os.path.abspath(args.final_root)
    out = os.path.abspath(args.out) if args.out else os.path.join(final_root, "master")
    workers = min(args.workers, MAX_WORKERS)
    check_out_overlap(out, [os.path.join(final_root, name) for name in args.stages])

    problems = Problems()
    stages: list[Stage] = []
    if not os.path.isdir(final_root):
        problems.add("final_root", "directory missing", final_root)
    else:
        log(f"validating {len(args.stages)} stage(s) under {final_root} (workers={workers})")
        for name in args.stages:
            stage = load_stage(name, final_root, problems, workers)
            if stage is not None:
                stages.append(stage)
                log(f"  {name}: {len(stage.records)} labels = images = llava entries, ids unique, images complete")
    check_out_dir(out, problems)
    if problems:
        problems.report()
        print(f"[migrate] VALIDATION FAILED: {problems.count} problem(s); nothing was written", file=sys.stderr)
        return 1
    validate_s = time.time() - started

    plan = build_plan(stages, args.shuffle_seed)
    describe_plan(args, stages, plan, out)
    if args.dry_run:
        log(f"dry run: validation passed in {validate_s:.2f}s; nothing was written")
        return 0

    created = Created(out, len(plan))
    transferer = Transferer(args.mode, args.verify_hash)
    jobs = [
        (
            stages[si].image_path(ri, resolved=args.mode == "symlink"),
            os.path.join(created.images_dir, os.path.basename(master_image(g))),
            stages[si].sizes[ri],
        )
        for g, (si, ri) in enumerate(plan)
    ]
    try:
        created.make_dirs()
        t0 = time.time()
        methods = parallel_map(transferer, jobs, workers, f"images ({args.mode})")
        transfer_s = time.time() - t0
        t0 = time.time()
        write_master_files(created, stages, plan)
        write_s = time.time() - t0
        t0 = time.time()
        stats, checks = verify_master(created, stages, plan, workers)
        verify_s = time.time() - t0

        method_counts: dict[str, int] = {}
        fallback_errnos: dict[str, int] = {}
        for method in methods:
            base, _sep, reason = method.partition(":")
            method_counts[base] = method_counts.get(base, 0) + 1
            if reason:
                fallback_errnos[reason] = fallback_errnos.get(reason, 0) + 1
        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total": len(plan),
            "stages": [stage.name for stage in stages],
            "order": {
                "mode": "stage_order" if args.shuffle_seed is None else "shuffled",
                "shuffle_seed": args.shuffle_seed,
                "rule": "labels.jsonl line order within a stage"
                if args.shuffle_seed is None
                else "sort by sha256(f'{seed}|{stage}|{orig_id}')",
            },
            **stats,
            "text_kind_definitions": TEXT_KIND_DEFINITIONS,
            "input_labels_md5": {stage.name: stage.labels_md5 for stage in stages},
            "input_llava_md5": {stage.name: stage.llava_md5 for stage in stages},
            "input_text_plan_md5": {stage.name: stage.text_plan_md5 for stage in stages},
            "transfer": {
                "mode": args.mode,
                "files": len(jobs),
                "bytes": sum(job[2] for job in jobs),
                "methods": method_counts,
                "hardlink_fallback_errnos": fallback_errnos,
                "verify": "md5" if args.verify_hash else "size",
                "workers": workers,
            },
            "args": {
                "final_root": final_root,
                "stages": list(args.stages),
                "out": out,
                "mode": args.mode,
                "shuffle_seed": args.shuffle_seed,
                "verify_hash": args.verify_hash,
                "dry_run": args.dry_run,
                "workers": args.workers,
            },
            "checks": checks,
            "timing_s": {
                "validate": round(validate_s, 3),
                "transfer": round(transfer_s, 3),
                "write_labels_llava": round(write_s, 3),
                "verify": round(verify_s, 3),
            },
        }
        write_atomic(created.manifest_path, lambda fh: fh.write(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"))
    except (MigrateError, OSError) as exc:
        print(f"[migrate] FAILED: {exc}", file=sys.stderr)
        print(f"[migrate] {created.rollback()}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[migrate] interrupted", file=sys.stderr)
        print(f"[migrate] {created.rollback()}", file=sys.stderr)
        return 130
    except BaseException:
        print(f"[migrate] unexpected error; {created.rollback()}", file=sys.stderr)
        raise

    status = 0
    if args.mode == "move":
        try:
            removed, failed = remove_sources(jobs, workers)
        except KeyboardInterrupt:
            print("[migrate] interrupted while removing sources: the master is complete (manifest.json written); some source images remain", file=sys.stderr)
            return 130
        log(f"move: removed {removed} source image(s) after the master was verified")
        if failed:
            print(
                f"[migrate] ERROR: the master is complete (manifest.json written) but {len(failed)} source image(s) "
                f"could not be removed, e.g. {failed[:3]}",
                file=sys.stderr,
            )
            status = 1
    per_stage = ", ".join(f"{stage.name} {len(stage.records)}" for stage in stages)
    methods_text = ", ".join(f"{k}: {v}" for k, v in sorted(method_counts.items()))
    log(f"OK: {len(plan)} samples ({per_stage}) -> {out}")
    log(
        f"    mode={args.mode} ({methods_text}) verify={'md5' if args.verify_hash else 'size'} "
        f"order={manifest['order']['mode']} validate {validate_s:.2f}s transfer {transfer_s:.2f}s "
        f"write {write_s:.2f}s verify {verify_s:.2f}s total {time.time() - started:.2f}s"
    )
    print_summary(manifest)
    return status


def print_summary(manifest: dict) -> None:
    def fmt(counts: dict, skip_zero: bool = False) -> str:
        return ", ".join(f"{key or '(none)'} {value}" for key, value in counts.items() if value or not skip_zero)

    categories = sorted(manifest["category_counts"].items(), key=lambda kv: (-kv[1], kv[0]))
    top = ", ".join(f"{key or '(none)'} {value}" for key, value in categories[:6])
    more = f", ... +{len(categories) - 6} more" if len(categories) > 6 else ""
    checks = manifest["checks"]
    log("manifest summary:")
    log(f"  stages         {fmt(manifest['stage_counts'])}")
    log(f"  categories     {len(categories)}: {top}{more}")
    log(f"  length (chars) {fmt(manifest['length_histogram'], skip_zero=True)}")
    log(f"  text kinds     {fmt(manifest['text_kind_counts'])}")
    log(f"  shuffle kinds  {fmt(manifest['shuffle_kind_counts'])}")
    log(f"  grades         {fmt(manifest['grade_counts'])}")
    log(f"  checks         {sum(checks.values())}/{len(checks)} true")


def remove_sources(jobs: list[tuple[str, str, int]], workers: int) -> tuple[int, list[str]]:
    """move mode: drop each source image once its destination is confirmed again; never raises."""

    def remove(job: tuple[str, str, int]) -> Optional[str]:
        src, dst, size = job
        try:
            if os.stat(dst).st_size != size:
                return f"{dst}: destination size changed; source kept"
            os.unlink(src)
        except OSError as exc:
            return f"{src}: {exc}"
        return None

    results = parallel_map(remove, jobs, workers, "removing sources")
    failed = [r for r in results if r]
    return len(results) - len(failed), failed


def _on_sigterm(_signum, _frame) -> None:
    raise KeyboardInterrupt  # same clean rollback as Ctrl-C (batch schedulers send SIGTERM)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:  # not called from the main thread
        pass
    try:
        return run(args)
    except UsageError as exc:
        print(f"[migrate] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
