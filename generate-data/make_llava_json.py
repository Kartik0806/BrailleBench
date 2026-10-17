#!/usr/bin/env python3
"""Build llava.json (LLaVA-style conversations) from a stage's labels.jsonl.

The output is one JSON array with one entry per line. The prompt of a sample is picked
from the stage's prompt bank by hashing (seed, stage, label id), so it does not depend on
record order; the answer is the label's ``source_text`` verbatim.

    python src/make_llava_json.py --labels final_data/pretrain/labels/labels.jsonl \
        --out final_data/pretrain/llava.json --stage pretrain

Exit codes: 0 ok, 1 bad input (nothing is written), 2 usage error.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Optional

STAGES = ("pretrain", "midtrain", "posttrain")
DEFAULT_SEED = 1337

# Byte-identical to INSTRUCTIONS in train/train.py (same order).
PRETRAIN_PROMPTS = (
    "What is written in the image?",
    "Transcribe the text shown in this image.",
    "Read the image and output the text.",
    "Extract all text from this image.",
    "Please OCR this image.",
    "What text appears in the image?",
    "Read out the contents of this image.",
    "Perform OCR on the image and return the text.",
    "Give me the text content of this image.",
    "What does this image say?",
)

# Braille-aware paraphrases. The label is always the plain transcription, so none of these
# may ask for another output form, and they must fit digits, codes and scrambled text too.
MIDTRAIN_EXTRA_PROMPTS = (
    "Read the braille in this image and write out the text.",
    "What does the braille in this image say?",
    "Transcribe the braille shown in this image into plain text.",
    "Decode the braille text in the image.",
    "Read the braille dots in this picture and give me the text.",
    "Convert the braille in this image to ordinary print text.",
    "Please read this braille and type out what it says.",
    "What is written in braille in this image?",
    "Output the text that this braille represents.",
    "Read the braille and write the text exactly as it appears, without correcting anything.",
    "Read the braille shown here and return only the text.",
    "Turn the braille in this image into text.",
    "Give me a text transcription of the braille in the image.",
    "What text is encoded in the braille in this image?",
)

POSTTRAIN_EXTRA_PROMPTS = (
    "Extract all of the text from this braille document.",
    "Return only the transcription, no commentary.",
    "Transcribe the full contents of this braille page.",
    "Read this entire braille page and write out everything it says.",
    "Output the complete text of the braille document shown in the image.",
    "Transcribe everything written in braille on this page. Do not add anything else.",
    "This image shows a braille document. Transcribe all of it.",
    "Read all of the braille in the image, from start to finish, and give me the complete text.",
    "Please transcribe the whole braille page as plain text, with no extra remarks.",
    "Write out the full text of this braille page exactly as written and nothing else.",
    "Give me the entire text of this braille image with no additional remarks.",
    "OCR this braille document and output only the recognized text.",
)

BANKS = {
    "pretrain": PRETRAIN_PROMPTS,
    "midtrain": PRETRAIN_PROMPTS + MIDTRAIN_EXTRA_PROMPTS,
    "posttrain": PRETRAIN_PROMPTS + MIDTRAIN_EXTRA_PROMPTS + POSTTRAIN_EXTRA_PROMPTS,
}

# Output forms the label cannot deliver (it is plain text, not JSON/boxes/lines/another language).
FORBIDDEN_PROMPT_RE = re.compile(
    r"\b(json|markdown|html|xml|csv|yaml|tables?|bullets?|bbox(es)?|bounding|box(es)?|coordinates?)\b"
    r"|line[\s-]*by[\s-]*line|\beach line\b|\bper line\b|\bstep[\s-]*by[\s-]*step\b"
    r"|\btranslat\w*|\blanguages?\b|\bcaption\w*|\bdescri(be|ption)\w*|\bsummar\w*|\bexplain\w*",
    re.IGNORECASE,
)

# Written verbatim: the gpt answer must equal source_text, so JSON-escape only what JSON requires,
# plus the three separators that str.splitlines() would treat as line breaks.
_LINE_SEPARATORS = tuple((chr(code), f"\\u{code:04x}") for code in (0x2028, 0x2029, 0x85))


def validate_banks() -> None:
    """Raise RuntimeError if a prompt bank breaks the contract (sizes, nesting, wording)."""
    pre, mid, post = (BANKS[s] for s in STAGES)
    if len(pre) != 10:
        raise RuntimeError(f"pretrain bank must have exactly 10 prompts, has {len(pre)}")
    if len(mid) < 20 or mid[: len(pre)] != pre:
        raise RuntimeError("midtrain bank must be the pretrain bank plus at least 10 more prompts")
    if len(post) < 30 or post[: len(mid)] != mid or len(post) < len(mid) + 10:
        raise RuntimeError("posttrain bank must be the midtrain bank plus at least 10 more prompts")
    for stage, bank in BANKS.items():
        if len(set(bank)) != len(bank):
            raise RuntimeError(f"{stage} bank has duplicate prompts")
        for prompt in bank:
            if not prompt or prompt != prompt.strip() or "\n" in prompt:
                raise RuntimeError(f"{stage} prompt is empty, not stripped or multi-line: {prompt!r}")
            if "<image>" in prompt:
                raise RuntimeError(f"{stage} prompt contains <image>: {prompt!r}")
            hit = FORBIDDEN_PROMPT_RE.search(prompt)
            if hit:
                raise RuntimeError(f"{stage} prompt promises {hit.group(0)!r}: {prompt!r}")


def select_prompt(stage: str, label_id: str, seed: int = DEFAULT_SEED) -> str:
    bank = BANKS[stage]
    digest = hashlib.sha256(f"{seed}|{stage}|{label_id}".encode("utf-8")).hexdigest()
    return bank[int(digest, 16) % len(bank)]


def build_entry(record: dict, stage: str, seed: int = DEFAULT_SEED) -> dict:
    source = record.get("source")
    stratum = source.get("stratum") if isinstance(source, dict) else None
    return {
        "id": record["id"],
        "image": record["image"],
        "stage": stage,
        "category": stratum if isinstance(stratum, str) else "",
        "conversations": [
            {"from": "human", "value": f"<image>\n{select_prompt(stage, record['id'], seed)}"},
            {"from": "gpt", "value": record["source_text"]},
        ],
    }


def read_labels(path: str) -> tuple[list[dict], list[str]]:
    """Parse labels.jsonl strictly; returns (records, error messages)."""
    records: list[dict] = []
    errors: list[str] = []
    first_line: dict[str, int] = {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    errors.append(f"line {line_no}: blank line")
                    continue
                try:
                    rec = json.loads(line)
                except ValueError as exc:
                    errors.append(f"line {line_no}: invalid JSON ({exc})")
                    continue
                if not isinstance(rec, dict):
                    errors.append(f"line {line_no}: record is not a JSON object")
                    continue
                rid = rec.get("id")
                if not isinstance(rid, str) or not rid:
                    errors.append(f"line {line_no}: missing or non-string 'id'")
                    continue
                if not isinstance(rec.get("image"), str) or not rec["image"]:
                    errors.append(f"line {line_no} (id {rid}): missing or non-string 'image'")
                    continue
                if not isinstance(rec.get("source_text"), str) or not rec["source_text"]:
                    errors.append(f"line {line_no} (id {rid}): missing or empty 'source_text'")
                    continue
                if rid in first_line:
                    errors.append(f"line {line_no}: duplicate id {rid} (first seen on line {first_line[rid]})")
                    continue
                first_line[rid] = line_no
                records.append(rec)
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"cannot read {path}: {exc}")
    return records, errors


def dump_line(obj: dict) -> str:
    text = json.dumps(obj, ensure_ascii=False)
    for raw, escaped in _LINE_SEPARATORS:
        text = text.replace(raw, escaped)
    return text


def write_llava_array(entries: list[dict], out_path: str) -> None:
    """One JSON array, one entry per line; temp file + os.replace so readers never see a partial file."""
    body = ",\n".join(dump_line(entry) for entry in entries)
    tmp_path = f"{out_path}.tmp-{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(f"[\n{body}\n]\n" if entries else "[]\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build llava.json from a stage's labels.jsonl.")
    parser.add_argument("--labels", required=True, help="stage labels.jsonl")
    parser.add_argument("--out", required=True, help="llava.json to write (replaced atomically)")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="prompt-selection seed")
    args = parser.parse_args(argv)

    validate_banks()
    if not os.path.isfile(args.labels):
        print(f"[make_llava_json] ERROR: labels file not found: {args.labels}", file=sys.stderr)
        return 1
    if not os.path.isdir(os.path.dirname(os.path.abspath(args.out))):
        print(f"[make_llava_json] ERROR: output directory does not exist: {os.path.dirname(args.out)}", file=sys.stderr)
        return 1
    if os.path.realpath(args.out) == os.path.realpath(args.labels):
        print("[make_llava_json] ERROR: --out must not be the labels file", file=sys.stderr)
        return 1

    records, errors = read_labels(args.labels)
    if not records and not errors:
        errors.append("labels file has no records")
    if errors:
        print(f"[make_llava_json] ERROR: {len(errors)} problem(s) in {args.labels}; nothing written", file=sys.stderr)
        for message in errors[:10]:
            print(f"[make_llava_json]   {message}", file=sys.stderr)
        if len(errors) > 10:
            print(f"[make_llava_json]   ... and {len(errors) - 10} more", file=sys.stderr)
        return 1

    entries = [build_entry(rec, args.stage, args.seed) for rec in records]
    write_llava_array(entries, args.out)
    used = len({entry["conversations"][0]["value"] for entry in entries})
    print(
        f"[make_llava_json] stage={args.stage} records={len(entries)} "
        f"prompts_used={used}/{len(BANKS[args.stage])} seed={args.seed} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
