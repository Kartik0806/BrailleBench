"""
Evaluate a fine-tuned Chandra OCR-2 model on a Braille/Marathi OCR dataset.

Computes:
  - Character Error Rate (CER)
  - Word Error Rate (WER)
  - Exact-match accuracy
  - Per-language breakdown

Usage:
    python train/evaluate.py \
        --model_dir ./chandra-braille-ft \
        --data_json dataset_clean/manifest_clean.json \
        --image_root dataset_clean \
        --target braille \
        --split test
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compute_cer(prediction: str, reference: str) -> float:
    if len(reference) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    return levenshtein(prediction, reference) / len(reference)


def compute_wer(prediction: str, reference: str) -> float:
    ref_words = reference.split()
    pred_words = prediction.split()
    if len(ref_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return levenshtein(pred_words, ref_words) / len(ref_words)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
INSTRUCTIONS_BRAILLE = [
    "What is written in the image?",
]

INSTRUCTIONS_ORIGINAL = [
    "What is written in the image?",
]


def generate_prediction(model, processor, image: Image.Image, instruction: str, device: str, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        },
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the generated tokens (strip the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    generated = output_ids[0, prompt_len:]
    text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Chandra OCR model")
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--base_model_id", default="datalab-to/chandra-ocr-2", help="Base model ID (for LoRA)")
    parser.add_argument("--data_json", required=True, help="Path to dataset JSON")
    parser.add_argument("--image_root", required=True, help="Root dir for image paths")
    parser.add_argument("--target", choices=["braille", "original"], default="braille")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate on")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (for quick testing)")
    parser.add_argument("--output_file", default=None, help="Save detailed results to JSON")
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if args.bf16 and device != "cpu" else torch.float32

    # -------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------
    print(f"[Load] processor from {args.model_dir}")
    processor = AutoProcessor.from_pretrained(args.model_dir)

    # Check if this is a LoRA checkpoint (has adapter_config.json)
    is_lora = os.path.exists(os.path.join(args.model_dir, "adapter_config.json"))
    if is_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed. Run: pip install peft")
        print(f"[Load] base model from {args.base_model_id}")
        base_model = AutoModelForImageTextToText.from_pretrained(
            args.base_model_id, dtype=dtype, device_map="auto",
        )
        print(f"[Load] LoRA adapters from {args.model_dir}")
        model = PeftModel.from_pretrained(base_model, args.model_dir)
        model = model.merge_and_unload()
    else:
        print(f"[Load] full model from {args.model_dir}")
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_dir, dtype=dtype, device_map="auto",
        )

    model.eval()

    # -------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------
    with open(args.data_json, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    records = [r for r in all_records if r.get("split", "test") == args.split]
    if not records:
        print(f"[Warn] No records found for split='{args.split}', using all records")
        records = all_records

    if args.max_samples is not None:
        records = records[:args.max_samples]

    print(f"[Eval] split={args.split}  target={args.target}  samples={len(records)}")

    instruction = INSTRUCTIONS_BRAILLE[0] if args.target == "braille" else INSTRUCTIONS_ORIGINAL[0]

    # -------------------------------------------------------------------
    # Run evaluation
    # -------------------------------------------------------------------
    results = []
    overall_cer = 0.0
    overall_wer = 0.0
    exact_matches = 0

    lang_metrics = defaultdict(lambda: {"cer": 0.0, "wer": 0.0, "exact": 0, "count": 0})

    for rec in tqdm(records, desc="Evaluating"):
        image_path = os.path.join(args.image_root, rec["image_path"])
        if not os.path.exists(image_path):
            print(f"[Skip] Image not found: {image_path}")
            continue

        image = Image.open(image_path).convert("RGB")
        reference = rec["braille_unicode"] if args.target == "braille" else rec["original_text"]

        prediction = generate_prediction(model, processor, image, instruction, device, args.max_new_tokens)
        prediction = prediction.split("\nassistant")[0]
        cer = compute_cer(prediction, reference)
        wer = compute_wer(prediction, reference)
        exact = prediction.strip() == reference.strip()

        overall_cer += cer
        overall_wer += wer
        if exact:
            exact_matches += 1

        lang = rec.get("language", "unknown")
        lang_metrics[lang]["cer"] += cer
        lang_metrics[lang]["wer"] += wer
        lang_metrics[lang]["exact"] += int(exact)
        lang_metrics[lang]["count"] += 1

        results.append({
            "id": rec.get("id", ""),
            "language": lang,
            "reference": reference,
            "prediction": prediction,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "exact_match": exact,
        })

    n = len(results)
    if n == 0:
        print("[Error] No samples evaluated.")
        return

    # -------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------
    avg_cer = overall_cer / n
    avg_wer = overall_wer / n
    accuracy = exact_matches / n

    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    print(f"  Samples evaluated : {n}")
    print(f"  CER               : {avg_cer:.4f}")
    print(f"  WER               : {avg_wer:.4f}")
    print(f"  Exact Match Acc.  : {accuracy:.4f} ({exact_matches}/{n})")

    print("\n" + "-" * 60)
    print("PER-LANGUAGE BREAKDOWN")
    print("-" * 60)
    print(f"  {'Language':<12} {'Count':>6} {'CER':>8} {'WER':>8} {'Exact%':>8}")
    print(f"  {'--------':<12} {'-----':>6} {'---':>8} {'---':>8} {'------':>8}")
    for lang in sorted(lang_metrics.keys()):
        m = lang_metrics[lang]
        c = m["count"]
        print(f"  {lang:<12} {c:>6} {m['cer']/c:>8.4f} {m['wer']/c:>8.4f} {m['exact']/c*100:>7.1f}%")

    # -------------------------------------------------------------------
    # Show a few example predictions
    # -------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("SAMPLE PREDICTIONS (first 5)")
    print("-" * 60)
    for r in results[:5]:
        print(f"\n  ID  : {r['id']}")
        print(f"  Lang: {r['language']}")
        print(f"  Ref : {r['reference'][:80]}")
        print(f"  Pred: {r['prediction'][:80]}")
        print(f"  CER : {r['cer']}  WER: {r['wer']}  Exact: {r['exact_match']}")

    # -------------------------------------------------------------------
    # Save detailed results
    # -------------------------------------------------------------------
    if args.output_file:
        output = {
            "config": {
                "model_dir": args.model_dir,
                "split": args.split,
                "target": args.target,
                "num_samples": n,
            },
            "overall": {
                "cer": round(avg_cer, 4),
                "wer": round(avg_wer, 4),
                "exact_match_accuracy": round(accuracy, 4),
            },
            "per_language": {
                lang: {
                    "count": m["count"],
                    "cer": round(m["cer"] / m["count"], 4),
                    "wer": round(m["wer"] / m["count"], 4),
                    "exact_match_accuracy": round(m["exact"] / m["count"], 4),
                }
                for lang, m in sorted(lang_metrics.items())
            },
            "predictions": results,
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[Saved] Detailed results to {args.output_file}")


if __name__ == "__main__":
    main()
