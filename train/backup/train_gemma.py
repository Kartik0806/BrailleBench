
import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMultimodalLM,
    AutoProcessor,
    TrainerCallback,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


INSTRUCTIONS = [
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
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
import random


class BrailleOCRDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        image_root: str = "data",
        split: str = "train",
        seed: int = 42,
    ):
        all_records = []
        with open(json_path, "r") as f:
            for line in f:
                all_records.append(json.loads(line))

        print(f"[Dataset] Loaded {len(all_records)} total records from {json_path}")

        self.records = all_records
        self.image_root = image_root + os.sep + split + os.sep + "images"
        self.instructions = INSTRUCTIONS

        print(f"[Dataset] split={split}  size={len(self.records)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        # Load image
        img_path = os.path.join(self.image_root, record["file"])
        image = Image.open(img_path).convert("RGB")

        # Select random instruction
        instruction = random.choice(self.instructions)

        # Get target text
        answer = record["text"]
        return {
            "image": image,
            "instruction": instruction,
            "answer": answer,
        }


# ---------------------------------------------------------------------------
# Collator — builds chat-formatted inputs and masks the prompt tokens
# ---------------------------------------------------------------------------
@dataclass
class ChandraCollator:
    processor: Any

    def __call__(self, features):
        images = [f["image"] for f in features]

        # Build the full chat (user + assistant) for each example
        full_texts = []
        prompt_texts = []
        for f in features:
            messages_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f["instruction"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": f["answer"]}],
                },
            ]
            messages_prompt = [messages_full[0]]

            full_text = self.processor.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=False
            )
            prompt_text = self.processor.apply_chat_template(
                messages_prompt, tokenize=False, add_generation_prompt=True
            )
            full_texts.append(full_text)
            prompt_texts.append(prompt_text)


        # Tokenize + process images
    
        nested_images = [[img] for img in images]

        batch = self.processor(
            text=full_texts,
            images=nested_images,
            return_tensors="pt",
            padding=True,
        )



        # print(self.processor.decode(batch["input_ids"][0]))
        # import code
        # code.interact(local=locals())

        # Build labels — mask everything that is part of the prompt (and padding)
        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Mask image tokens if present
        image_token_id = getattr(self.processor.tokenizer, "image_token_id", None)
        if image_token_id is None:
            image_token = getattr(self.processor.tokenizer, "image_token", None)
            if image_token is not None:
                image_token_id = self.processor.tokenizer.convert_tokens_to_ids(image_token)
        if image_token_id is not None and image_token_id >= 0:
            labels[labels == image_token_id] = -100

        # Mask prompt portion per example
        for i, prompt_text in enumerate(prompt_texts):
            prompt_ids = self.processor.tokenizer(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids[0]
            prompt_len = prompt_ids.shape[0]
            # With left-padding the prompt sits after the pad tokens, so we
            # find where real content starts and mask prompt_len tokens from there
            seq = input_ids[i]
            if pad_id is not None:
                non_pad = (seq != pad_id).nonzero(as_tuple=True)[0]
                start = non_pad[0].item() if len(non_pad) > 0 else 0
            else:
                start = 0
            labels[i, start : start + prompt_len] = -100

        batch["labels"] = labels

        # print(batch["input_ids"].shape)
        # print(batch["input_ids"][0][-10:])
        # print(batch["labels"][0][-10:])
        return batch


# ---------------------------------------------------------------------------
# Sample prediction callback during training
# ---------------------------------------------------------------------------
class SamplePredictionCallback(TrainerCallback):
    def __init__(
        self,
        processor: Any,
        dataset: Dataset,
        target: str,
        every_steps: int = 100,
        num_samples: int = 1,
        max_new_tokens: int = 128,
        seed: int = 42,
        output_dir: str = "./sample_predictions_gemma_merged",  # <-- new
    ):
        self.processor = processor
        self.dataset = dataset
        self.target = target
        self.every_steps = max(1, every_steps)
        self.num_samples = max(1, num_samples)
        self.max_new_tokens = max(1, max_new_tokens)
        self.rng = random.Random(seed)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.all_results = []  # accumulates across all steps

    def _generate_prediction(self, model, image: Image.Image, instruction: str, device: torch.device) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[[image]], return_tensors="pt", padding=True)
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, prompt_len:]
        return self.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0 or state.global_step % self.every_steps != 0:
            return control

        model = kwargs.get("model")
        if model is None or len(self.dataset) == 0:
            return control

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        print("\n" + "=" * 80)
        print(f"[Sample Eval] step={state.global_step}")
        print("=" * 80)

        step_results = []
        for i in range(self.num_samples):
            idx = self.rng.randrange(len(self.dataset))
            sample = self.dataset[idx]
            image = sample["image"]
            instruction = sample["instruction"]
            reference = sample["answer"]

            try:
                prediction = self._generate_prediction(model, image, instruction, device)
            except Exception as e:
                print(f"[Sample Eval] sample={i+1} failed: {e}")
                continue

            exact_match = prediction.strip() == reference.strip()
            print(f"Sample {i+1}/{self.num_samples}")
            print(f"Instruction: {instruction}")
            print(f"Reference  : {reference[:160]}")
            print(f"Prediction : {prediction[:160]}")
            print(f"Exact Match: {exact_match}")
            print("-" * 80)

            step_results.append({
                "step": state.global_step,
                "sample_idx": idx,
                "instruction": instruction,
                "reference": reference,
                "prediction": prediction,
                "exact_match": exact_match,
            })

        # Save per-step file
        if step_results:
            step_file = os.path.join(self.output_dir, f"step_{state.global_step:07d}.json")
            with open(step_file, "w", encoding="utf-8") as f:
                json.dump(step_results, f, ensure_ascii=False, indent=2)

            # Also maintain a running cumulative file
            self.all_results.extend(step_results)
            all_file = os.path.join(self.output_dir, "all_predictions.json")
            with open(all_file, "w", encoding="utf-8") as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)

            print(f"[Sample Eval] Saved {len(step_results)} predictions to {step_file}")

        if was_training:
            model.train()

        return control

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", required=True, help="Path to dataset JSON")
    parser.add_argument("--image_root", required=True, help="Root dir that image_path is relative to")
    parser.add_argument("--model_id", default="google/gemma-4-E2B-it")
    parser.add_argument("--output_dir", default="./gemma4-braille-ft")
    parser.add_argument("--target", choices=["braille", "original"], default="original")

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_lora", action="store_true", help="Fine-tune with LoRA adapters (recommended)")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)

    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--sample_eval_steps", type=int, default=100, help="Run sample prediction eval every N steps")
    parser.add_argument("--sample_eval_count", type=int, default=5, help="Number of random samples per sample-eval run")
    parser.add_argument("--sample_eval_max_new_tokens", type=int, default=128, help="Max tokens for sample prediction generation")
    parser.add_argument("--disable_sample_eval", action="store_true", help="Disable periodic sample prediction during training")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # -----------------------------------------------------------------------
    # Processor + model
    # -----------------------------------------------------------------------
    print(f"[Load] processor + model from {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    processor.tokenizer.padding_side = "right"  # right-pad for training

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )


    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # LoRA
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise RuntimeError("peft is not installed. Run: pip install peft")
        print("[LoRA] wrapping model with LoRA adapters")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # Common attention projection names — peft will match whichever exist
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        for p in model.base_model.model.model.vision_tower.parameters():
            p.requires_grad = True

        model.print_trainable_parameters()#

    # -----------------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------------
    train_ds = BrailleOCRDataset(
        json_path=args.data_json,
        image_root=args.image_root,
        split="train",
        target=args.target,
    )
    eval_ds = None
    if not args.no_eval:
        try:
            eval_ds = BrailleOCRDataset(
                json_path=args.data_json,
                image_root=args.image_root,
                split=args.eval_split,
                target=args.target,
            )
            if len(eval_ds) == 0:
                eval_ds = None
        except Exception as e:
            print(f"[Eval] disabled: {e}")
            eval_ds = None

    collator = ChandraCollator(processor=processor)

    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup

    # Parameter groups with different LRs
    param_groups = [
        {"params": model.parameters(), "lr": args.lr},       # e.g. 1e-4 for LoRA         # lower LR for vision tower
    ]

    optimizer = AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )

    # Compute total steps for scheduler
    total_steps = (
        len(train_ds) // (args.batch_size * args.grad_accum) * args.epochs
    )
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    from torch.utils.data import Subset
    if eval_ds is not None and len(eval_ds) > 100:
        eval_indices = random.sample(range(len(eval_ds)), 100)
        eval_ds = Subset(eval_ds, eval_indices)
        print(f"[Eval] Subsampled to {len(eval_ds)} samples")
    
    # if train_ds is not None and len(train_ds) > 100:
    #     train_indices = random.sample(range(len(train_ds)), 100)
    #     train_ds = Subset(train_ds, train_indices)
    #     print(f"[Train] Subsampled to {len(train_ds)} samples")

    # Pass optimizer + scheduler into Trainer
    # -----------------------------------------------------------------------
    # TrainingArguments
    # -----------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=500,
        save_strategy="steps",
        save_total_limit=2,
        eval_strategy="steps" if eval_ds is not None else "no",
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
        label_names=["labels"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,          # <-- was missing
        optimizers=(optimizer, scheduler),
    )


    if not args.disable_sample_eval:
        sample_eval_dataset = eval_ds if eval_ds is not None else train_ds
        trainer.add_callback(
            SamplePredictionCallback(
                processor=processor,
                dataset=sample_eval_dataset,
                target=args.target,
                every_steps=args.sample_eval_steps,
                num_samples=args.sample_eval_count,
                max_new_tokens=args.sample_eval_max_new_tokens,
                seed=args.seed,
            )
        )

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    trainer.train()

    # trainer.save_model(args.output_dir)
    # processor.save_pretrained(args.output_dir)
    # print(f"[Done] saved to {args.output_dir}")

    

    # -----------------------------------------------------------------------
    # Post-training evaluation (uses manifest JSON, NOT datasets)
    # -----------------------------------------------------------------------
    print("\n[Eval] Running post-training evaluation from manifest JSON...")
    model.eval()  # set to eval mode just in case
    import json
    from collections import defaultdict
    from PIL import Image
    from tqdm import tqdm

    def levenshtein(s1, s2):
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

    def compute_cer(pred, ref):
        if len(ref) == 0:
            return 0.0 if len(pred) == 0 else 1.0
        return levenshtein(pred, ref) / len(ref)

    def compute_wer(pred, ref):
        pw, rw = pred.split(), ref.split()
        if len(rw) == 0:
            return 0.0 if len(pw) == 0 else 1.0
        return levenshtein(pw, rw) / len(rw)

    def generate_prediction(model, processor, image, instruction, max_new_tokens):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ],
        }]

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(text=[prompt], images=[[image]], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )

        prompt_len = inputs["input_ids"].shape[1]
        pred = processor.tokenizer.decode(
            output_ids[0, prompt_len:], skip_special_tokens=True
        ).strip()

        return pred

    def evaluate_split(records, split_name):
        records = records[:100]  # limit

        results = []
        total_cer, total_wer, exact = 0.0, 0.0, 0
        lang_metrics = defaultdict(lambda: {"cer": 0.0, "wer": 0.0, "exact": 0, "count": 0})

        instruction = "What is written in the image?"

        instruction = random.choice(INSTRUCTIONS_ORIGINAL)

        for rec in tqdm(records, desc=f"Evaluating ({split_name})"):
            image_path = os.path.join(args.image_root, rec["image_path"])
            if not os.path.exists(image_path):
                continue

            image = Image.open(image_path).convert("RGB")
            reference = rec["braille_unicode"] if args.target == "braille" else rec["original_text"]

            pred = generate_prediction(model, processor, image, instruction, 128)

            cer = compute_cer(pred, reference)
            wer = compute_wer(pred, reference)
            is_exact = pred.strip() == reference.strip()

            total_cer += cer
            total_wer += wer
            exact += int(is_exact)

            lang = rec.get("language", "unknown")
            lang_metrics[lang]["cer"] += cer
            lang_metrics[lang]["wer"] += wer
            lang_metrics[lang]["exact"] += int(is_exact)
            lang_metrics[lang]["count"] += 1

            results.append({
                "id": rec.get("id", ""),
                "language": lang,
                "reference": reference,
                "prediction": pred,
                "cer": round(cer, 4),
                "wer": round(wer, 4),
                "exact_match": is_exact,
            })

        n = len(results)

        return {
            "num_samples": n,
            "cer": total_cer / n if n else 0,
            "wer": total_wer / n if n else 0,
            "accuracy": exact / n if n else 0,
            "per_language": {
                lang: {
                    "count": m["count"],
                    "cer": round(m["cer"] / m["count"], 4),
                    "wer": round(m["wer"] / m["count"], 4),
                    "accuracy": round(m["exact"] / m["count"], 4),
                }
                for lang, m in lang_metrics.items()
            },
            "predictions": results,
        }

    # -----------------------------------------------------------------------
    # Load manifest JSON
    # -----------------------------------------------------------------------
    with open(args.data_json, "r", encoding="utf-8") as f:
        all_records = json.load(f)

    train_records = [r for r in all_records if r.get("split") == "train"]
    test_records  = [r for r in all_records if r.get("split") == "test"]

    print(f"[Eval] train samples: {len(train_records)}")
    print(f"[Eval] test samples : {len(test_records)}")

    # -----------------------------------------------------------------------
    # Run evaluation
    # -----------------------------------------------------------------------
    train_output = evaluate_split(train_records, "train")
    test_output  = evaluate_split(test_records, "test")

    print(f"\n[Train] CER={train_output['cer']:.4f} WER={train_output['wer']:.4f} Acc={train_output['accuracy']:.4f}")
    print(f"[Test ] CER={test_output['cer']:.4f} WER={test_output['wer']:.4f} Acc={test_output['accuracy']:.4f}")

    # -----------------------------------------------------------------------
    # Save JSON
    # -----------------------------------------------------------------------
    output = {
        "config": {
            "model": args.model_id,
            "target": args.target,
            "max_samples": 100,
        },
        "train": train_output,
        "test": test_output,
    }

    output_path = os.path.join("gemma4_preds_in_memory", "unmerged_preds.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] predictions → {output_path}")

    os.makedirs("merged-gemma-full", exist_ok=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained("./merged-gemma-full")
    processor.save_pretrained("./merged-gemma-full")

if __name__ == "__main__":
    main()
