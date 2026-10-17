"""Braille OCR dataset and train/eval split construction."""

import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset, Subset

# Instruction prompts, randomly sampled per example
INSTRUCTIONS = [
    "What Braille text is encoded in this image?",
    "Transcribe the Braille characters shown in this image.",
    "Decode the Braille dots in this image and output the text.",
    "Read the Braille content in this image and provide the translated text.",
    "Extract the Braille text from this image and convert it to readable text.",
    "What does this Braille image say?",
    "Identify the Braille cells and translate them into plain text.",
    "Perform Braille OCR on this image and return the decoded text.",
    "Read the raised-dot Braille pattern in this image and transcribe it.",
    "Convert the Braille writing in this image into readable text.",
]

SAMPLE_EVAL_MAX_SIZE = 100


class BrailleOCRDataset(Dataset):
    def __init__(self, image_dir: str, label_path: str, split: str = "train", seed: int = 42):
        if split == "val":
            val_image_dir = image_dir.replace("train", "val")
            val_label_path = label_path.replace("train", "val")
            if val_image_dir == image_dir and val_label_path == label_path:
                raise ValueError(
                    "Cannot derive a validation split: neither image_dir "
                    f"({image_dir!r}) nor label_path ({label_path!r}) contains "
                    "'train', so the 'train' -> 'val' substitution would load the "
                    "TRAINING files as the eval set. Use a train/ and val/ "
                    "directory layout, or pass --no_eval."
                )
            image_dir, label_path = val_image_dir, val_label_path

        label_path = os.path.abspath(label_path)

        print(f"[Dataset] Loading {split} split from {label_path}")
        try:
            all_records = []
            with open(label_path, "r") as f:
                for line in f:
                    all_records.append(json.loads(line))
        except Exception:
            all_records = json.load(open(label_path, "r"))

        print(f"[Dataset] Loaded {len(all_records)} total records from {label_path}")

        self.records = all_records
        self.image_dir = image_dir
        self.instructions = INSTRUCTIONS

        print(f"[Dataset] split={split}  size={len(self.records)}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]

        img_path = (
            os.path.join(self.image_dir, record["image"])
            if "id" in record
            else os.path.join(self.image_dir, record["file"])
        )
        image = Image.open(img_path).convert("RGB")

        if "conversations" in record:
            # LLaVA-style record: first human turn (minus "<image>") is the
            # instruction, first gpt turn is the answer.
            turns = record["conversations"]
            human = next((t["value"] for t in turns if t["from"] == "human"), None)
            gpt = next((t["value"] for t in turns if t["from"] == "gpt"), None)
            if human is None or gpt is None:
                raise ValueError(
                    f"LLaVA record {record.get('id', idx)!r} needs a 'human' and a "
                    f"'gpt' turn in 'conversations', got "
                    f"{[t.get('from') for t in turns]}"
                )
            instruction = human.replace("<image>", "").strip()
            return {"image": image, "instruction": instruction, "answer": gpt}

        instruction = random.choice(self.instructions)
        answer = (
            record["reasoning_text"]
            if "reasoning_text" in record
            else record["source_text"]
        )
        return {"image": image, "instruction": instruction, "answer": answer}


def build_datasets(image_dir: str, label_path: str, eval_split: str, no_eval: bool):
    """Returns (train_ds, eval_ds, sample_eval_ds). eval_ds is None when evaluation is off or
    unavailable; sample_eval_ds is eval_ds capped to SAMPLE_EVAL_MAX_SIZE random samples."""
    train_ds = BrailleOCRDataset(image_dir=image_dir, split="train", label_path=label_path)
    eval_ds = None

    if not no_eval:
        try:
            eval_ds = BrailleOCRDataset(image_dir=image_dir, split=eval_split, label_path=label_path)
            if len(eval_ds) == 0:
                eval_ds = None
        except Exception as e:
            print(f"[Eval] disabled: {e}")
            eval_ds = None

    sample_eval_ds = eval_ds
    if eval_ds is not None and len(eval_ds) > SAMPLE_EVAL_MAX_SIZE:
        eval_indices = random.sample(range(len(eval_ds)), SAMPLE_EVAL_MAX_SIZE)
        sample_eval_ds = Subset(eval_ds, eval_indices)
        print(f"[Eval] Subsampled to {len(sample_eval_ds)} samples")

    return train_ds, eval_ds, sample_eval_ds
