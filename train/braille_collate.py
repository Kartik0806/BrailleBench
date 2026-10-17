import copy
from typing import Any

import torch
from PIL import Image
from transformers import MllamaProcessor
from transformers.feature_extraction_utils import BatchFeature

# Adapted from visual-assistant-eval/tasks/obr/collate.py for local Braille jsonl data.


def check_header(targets: list[list[int]], seq: torch.Tensor) -> bool:
    for i in range(len(seq) - 3):
        if seq[i : i + 3] in targets:
            return True
    return False


def replace_target(target: list[int], seq: list[int]) -> list[int]:
    for i in range(len(seq) - 3):
        if seq[i : i + 3] == target:
            seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
    return seq


def tokenize_dialogs(
    dialogs: list[dict[str, Any]],
    images: list[Image.Image],
    processor: MllamaProcessor,
) -> BatchFeature:
    text_prompt = processor.apply_chat_template(dialogs)
    text_prompt = [prompt.replace("<|begin_of_text|>", "") for prompt in text_prompt]
    # Mllama expects one sub-list of images per sample
    batch = processor(images=[[image] for image in images], text=text_prompt, padding=True, return_tensors="pt")
    label_list = []
    for i in range(len(batch["input_ids"])):
        dialog_tokens = batch["input_ids"][i].tolist()
        labels = copy.copy(dialog_tokens)
        eot_indices = [i for i, n in enumerate(labels) if n == 128009]
        last_idx = 0
        # system prompt header "<|start_header_id|>system<|end_header_id|>" -> [128006, 9125, 128007]
        # user prompt header "<|start_header_id|>user<|end_header_id|>" -> [128006, 882, 128007]
        prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
        for n, idx in enumerate(eot_indices):
            current_seq = labels[last_idx : idx + 1]
            if check_header(prompt_header_seqs, current_seq):
                # found prompt header, indicating that this seq should be masked
                labels[last_idx : idx + 1] = [-100] * (idx - last_idx + 1)
            else:
                last_idx = idx + 1
        # Mask the assistant header "<|start_header_id|>assistant<|end_header_id|>" -> [128006, 78191, 128007]
        assistant_header_seq = [128006, 78191, 128007]
        labels = replace_target(assistant_header_seq, labels)
        # Mask the padding token and image token 128256
        for i in range(len(labels)):
            if labels[i] == processor.tokenizer.pad_token_id or labels[i] == 128256:  # type: ignore
                labels[i] = -100
        label_list.append(labels)
    batch["labels"] = torch.tensor(label_list)
    return batch


class BrailleCollator:
    """Same chat-template + masking scheme as obr's Collator, for Braille OCR."""

    def __init__(
        self,
        processor: MllamaProcessor,
        instruction: str = "What Braille text is encoded in this image?",
        is_training: bool = True,
    ) -> None:
        self.processor = processor
        self.instruction = instruction
        self.is_training = is_training

    def __call__(self, examples: list[dict[str, Any]]) -> BatchFeature:
        images = [example["image"] for example in examples]
        dialogs = []
        for example in examples:
            dialog = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.instruction},
                    ],
                },
            ]
            if self.is_training:
                dialog.append(
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": example["answer"]}],
                    },
                )
            dialogs.append(dialog)
        return tokenize_dialogs(dialogs, images, self.processor)
