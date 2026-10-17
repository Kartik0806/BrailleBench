"""Training-time callback that generates sample predictions and saves them to disk."""

import json
import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import TrainerCallback

from model_handler import ModelHandler


class SamplePredictionCallback(TrainerCallback):
    def __init__(
        self,
        handler: ModelHandler,
        dataset: Dataset,
        every_steps: int = 100,
        num_samples: int = 1,
        max_new_tokens: int = 128,
        seed: int = 42,
        output_dir: str = "./sample_predictions",
    ):
        self.handler = handler
        self.dataset = dataset
        self.every_steps = max(1, every_steps)
        self.num_samples = max(1, num_samples)
        self.max_new_tokens = max(1, max_new_tokens)
        self.rng = random.Random(seed)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.all_results = []

    def _generate_prediction(
        self, model, image: Image.Image, instruction: str, device: torch.device
    ) -> str:
        prompt = self.handler.prompt_text(instruction)
        inputs = self.handler.encode([prompt], [image])
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # generate() needs kv-cache on regardless of the train-time
        # use_cache=False setting (which only exists for grad checkpointing).
        prev_use_cache = model.config.use_cache
        model.config.use_cache = True
        try:
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=model.generation_config.pad_token_id,
                )
        finally:
            model.config.use_cache = prev_use_cache

        prompt_len = inputs["input_ids"].shape[1]
        generated = output_ids[0, prompt_len:]
        return self.handler.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0 or state.global_step % self.every_steps != 0:
            return control

        model = kwargs.get("model")
        if model is None or len(self.dataset) == 0:
            return control

        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        # Only rank 0 prints/saves. generate() itself still runs on every
        # rank below — under FSDP the forward pass is a collective op
        # (all-gather of sharded params), so skipping it on non-main ranks
        # would hang the other ranks waiting at the next sync point.
        is_main = state.is_world_process_zero

        if is_main:
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
                prediction = self._generate_prediction(
                    model, image, instruction, device
                )
            except Exception as e:
                if is_main:
                    print(f"[Sample Eval] sample={i + 1} failed: {e}")
                continue

            if not is_main:
                continue

            exact_match = prediction.strip() == reference.strip()
            print(f"Sample {i + 1}/{self.num_samples}")
            print(f"Instruction: {instruction}")
            print(f"Reference  : {reference[:160]}")
            print(f"Prediction : {prediction[:160]}")
            print(f"Exact Match: {exact_match}")
            print("-" * 80)

            step_results.append(
                {
                    "step": state.global_step,
                    "sample_idx": idx,
                    "instruction": instruction,
                    "reference": reference,
                    "prediction": prediction,
                    "exact_match": exact_match,
                }
            )

        if is_main and step_results:
            step_file = os.path.join(
                self.output_dir, f"step_{state.global_step:07d}.json"
            )
            with open(step_file, "w", encoding="utf-8") as f:
                json.dump(step_results, f, ensure_ascii=False, indent=2)

            self.all_results.extend(step_results)
            all_file = os.path.join(self.output_dir, "all_predictions.json")
            with open(all_file, "w", encoding="utf-8") as f:
                json.dump(self.all_results, f, ensure_ascii=False, indent=2)

            print(f"[Sample Eval] Saved {len(step_results)} predictions to {step_file}")

        if was_training:
            model.train()

        return control
