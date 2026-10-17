"""Fine-tune a vision-language model on Braille OCR.

    accelerate launch --config_file accelerate_config.yaml train/train.py
    --model_id Qwen/Qwen3.5-2B --image_dir data/train/ --label_path data/train/labels/labels.jsonl

Layout: cli.py (arguments), braille_dataset.py (dataset + splits), collator.py (loss masking),
callbacks.py (sample predictions), modeling.py (model / LoRA / saving), model_handler.py
(per-checkpoint-family handling). Gated models need HF_TOKEN in the environment.
"""

import os
import random

import torch
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from braille_dataset import build_datasets
from callbacks import SamplePredictionCallback
from cli import parse_args
from collator import AnswerOnlyCollator
from modeling import apply_lora, load_model, save_final


def build_training_args(args, has_eval: bool) -> TrainingArguments:
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        # a float < 1 is a fraction of the total optimizer steps, which Trainer
        # computes with the number of GPUs (world size) included
        warmup_steps=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=100,
        save_strategy="steps",
        lr_scheduler_type="cosine",
        save_total_limit=2,
        eval_strategy="steps" if has_eval else "no",
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        report_to="none",
        seed=args.seed,
        label_names=["labels"],
        load_best_model_at_end=has_eval,
        metric_for_best_model="loss",
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
    )


def main():
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    handler, model = load_model(args.model_id, args.bf16, args.gradient_checkpointing)
    if args.use_lora:
        model = apply_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout)

    train_ds, eval_ds, sample_eval_ds = build_datasets(
        args.image_dir, args.label_path, args.eval_split, args.no_eval
    )

    trainer = Trainer(
        model=model,
        args=build_training_args(args, has_eval=eval_ds is not None),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=AnswerOnlyCollator(handler=handler),
        processing_class=handler.processor,  # needed so Trainer knows the processor
    )

    if not args.disable_sample_eval:
        if sample_eval_ds is not None:
            print("Adding Sample Prediction Callback on eval_ds")
        trainer.add_callback(
            SamplePredictionCallback(
                handler=handler,
                dataset=sample_eval_ds if sample_eval_ds is not None else train_ds,
                every_steps=args.sample_eval_steps,
                num_samples=args.sample_eval_count,
                max_new_tokens=args.sample_eval_max_new_tokens,
                seed=args.seed,
            )
        )

    # Auto-detect the latest checkpoint under output_dir unless one is given explicitly
    resume_ckpt = args.resume_from_checkpoint
    if resume_ckpt is None and os.path.isdir(args.output_dir):
        resume_ckpt = get_last_checkpoint(args.output_dir)
    if resume_ckpt:
        print(f"[Resume] resuming from checkpoint: {resume_ckpt}")

    trainer.train(resume_from_checkpoint=resume_ckpt)

    save_final(trainer, model, handler.processor, args.output_dir, merge_lora=args.use_lora)


if __name__ == "__main__":
    main()
