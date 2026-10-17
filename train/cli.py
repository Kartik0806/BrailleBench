"""Command-line arguments for train.py."""

import argparse

from model_handler import DEFAULT_MODEL_ID


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a vision-language model on Braille OCR.")

    data = parser.add_argument_group("data")
    data.add_argument("--image_dir", required=True, help="Root dir that image_path is relative to")
    data.add_argument("--label_path", default="data/labels/labels.jsonl")
    data.add_argument("--eval_split", default="val")
    data.add_argument("--no_eval", action="store_true")
    # Not read anywhere; accepted so existing launch scripts keep working.
    data.add_argument("--label_file", type=str, default="labels_with_reasoning.json", help=argparse.SUPPRESS)

    model = parser.add_argument_group("model")
    model.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    model.add_argument("--output_dir", default="./braille-ft")
    model.add_argument("--use_lora", action="store_true", help="Fine-tune with LoRA adapters (recommended)")
    model.add_argument("--lora_r", type=int, default=16)
    model.add_argument("--lora_alpha", type=int, default=32)
    model.add_argument("--lora_dropout", type=float, default=0.05)
    model.add_argument("--bf16", action="store_true", default=True)
    model.add_argument("--gradient_checkpointing", action="store_true", default=False)

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--epochs", type=int, default=1)
    optim.add_argument("--batch_size", type=int, default=8)
    optim.add_argument("--grad_accum", type=int, default=1)
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--warmup_ratio", type=float, default=0.03)
    optim.add_argument("--weight_decay", type=float, default=0.01)
    optim.add_argument("--max_grad_norm", type=float, default=1.0)
    optim.add_argument("--seed", type=int, default=42)

    sample = parser.add_argument_group("sample predictions during training")
    sample.add_argument("--sample_eval_steps", type=int, default=100)
    sample.add_argument("--sample_eval_count", type=int, default=5)
    sample.add_argument("--sample_eval_max_new_tokens", type=int, default=128)
    sample.add_argument("--disable_sample_eval", action="store_true")

    run = parser.add_argument_group("run")
    run.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Explicit checkpoint dir to resume from. If omitted, auto-detects "
        "the latest checkpoint under --output_dir (if any).",
    )
    run.add_argument(
        "--ddp_find_unused_parameters",
        default=False,
        help="Set to True if using DDP and the model has unused parameters. "
        "Otherwise leave False (default) for better performance.",
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)
