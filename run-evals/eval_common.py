"""Pieces shared by run_hf.py and run_vllm.py so the two backends stay equivalent:
the CLI flags, data loading, chat messages, answer extraction and the predictions.json format.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from data_utils import get_data

DEFAULT_LABEL_FILE = "data/brailleBench.jsonl"
DEFAULT_PROMPT = "Identify the Braille cells and translate them into plain text."

# Predefined eval sets, selected with --task. A task only supplies defaults for --label-file and
# --prompt; explicit flags still win.
TASK_CONFIGS = {
    "obr_wmt": {
        # Same 1000-row split tasks/obr/train.py and tasks/obr/train_qwen.py hold out of
        # gpantaz/wmt2024train at training time (test_size=1000, seed=12345). See
        # data/make_obr_wmt_val.py for how this was generated.
        "label_file": "data/obr_wmt1000/labels.jsonl",
        "prompt": "Translate the Braille to English.",
    },
}


def add_common_args(parser):
    model = parser.add_argument_group("model")
    model.add_argument("--model", required=True, help="HF hub id or local path of the base model.")
    model.add_argument("--adapter", default=None, help="LoRA adapter (hub id or local path). Omit to evaluate the base model.")
    model.add_argument("--processor", default=None, help="Processor/tokenizer source. Defaults to --model.")
    model.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    model.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass trust_remote_code to the model/processor loaders (default: on).",
    )

    data = parser.add_argument_group("data")
    data.add_argument("--task", default=None, choices=sorted(TASK_CONFIGS), help="Predefined eval set + prompt.")
    data.add_argument("--label-file", default=None, help=f"JSONL with image_path/ground_truth per line (default: {DEFAULT_LABEL_FILE}).")
    data.add_argument("--image-root", default="data", help="Directory that image_path entries are relative to (default: data).")
    data.add_argument("--limit", type=int, default=None, help="Only evaluate the first N samples whose image exists.")

    gen = parser.add_argument_group("generation")
    gen.add_argument("--prompt", default=None, help=f"User prompt (default: {DEFAULT_PROMPT!r}).")
    gen.add_argument("--system-prompt", default=None, help="Optional system prompt. Not used by default.")
    gen.add_argument("--max-new-tokens", type=int, default=1024)

    out = parser.add_argument_group("output")
    out.add_argument("--output-dir", default="results", help="Predictions go to <output-dir>/<output-name>/<timestamp>/predictions.json.")
    out.add_argument("--output-name", default=None, help="Subfolder name. Defaults to --model with '/' replaced by '_'.")
    return parser


def resolve_args(args):
    """Fill label_file/prompt from --task (explicit flags win), then the built-in defaults."""
    task = TASK_CONFIGS.get(args.task, {})
    args.label_file = args.label_file or task.get("label_file") or DEFAULT_LABEL_FILE
    args.prompt = args.prompt or task.get("prompt") or DEFAULT_PROMPT
    args.processor = args.processor or args.model
    return args


def load_samples(label_file, image_root, limit=None):
    """Read the label file and keep samples whose image exists, so samples and outputs line up."""
    data = get_data(label_file)
    root = Path(image_root)
    samples = [s for s in data if (root / s["image_path"]).exists()]
    if len(samples) < len(data):
        print(f"Skipping {len(data) - len(samples)} of {len(data)} samples with missing images under {root}")
    if limit is not None:
        samples = samples[:limit]
    return samples


def build_messages(image, prompt, system_prompt=None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages


def extract_prediction(text):
    match = re.search(r"####\s*(.*)$", text.strip(), re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def save_predictions(samples, texts, output_dir, output_name, model):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir) / (output_name or model.replace("/", "_")) / run_id
    out.mkdir(parents=True, exist_ok=True)

    results = [
        {"truth": s.get("ground_truth") or "N/A", "predicted": extract_prediction(t)}
        for s, t in zip(samples, texts)
    ]
    with open(out / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} predictions to {out / 'predictions.json'}")
