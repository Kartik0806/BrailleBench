# Braille-Vision

Teaching vision-language models to read braille from photographs. The project covers the whole loop:
render synthetic braille photos that statistically match real ones, fine-tune a model on them, and
evaluate it on real braille images.

| Folder | What it does |
|---|---|
| [`generate-data/`](generate-data/) | Synthetic data generator: UEB Grade 1 / Grade 2 braille rendered as photos with bounding-box labels, plus a "real-matching" mode that follows measured statistics of real photos. |
| [`train/`](train/) | Fine-tuning (full or LoRA) of image-text-to-text models such as Qwen3.5, Qwen-VL and Gemma. |
| [`run-evals/`](run-evals/) | Evaluation of a model (optionally with a LoRA adapter) on a braille image set, with Hugging Face or vLLM. |
| [`real-braille-images/`](real-braille-images/) | Real braille photographs used for evaluation and for measuring the statistics the generator matches. |

## Workflow

```
generate-data  ->  data/train, data/val   (images + labels.jsonl)
train          ->  output/<run>           (fine-tuned model + processor)
run-evals      ->  results/<name>/<timestamp>/predictions.json
```

1. **Generate** a training and a validation set: see [`generate-data/README.md`](generate-data/README.md).
2. **Train** on it: see [`train/README.md`](train/README.md).
3. **Evaluate** the result on real images: see [`run-evals/README.md`](run-evals/README.md).

## Requirements

- Python 3 with PyTorch and Hugging Face `transformers` (training and evaluation), plus `peft` for LoRA.
- A CUDA GPU for training and evaluation. Data generation runs on CPU.
- [liblouis](https://liblouis.io/) with Python bindings for the generator (see its README).
- Gated Hugging Face models need `HF_TOKEN` set in the environment (or `huggingface-cli login`).
  No credentials are stored in this repository.
