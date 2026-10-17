# Eval

Run a vision-language model (optionally with a LoRA adapter) on a Braille image set and write
`predictions.json`. Two equivalent backends with the same flags:

- `run_hf.py` — HuggingFace transformers, one sample at a time
- `run_vllm.py` — vLLM, batched (faster; adapter support depends on the model architecture)

Gated models need `HF_TOKEN` in the environment (or `huggingface-cli login`).

## Usage

```bash
# base model
python run_vllm.py --model Qwen/Qwen3.5-4B --output-name qwen3.5-4b

# with a LoRA adapter
python run_hf.py --model meta-llama/Llama-3.2-11B-Vision-Instruct \
    --adapter path/to/checkpoint --output-name llama-lora

# predefined task, quick check on 5 samples
python run_hf.py --model path/to/model --task obr_wmt --limit 5
```

Run `python run_hf.py --help` for all flags (`run_vllm.py` adds vLLM engine options).

## Data

`--label-file` is a JSONL with one `{"image_path": ..., "ground_truth": ...}` per line, with
`image_path` relative to `--image-root` (default `data`). Default label file:
`data/brailleBench.jsonl`.

## Output

`<output-dir>/<output-name>/<timestamp>/predictions.json` (default `results/`), a list of
`{"truth": ..., "predicted": ...}`. Text after a `####` marker in the model output is taken as
the prediction. Score it with `ocr_eval.py`.
