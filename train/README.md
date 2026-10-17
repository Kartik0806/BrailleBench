# train

Fine-tune an image-text-to-text model to transcribe braille images into plain text. Only the answer
and the end-of-turn token are supervised; the prompt, image tokens and padding are masked.

```
train.py            entry point: wires everything together and runs the Trainer
cli.py              command-line arguments
braille_dataset.py  dataset, instruction prompts, train/val split
collator.py         model-agnostic loss-masking collator
callbacks.py        prints and saves sample predictions during training
modeling.py         model loading, LoRA wrapping, final save
model_handler.py    per-checkpoint handling (chat template, end-of-turn, stop ids)
```

## Usage

```bash
# one GPU
python train.py --model_id Qwen/Qwen3.5-2B \
    --image_dir data/train/ --label_path data/train/labels/labels.jsonl \
    --output_dir output/qwen-2b --epochs 1 --batch_size 1 --grad_accum 8 --lr 7e-6

# several GPUs (effective batch = batch_size x grad_accum x GPUs)
accelerate launch --config_file <accelerate_config.yaml> train.py <same arguments>

# LoRA instead of full fine-tuning
python train.py ... --use_lora --lora_r 16 --lora_alpha 32
```

Run `python train.py --help` for all flags. Useful ones: `--gradient_checkpointing`,
`--eval_split`, `--no_eval`, `--sample_eval_steps`, `--disable_sample_eval`,
`--resume_from_checkpoint`.

## Data

`--label_path` is a JSONL with one record per line and `--image_dir` is the directory the images are
relative to. This is exactly what [`generate-data`](../generate-data/) writes:

```
data/train/images/sample_000000.jpg
data/train/labels/labels.jsonl      {"id": ..., "image": "images/sample_000000.jpg", "source_text": "...", ...}
```

The answer is `source_text` (or `reasoning_text` if present). LLaVA-style records with a
`conversations` list (`human` / `gpt` turns) are also accepted.

The validation set is found by replacing `train` with `val` in `--image_dir` and `--label_path`, so
keep a sibling `data/val/`. If it does not exist, evaluation is switched off and a `[Eval] disabled`
message is printed; pass `--no_eval` to do that on purpose.

## Models

The chat template, end-of-turn token and stop ids are read from the checkpoint itself, so changing
`--model_id` needs no code change. Qwen3.5, Qwen2/2.5/3-VL and Gemma 3 / 3n / 4 are recognised in
`model_handler.py` (add per-family chat-template or processor settings there if a model needs them);
other image-text-to-text checkpoints use the generic path and are not all verified.

## Output

- Training resumes automatically from the latest checkpoint in `--output_dir`.
- At the end the model (LoRA merged into the base) and the processor are saved to `--output_dir`, ready
  for [`run-evals`](../run-evals/).
- Every `--sample_eval_steps` steps a few predictions are printed and saved under `sample_predictions/`.

## Llama 3.2 Vision

`train_llama_br.py` is a separate LoRA trainer for `meta-llama/Llama-3.2-11B-Vision-Instruct`
(it uses `braille_collate.py`, which is specific to Mllama's token ids):

```bash
torchrun --standalone --nproc_per_node=8 train_llama_br.py \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --eval_holdout_size 1000 --output_dir output/llama-lora --lora_rank 16 --lora_alpha 32
```
