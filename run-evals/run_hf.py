"""Evaluate a vision-language model (optionally with a LoRA adapter) on a Braille image set using
HuggingFace transformers, one sample at a time with greedy decoding.

Example:
    python run_hf.py --model meta-llama/Llama-3.2-11B-Vision-Instruct --adapter path/to/checkpoint

Gated models need HF_TOKEN set in the environment (or `huggingface-cli login`).
run_vllm.py takes the same flags and writes the same predictions.json.
"""

import argparse

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from eval_common import (
    add_common_args,
    build_messages,
    load_samples,
    resolve_args,
    save_predictions,
)


def load_model(args):
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter, device_map="auto")
        print(f"Loaded {args.model} with adapter {args.adapter}")
    else:
        print(f"Loaded {args.model} without adapter")
    return model.eval()


def main(args):
    samples = load_samples(args.label_file, args.image_root, args.limit)
    processor = AutoProcessor.from_pretrained(args.processor, trust_remote_code=args.trust_remote_code)
    model = load_model(args)

    texts = []
    for sample in tqdm(samples, desc="Processing samples"):
        image = Image.open(f"{args.image_root}/{sample['image_path']}").convert("RGB")
        messages = build_messages(image, args.prompt, args.system_prompt)

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]

        text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        print(text)
        texts.append(text)

    save_predictions(samples, texts, args.output_dir, args.output_name, args.model)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    return resolve_args(parser.parse_args())


if __name__ == "__main__":
    main(parse_args())
