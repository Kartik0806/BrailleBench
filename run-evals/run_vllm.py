"""Evaluate a vision-language model (optionally with a LoRA adapter) on a Braille image set using
vLLM with greedy decoding and batched generation.

Example:
    python run_vllm.py --model Qwen/Qwen3.5-4B --output-name qwen3.5-4b

Gated models need HF_TOKEN set in the environment (or `huggingface-cli login`).
run_hf.py takes the same flags and writes the same predictions.json. Adapter support depends on
the model architecture being LoRA-capable in vLLM; use run_hf.py if it is not.
"""

import argparse
import json
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from eval_common import (
    add_common_args,
    build_messages,
    load_samples,
    resolve_args,
    save_predictions,
)


def build_requests(samples, processor, args):
    requests = []
    for sample in samples:
        image = Image.open(Path(args.image_root) / sample["image_path"]).convert("RGB")
        messages = build_messages(image, args.prompt, args.system_prompt)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        requests.append({"prompt": prompt, "multi_modal_data": {"image": image}})
    return requests


def adapter_rank(adapter):
    """LoRA rank from a local adapter's config, so vLLM can size its LoRA buffers."""
    config = Path(adapter) / "adapter_config.json"
    if config.is_file():
        return json.loads(config.read_text())["r"]
    return None


def build_llm(args):
    kwargs = dict(
        model=args.model,
        tokenizer=args.processor,
        dtype=args.dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=args.trust_remote_code,
    )
    if args.max_model_len:
        kwargs["max_model_len"] = args.max_model_len
    if args.gpu_memory_utilization:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.adapter:
        kwargs["enable_lora"] = True
        rank = args.max_lora_rank or adapter_rank(args.adapter)
        if rank:
            kwargs["max_lora_rank"] = rank
    return LLM(**kwargs)


def main(args):
    samples = load_samples(args.label_file, args.image_root, args.limit)
    processor = AutoProcessor.from_pretrained(args.processor, trust_remote_code=args.trust_remote_code)
    llm = build_llm(args)

    lora_request = None
    if args.adapter:
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("adapter", 1, args.adapter)

    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_new_tokens)
    outputs = llm.generate(
        build_requests(samples, processor, args),
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    texts = [o.outputs[0].text for o in outputs]
    for text in texts:
        print(text)
    save_predictions(samples, texts, args.output_dir, args.output_name, args.model)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    vllm_group = parser.add_argument_group("vLLM engine")
    vllm_group.add_argument("--tensor-parallel-size", type=int, default=1)
    vllm_group.add_argument("--max-num-seqs", type=int, default=32)
    vllm_group.add_argument("--max-model-len", type=int, default=None)
    vllm_group.add_argument("--gpu-memory-utilization", type=float, default=None)
    vllm_group.add_argument("--max-lora-rank", type=int, default=None, help="Defaults to `r` from a local adapter_config.json.")
    return resolve_args(parser.parse_args())


if __name__ == "__main__":
    main(parse_args())
