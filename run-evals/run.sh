# Same run as the old hard-coded hf.py config (Llama-3.2-11B-Vision + llama-ddp-8 adapter).
CUDA_VISIBLE_DEVICES=0 python run_hf.py \
    --model meta-llama/Llama-3.2-11B-Vision-Instruct \
    --adapter ../Braille-Vision/output/llama-ddp-8/checkpoint-4309 \
    --output-name fw-8
