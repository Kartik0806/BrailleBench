# Train on master_data/train (now including the 20k CC-MAIN-2013-20 samples merged by merge_into_master.py)
# with a different seed. Identical to train_realmatch_combined.sh (which produced output/qwen2b-m with
# the default --seed 42) except for --seed and --output_dir.
#   SEED=7 bash train_master_seed.sh        # default SEED=1337
# Fresh --output_dir per seed (train.py auto-resumes from checkpoints in it).
SEED="${SEED:-0810}"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file "accelerate_config.yaml"  train/train.py \
    --model_id "meta-llama/Llama-3.2-11B-Vision-Instruct" \
    --image_dir master_data/train/ \
    --label_path master_data/train/labels/labels.jsonl \
    --output_dir ./output/meta-lora-${SEED} \
    --epochs 1 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 1e-4 \
    --eval_split val \
    --seed "${SEED}" \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
