import json
import os
from dataclasses import dataclass, field

import torch
import transformers
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset

from braille_collate import BrailleCollator


class LossLoggerCallback(transformers.TrainerCallback):
    """Appends every train_loss / eval_loss entry Trainer logs to output_dir/loss_log.jsonl."""

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.log_path = os.path.join(output_dir, "loss_log.jsonl")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None:
            return control
        if "loss" not in logs and "eval_loss" not in logs:
            return control
        record = {"step": state.global_step, "epoch": logs.get("epoch", state.epoch), **logs}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return control


def compute_trainable_params(model: torch.nn.Module) -> None:
    """Compute trainable parameters."""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    train_params = sum([p.numel() for p in model_parameters])
    print(f"{sum([p.numel() for p in model.parameters()])} params and {train_params} trainable params")


class BrailleDataset(Dataset):
    """Reads image_dir + a labels.jsonl of {"image": "images/x.jpg", "source_text": "..."} records."""

    def __init__(self, image_dir: str, label_path: str):
        self.image_dir = image_dir
        self.records = []
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        print(f"[Dataset] Loaded {len(self.records)} records from {label_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = Image.open(os.path.join(self.image_dir, record["image"])).convert("RGB")
        return {"image": image, "answer": record["source_text"]}


@dataclass
class DataArguments:
    """Data arguments."""

    image_dir: str = field(default="master_data/train/")
    label_path: str = field(default="master_data/train/labels/labels.jsonl")
    # If > 0, hold out this many rows as the eval set (random, fixed by data_seed)
    eval_holdout_size: int = field(default=0)
    instruction: str = field(default="What Braille text is encoded in this image?")


@dataclass
class ModelArguments:
    """Model arguments."""

    model_id: str = field(default="meta-llama/Llama-3.2-11B-Vision-Instruct")


@dataclass
class TrainArgs(transformers.TrainingArguments):
    """Training arguments."""

    output_dir: str = field(default="checkpoints")
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=16)
    logging_steps: int = field(default=1)
    save_strategy: str = field(default="best")
    save_steps: float = field(default=0.05)
    num_train_epochs: int = field(default=1)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.1)
    warmup_ratio: float = field(default=0.1)
    lr_scheduler_type: str = field(default="linear")
    bf16: bool = field(default=True)
    fp16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=False)
    save_total_limit: int = field(default=30)
    load_best_model_at_end: bool = field(default=True)
    log_level: str = field(default="debug")
    save_safetensors: bool = field(default=True)
    eval_strategy: str = field(default="steps")
    eval_steps: float = field(default=0.05)
    seed: int = field(default=12345)
    data_seed: int = field(default=12345)
    dataloader_num_workers: int = field(default=4)
    logging_nan_inf_filter: bool = field(default=False)
    run_name: str = field(default="llama-3.2-11b-vision-instruct-braille")
    project_name: str = field(default="braille-vision")
    remove_unused_columns: bool = field(default=False)
    lora_rank: int = field(default=8)
    lora_alpha: int = field(default=16)
    # Force every visible GPU onto a single process with the whole model on one
    # device, so Trainer's own local_rank==-1 + n_gpu>1 path wraps it in
    # torch.nn.DataParallel. Leave off (default) to let the model shard itself
    # across the visible GPUs instead (device_map="auto", no data parallelism),
    # or launch with torchrun for one full copy per GPU (DDP) -- see
    # train_llama_ddp.sh / train_llama_dp.sh.
    data_parallel: bool = field(default=False)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainArgs))
    model_args, data_args, train_args = parser.parse_args_into_dataclasses()

    # Under torchrun, rank 0 builds the dataset cache first so ranks don't race on the same files
    with train_args.main_process_first(desc="dataset loading"):
        dataset = BrailleDataset(image_dir=data_args.image_dir, label_path=data_args.label_path)
        if data_args.eval_holdout_size > 0:
            split = torch.utils.data.random_split(
                dataset,
                [len(dataset) - data_args.eval_holdout_size, data_args.eval_holdout_size],
                generator=torch.Generator().manual_seed(train_args.data_seed),
            )
            train_dataset, eval_dataset = split
        else:
            train_dataset, eval_dataset = dataset, None

    processor = transformers.AutoProcessor.from_pretrained(
        model_args.model_id,
        padding=True,
    )
    processor.tokenizer.padding_side = "right"

    collator = BrailleCollator(processor=processor, instruction=data_args.instruction)

    # Single process: shard the model across visible GPUs (model parallel, no
    # data parallelism). Under torchrun: each rank keeps a full copy on its own
    # GPU (DDP). --data_parallel: whole model on one GPU, Trainer wraps it in
    # torch.nn.DataParallel across the other visible GPUs.
    if "LOCAL_RANK" in os.environ:
        device_map = {"": int(os.environ["LOCAL_RANK"])}
    elif train_args.data_parallel:
        device_map = {"": 0}
    else:
        device_map = "auto"
    model = transformers.MllamaForConditionalGeneration.from_pretrained(
        model_args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

    lora_config = {
        "r": train_args.lora_rank,
        "lora_alpha": train_args.lora_alpha,
        "lora_dropout": 0.01,
        "target_modules": ["q_proj", "v_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
    }
    peft_config = LoraConfig(**lora_config)
    model = get_peft_model(model, peft_config)

    compute_trainable_params(model)

    if train_args.project_name is not None:
        os.environ["WANDB_PROJECT"] = train_args.project_name

    if eval_dataset is None:
        train_args.eval_strategy = "no"
        train_args.save_strategy = "steps"
        train_args.load_best_model_at_end = False

    trainer = transformers.Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.add_callback(LossLoggerCallback(train_args.output_dir))

    trainer.train()


if __name__ == "__main__":
    train()
