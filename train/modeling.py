"""Model loading, LoRA wrapping and final saving."""

import torch

from model_handler import ModelHandler

try:
    from peft import LoraConfig, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


def load_model(model_id: str, bf16: bool, gradient_checkpointing: bool):
    """Processor + model. Model-specific handling lives in model_handler.py."""
    print(f"[Load] processor + model from {model_id}")
    handler = ModelHandler.from_pretrained(model_id)

    model = handler.load_model(
        dtype=torch.bfloat16 if bf16 else torch.float32,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return handler, model


def apply_lora(model, r: int, alpha: int, dropout: float):
    if not PEFT_AVAILABLE:
        raise RuntimeError("peft is not installed. Run: pip install peft")
    print("[LoRA] wrapping model with LoRA adapters")
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # "all-linear" lets peft auto-match linear layers regardless of
        # architecture naming (q_proj/k_proj vs qkv_proj vs fused
        # variants), so this doesn't need to change per model_id.
        target_modules="all-linear",
    )
    # autocast_adapter_dtype=True (peft's default) would upcast the new LoRA
    # weights to float32 while the frozen base stays bf16; FSDP2's fully_shard
    # requires one dtype per wrapped unit and asserts on that split.
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    model.print_trainable_parameters()
    return model


def save_final(trainer, model, processor, output_dir: str, merge_lora: bool):
    """Save model + processor. Gathers full (unsharded) weights under FSDP."""
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    if merge_lora:
        # merge weights back into the base model for saving, so the final checkpoint is a single
        # model that can be loaded without peft
        print("[LoRA] merging LoRA weights into base model for saving")
        model = model.merge_and_unload()

    trainer.save_model(output_dir)
    if trainer.accelerator.is_main_process:
        processor.save_pretrained(output_dir)
        print(f"[Done] model + processor saved to {output_dir}")
