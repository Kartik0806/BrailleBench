"""
Model-specific handling for train/train.py.

Everything that differs between vision-language checkpoints lives here, so train.py does
not change when --model_id does:

  * which processor and model class to load (read from the checkpoint's config.json),
  * how a prompt is rendered (the checkpoint's chat template, plus per-family kwargs),
  * how the processor is called (one image per sample; no second BOS when the template
    already writes one, as Gemma 3's does),
  * how an assistant turn ends and which tokens sit directly in front of every answer,
  * which token ids stop generate() (tokenizer eos + the template's end-of-turn token +
    the checkpoint's own generation config).

    handler = ModelHandler.from_pretrained("Qwen/Qwen3.5-2B")   # processor only
    model = handler.load_model(dtype=torch.bfloat16)             # model, stop ids synced

Nothing here hardcodes a token string: end-of-turn, generation header and BOS handling
are read off the chat template by rendering probe conversations.
"""

from dataclasses import dataclass, field
from typing import Any

import transformers
from PIL import Image
from transformers import AutoConfig, AutoProcessor, GenerationConfig
from transformers.models.auto import modeling_auto
import torch
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-2B"

_PROBE_REPLY = "ANSWERSENTINEL"


# ---------------------------------------------------------------------------
# Families: settings the checkpoint's own processor / chat template do not carry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FamilySpec:
    """Per-family settings. Empty dicts mean "use the checkpoint's own defaults"."""

    name: str
    model_types: tuple = ()
    # extra kwargs for every apply_chat_template call (e.g. {"enable_thinking": False})
    chat_template_kwargs: dict = field(default_factory=dict)
    # extra kwargs for AutoProcessor.from_pretrained (e.g. image resolution limits)
    processor_kwargs: dict = field(default_factory=dict)


# Keyed by config.json "model_type". Add an entry when a new model needs settings that
# its chat template / processor config do not already carry; unknown types use GENERIC.
FAMILIES = (
    FamilySpec("qwen3.5", ("qwen3_5", "qwen3_5_moe")),  # also datalab-to/chandra-ocr-2
    FamilySpec("qwen-vl", ("qwen2_vl", "qwen2_5_vl", "qwen3_vl", "qwen3_vl_moe")),
    FamilySpec("gemma", ("gemma3", "gemma3n", "gemma4")),
)
GENERIC = FamilySpec("generic")


def family_for(model_type: str) -> FamilySpec:
    return next((f for f in FAMILIES if model_type in f.model_types), GENERIC)


def model_class_for(model_type: str):
    """AutoModelForImageTextToText where it maps the type, else AutoModelForMultimodalLM."""
    if model_type in modeling_auto.MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES:
        return transformers.AutoModelForImageTextToText
    multimodal = getattr(modeling_auto, "MODEL_FOR_MULTIMODAL_LM_MAPPING_NAMES", {})
    if model_type in multimodal and hasattr(transformers, "AutoModelForMultimodalLM"):
        return transformers.AutoModelForMultimodalLM
    raise ValueError(
        f"model_type {model_type!r} is not an image-text-to-text model in transformers "
        f"{transformers.__version__}; upgrade transformers or pick another --model_id."
    )


def _as_list(ids):
    if ids is None:
        return []
    return list(ids) if isinstance(ids, (list, tuple)) else [ids]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
class ModelHandler:
    """Processor + chat-template knowledge for one checkpoint (see module docstring)."""

    def __init__(self, model_id: str, processor: Any, config: Any, family: FamilySpec):
        self.model_id = model_id
        self.processor = processor
        self.config = config
        self.family = family
        self.model_type = config.model_type
        self.model_class = model_class_for(self.model_type)

        tok = self.tokenizer
        tok.padding_side = "right"  # right-pad for training
        self._resolve_eos_pad()

        self.end_of_turn_text, self.end_of_turn_id, self.trailing_ids = self._resolve_end_of_turn()
        self.header_ids = self._resolve_generation_header()
        # Gemma 3: the template writes <bos> and the tokenizer adds another one.
        self.encode_kwargs = {}
        if self._template_writes_bos() and self._tokenizer_adds_bos():
            self.encode_kwargs["add_special_tokens"] = False
        self._check_single_bos()

        print(
            f"[Model] {model_id}: model_type={self.model_type} family={family.name} "
            f"processor={type(processor).__name__} model_class={self.model_class.__name__}"
        )
        print(
            f"[Model] end_of_turn={self.end_of_turn_text!r} (id={self.end_of_turn_id}) "
            f"header={tok.decode(self.header_ids)!r} encode_kwargs={self.encode_kwargs}"
        )

    @classmethod
    def from_pretrained(cls, model_id: str = DEFAULT_MODEL_ID, **processor_kwargs):
        config = AutoConfig.from_pretrained(model_id)
        family = family_for(config.model_type)
        kwargs = {**family.processor_kwargs, **processor_kwargs}
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        return cls(model_id, processor, config, family)

    @property
    def tokenizer(self):
        return self.processor.tokenizer

    # -- rendering / encoding ------------------------------------------------
    @staticmethod
    def messages(instruction: str, answer: str = None) -> list:
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            },
        ]
        if answer is not None:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return msgs

    def apply_chat_template(self, messages: list, add_generation_prompt: bool) -> str:
        return self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **self.family.chat_template_kwargs,
        )

    def prompt_text(self, instruction: str) -> str:
        """Inference-time prompt: user turn + generation header, template defaults."""
        return self.apply_chat_template(self.messages(instruction), add_generation_prompt=True)

    def training_text(self, instruction: str, answer: str) -> str:
        """Inference prompt + answer + the template's own end-of-turn text."""
        return self.prompt_text(instruction) + answer + self.end_of_turn_text

    def encode(self, texts: list, images: list):
        """Processor call for a batch: one image per text, padded, as torch tensors."""
        return self.processor(
            text=texts,
            images=[[image] for image in images],
            return_tensors="pt",
            padding=True,
            **self.encode_kwargs,
        )

    def answer_token_ids(self, answers: list) -> list:
        return self.tokenizer(answers, add_special_tokens=False)["input_ids"]

    # -- model ---------------------------------------------------------------
    def load_model(self, **kwargs):
        """Load the model with the right Auto class and sync stop/pad ids onto it."""
        model = self.model_class.from_pretrained(self.model_id, **kwargs)
        self.sync_generation_config(model)
        return model

    def stop_token_ids(self, model=None) -> list:
        """tokenizer eos, the template's end-of-turn, then the checkpoint's own eos ids."""
        checkpoint = []
        if model is not None:
            checkpoint = _as_list(getattr(model.generation_config, "eos_token_id", None))
        else:
            try:
                checkpoint = _as_list(GenerationConfig.from_pretrained(self.model_id).eos_token_id)
            except OSError:
                checkpoint = _as_list(GenerationConfig.from_model_config(self.config).eos_token_id)
        ids = []
        for i in [self.tokenizer.eos_token_id, self.end_of_turn_id] + checkpoint:
            if i is not None and i not in ids:
                ids.append(i)
        return ids

    def sync_generation_config(self, model) -> list:
        """generate() stops on generation_config.eos_token_id, so it must hold the end-of-turn id."""
        ids = self.stop_token_ids(model)
        model.generation_config.eos_token_id = ids
        model.config.eos_token_id = ids
        if model.generation_config.pad_token_id is None:
            model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        print(
            f"[EOS/PAD] stop ids={ids} {self.tokenizer.convert_ids_to_tokens(ids)}  "
            f"pad_token={self.tokenizer.pad_token!r} (id={self.tokenizer.pad_token_id})"
        )
        return ids

    # -- resolution helpers ----------------------------------------------------
    def _resolve_eos_pad(self):
        """Tokenizer eos falls back to the checkpoint's eos; pad falls back to eos."""
        tok = self.tokenizer
        if tok.eos_token_id is None or tok.eos_token_id == getattr(tok, "unk_token_id", None):
            ids = _as_list(GenerationConfig.from_model_config(self.config).eos_token_id)
            if not ids:
                raise ValueError(
                    f"Could not resolve an eos_token_id for {self.model_id!r}. Check its "
                    f"generation_config.json / tokenizer_config.json on the Hub."
                )
            tok.eos_token_id = ids[0]
            tok.eos_token = tok.convert_ids_to_tokens(ids[0])
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
            tok.pad_token = tok.eos_token

    def _special_ids(self) -> set:
        tok = self.tokenizer
        special = set(tok.all_special_ids)
        special |= {i for i, t in tok.added_tokens_decoder.items() if t.special}
        return special

    def _resolve_end_of_turn(self):
        """
        Ask the chat template how *it* ends an assistant turn: render a probe conversation
        with a sentinel reply and keep everything after the sentinel ("<|im_end|>\\n" for
        Qwen, "<end_of_turn>\\n" for Gemma 3, "<turn|>\\n" for Gemma 4). Returns
        (text, end_of_turn_id, trailing_ids): the end-of-turn token is supervised, the
        tokens after it (the newline) stay in input_ids but are never supervised.
        """
        probe = self.apply_chat_template(
            self.messages("hi", _PROBE_REPLY), add_generation_prompt=False
        )
        if probe.count(_PROBE_REPLY) != 1:
            raise ValueError(
                f"Cannot locate the assistant reply in the chat template output: {probe!r}"
            )
        text = probe.split(_PROBE_REPLY)[1]
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids or ids[0] not in self._special_ids():
            raise ValueError(
                "The chat template does not end an assistant turn with a special "
                f"end-of-turn token (text after the reply: {text!r}); cannot build "
                "labels that teach the model to stop."
            )
        return text, ids[0], ids[1:]

    def _resolve_generation_header(self) -> list:
        """
        Token ids add_generation_prompt=True appends (Qwen3.5:
        "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n"), i.e. what must sit directly
        in front of every answer. Empty if it cannot be isolated.
        """
        msgs = self.messages("hi")
        bare = self.apply_chat_template(msgs, add_generation_prompt=False)
        full = self.apply_chat_template(msgs, add_generation_prompt=True)
        if full == bare or not full.startswith(bare):
            return []
        return self.tokenizer(full[len(bare):], add_special_tokens=False)["input_ids"]

    def _template_writes_bos(self) -> bool:
        bos = self.tokenizer.bos_token
        return bool(bos) and self.prompt_text("hi").startswith(bos)

    def _tokenizer_adds_bos(self) -> bool:
        bos_id = self.tokenizer.bos_token_id
        return bos_id is not None and self.tokenizer("x")["input_ids"][:1] == [bos_id]

    def _check_single_bos(self):
        """Encode a probe exactly like training does; a doubled BOS would be silently trained on."""
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            return
        probe = self.encode([self.prompt_text("hi")], [Image.new("RGB", (64, 64), "white")])
        ids = probe["input_ids"][0].tolist()
        if ids[:2] == [bos_id, bos_id]:
            raise ValueError(
                f"{self.model_id}: encoded prompt starts with two BOS tokens; the processor "
                f"call needs add_special_tokens=False for this template."
            )
