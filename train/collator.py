"""Model-agnostic collator: supervises only the answer and the chat template's end-of-turn token."""

from dataclasses import dataclass

import torch

from model_handler import ModelHandler


@dataclass
class AnswerOnlyCollator:
    """
    Chat collator: loss on the answer and the template's end-of-turn token only.

    Training text = inference prompt + answer + the template's own end-of-turn
    (see ModelHandler.training_text). Labels are -100 for the whole prompt
    (image tokens, vision markers, instruction, role headers, any think block),
    for padding, and for whatever the template appends after the end-of-turn
    token.

    The supervised span is located from the end of each real sequence, taken
    from attention_mask, so it does not depend on the padding side, on
    pad_token_id == eos_token_id, or on how many tokens an image expands to.
    Anything that does not line up raises instead of training on wrong tokens.
    """

    handler: ModelHandler

    def __call__(self, features):
        h = self.handler
        tok = h.tokenizer
        eot_id, trailing_ids, header_ids = h.end_of_turn_id, h.trailing_ids, h.header_ids

        answers = [f["answer"] for f in features]
        texts = [h.training_text(f["instruction"], f["answer"]) for f in features]
        batch = h.encode(texts, [f["image"] for f in features])

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = torch.full_like(input_ids, -100)

        answer_ids = h.answer_token_ids(answers)
        for i, ids in enumerate(answer_ids):
            real = attention_mask[i].nonzero(as_tuple=True)[0]
            first, last = int(real[0]), int(real[-1])
            if last - first + 1 != real.numel():
                raise ValueError(
                    f"Collator sample {i}: attention_mask is not one contiguous block"
                )
            if eot_id in ids:
                raise ValueError(
                    f"Collator sample {i}: the answer contains the end-of-turn token "
                    f"{tok.convert_ids_to_tokens(eot_id)!r}, the model would learn to "
                    f"stop early (answer={answers[i][:80]!r})"
                )

            # [prompt ... | answer | end-of-turn | trailing template text]
            tail = ids + [eot_id] + trailing_ids
            end = last + 1
            start = end - len(tail)
            got = input_ids[i, max(start, first) : end].tolist()
            if start <= first or got != tail:
                n = min(len(tail), 8)
                raise ValueError(
                    f"Collator sample {i}: the sequence does not end with the "
                    f"tokenised answer + end-of-turn (expected ...{tail[-n:]}, got "
                    f"...{got[-n:]}); the answer probably contains a special-token "
                    f"string, or the tokenizer merges across the prompt/answer "
                    f"boundary (answer={answers[i][:80]!r})"
                )

            # the answer must directly follow the generation-prompt header
            h0 = start - len(header_ids)
            if h0 < first or input_ids[i, h0:start].tolist() != header_ids:
                raise ValueError(
                    f"Collator sample {i}: the answer is not preceded by the "
                    f"generation-prompt header (expected {header_ids}, got "
                    f"{input_ids[i, max(h0, first):start].tolist()}); the training "
                    f"prompt differs from the inference prompt "
                    f"(answer={answers[i][:80]!r})"
                )

            # supervise answer + end-of-turn token; the trailing newline the
            # template adds after it is input only
            stop = start + len(ids) + 1
            labels[i, start:stop] = input_ids[i, start:stop]

        batch["labels"] = labels
        return batch
