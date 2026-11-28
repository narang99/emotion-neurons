"""Instructions sampled per category, in the model's chat format (the refusal paper's setup).

- Each category lists URLs per split: JSON lists of {"instruction": ...} rows, e.g. the refusal paper's published
  splits. The URLs of a split are merged and de-duplicated, then `n_train` / `n_val` instructions are sampled with
  a fixed seed.
- Rendering: the adapter's chat prompt around the instruction. The positions read are the post-instruction tokens:
  the template that follows the user's message (Qwen-1: <|im_end|> \\n <|im_start|> assistant \\n).
  - Their count comes from the template itself (`post_instruction_length`), not a fixed number, so another model
    family (e.g. Qwen3 with its empty think block) gets its own positions.
"""

import json
import random
import urllib.request
from functools import cache

from transformers import PreTrainedTokenizerBase

from ..config import InstructionsData
from ..models import ModelAdapter
from .base import Example, Rendered

SENTINEL = "⁣SENTINEL⁣"
"""Stands in for the instruction while the chat template is split into prefix and suffix."""


@cache
def read_instructions(url: str) -> tuple[str, ...]:
    with urllib.request.urlopen(url, timeout=60) as response:
        rows = json.loads(response.read())
    texts = [row["instruction"].strip() for row in rows]
    return tuple(t for t in texts if t)


def post_instruction_length(tokenizer: PreTrainedTokenizerBase, adapter: ModelAdapter) -> int:
    """How many tokens the chat template puts after the instruction.

    Worked example for Qwen-1. The template is rendered with `SENTINEL` in place of the instruction:

        <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n⁣SENTINEL⁣<|im_end|>\n<|im_start|>assistant\n

    - The text after `SENTINEL` is `<|im_end|>\n<|im_start|>assistant\n`.
    - It tokenizes to 5 tokens: `<|im_end|>`, `\n`, `<|im_start|>`, `assistant`, `\n`. So this returns 5.
    """
    prompt = adapter.chat_prompt(tokenizer, SENTINEL)
    suffix = prompt.split(SENTINEL, 1)[1]
    return len(tokenizer(suffix, add_special_tokens=False)["input_ids"])


def merged_instructions(urls: list[str]) -> list[str]:
    """The instructions of every URL, de-duplicated and sorted, so the pool does not depend on URL order."""
    merged: set[str] = set()
    for url in urls:
        merged.update(read_instructions(url))
    return sorted(merged)


class InstructionSet:
    def __init__(self, cfg: InstructionsData):
        self.cfg = cfg
        self.categories = list(cfg.categories)

    def examples(self) -> list[Example]:
        rng = random.Random(self.cfg.seed)
        out = []
        for category, splits in self.cfg.categories.items():
            for split, urls, n in (
                ("train", splits.train, self.cfg.n_train),
                ("val", splits.val, self.cfg.n_val),
            ):
                pool = merged_instructions(urls)
                assert len(pool) >= n, f"{category}/{split}: {len(pool)} instructions, {n} wanted"
                for i, text in enumerate(rng.sample(pool, n)):
                    out.append(Example(id=f"{category}-{split}-{i:03d}", category=category, split=split, text=text))
        return out

    def render(self, example: Example, tokenizer: PreTrainedTokenizerBase, adapter: ModelAdapter) -> Rendered:
        ids = tokenizer(adapter.chat_prompt(tokenizer, example.text), add_special_tokens=False)["input_ids"]
        k = post_instruction_length(tokenizer, adapter)
        return Rendered(input_ids=ids, positions=list(range(len(ids) - k, len(ids))))
