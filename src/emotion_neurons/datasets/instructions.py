"""Instructions sampled per category from benchmark sources, in the model's chat format (the refusal paper's setup).

- Each category lists sources per split. A source is a JSON list at a URL (e.g. the refusal paper's published
  splits) or a Hub dataset and field (e.g. walledai/MaliciousInstruct). Sources of a split are merged and
  de-duplicated, then `n_train` / `n_val` instructions are sampled with a fixed seed.
- Rendering: the adapter's chat prompt around the instruction. The positions read are the post-instruction tokens:
  the template that follows the user's message (Qwen-1: <|im_end|> \\n <|im_start|> assistant \\n).
  - Their count comes from the template itself (`post_instruction_length`), not a fixed number, so another model
    family (e.g. Qwen3 with its empty think block) gets its own positions.
"""

import json
import random
import urllib.request
from functools import cache

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase

from ..config import InstructionSetConfig, InstructionSource
from ..models import ModelAdapter
from .base import Example, Rendered

SENTINEL = "⁣SENTINEL⁣"
"""Stands in for the instruction while the chat template is split into prefix and suffix."""


@cache
def read_source(source: InstructionSource) -> tuple[str, ...]:
    if source.url:
        with urllib.request.urlopen(source.url, timeout=60) as response:
            rows = json.loads(response.read())
    else:
        rows = load_dataset(source.path, source.config, split=source.split)
    texts = [row[source.field] for row in rows if not source.require_empty or not row.get(source.require_empty)]
    return tuple(t.strip() for t in texts if isinstance(t, str) and t.strip())


def post_instruction_length(tokenizer: PreTrainedTokenizerBase, adapter: ModelAdapter) -> int:
    """How many tokens the chat template puts after the instruction."""
    prompt = adapter.chat_prompt(tokenizer, SENTINEL)
    suffix = prompt.split(SENTINEL, 1)[1]
    return len(tokenizer(suffix, add_special_tokens=False)["input_ids"])


class InstructionSet:
    def __init__(self, cfg: InstructionSetConfig):
        self.cfg = cfg
        self.categories = list(cfg.categories)

    def examples(self) -> list[Example]:
        rng = random.Random(self.cfg.seed)
        out = []
        for category, splits in self.cfg.categories.items():
            for split, n in (("train", self.cfg.n_train), ("val", self.cfg.n_val)):
                pool = sorted({t for source in getattr(splits, split) for t in read_source(source)})
                assert len(pool) >= n, f"{category}/{split}: {len(pool)} instructions, {n} wanted"
                for i, text in enumerate(rng.sample(pool, n)):
                    out.append(Example(id=f"{category}-{split}-{i:03d}", category=category, split=split, text=text))
        return out

    def render(self, example: Example, tokenizer: PreTrainedTokenizerBase, adapter: ModelAdapter) -> Rendered:
        ids = tokenizer(adapter.chat_prompt(tokenizer, example.text), add_special_tokens=False)["input_ids"]
        k = post_instruction_length(tokenizer, adapter)
        return Rendered(input_ids=ids, positions=list(range(len(ids) - k, len(ids))))
