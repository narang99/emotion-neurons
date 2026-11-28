"""Datasets as categories of examples, and how an example becomes tokens and the positions to read.

- An adapter returns `Example`s (id, category, split, text) and renders each one for a model:
  - `Rendered.input_ids`: the tokens fed to the model;
  - `Rendered.positions`: the token positions whose activations count (pooled or kept one by one downstream).
- Two adapters so far:
  - `datasets.stories.StoriesSet`: the emotion stories, from jsonl files; plain text, positions from token 50 on;
  - `datasets.instructions.InstructionSet`: instructions sampled per category from benchmark sources, wrapped in the
    model's chat format; positions are the post-instruction tokens (the template after the user's message).
"""

from typing import Literal, Protocol

from pydantic import BaseModel
from transformers import PreTrainedTokenizerBase

from ..models import ModelAdapter

Split = Literal["train", "val"]


class Example(BaseModel):
    id: str
    category: str
    split: Split
    text: str


def examples_in(examples: list[Example], split: Split, category: str) -> list[Example]:
    return [e for e in examples if e.split == split and e.category == category]


class Rendered(BaseModel):
    input_ids: list[int]
    positions: list[int]


class ExampleSet(Protocol):
    categories: list[str]

    def examples(self) -> list[Example]: ...

    def render(self, example: Example, tokenizer: PreTrainedTokenizerBase, adapter: ModelAdapter) -> Rendered: ...
