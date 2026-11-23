"""The emotion stories (data/stories/*.jsonl) as an example set: plain text, read from token `split_token` on.

- Same stories, order and held-out split as `texts.load_stories`; held-out stories are the "val" split.
- The existing emotion pipeline (extract.py, build_vectors.py) still reads the stories directly; this adapter lets
  generic code (e.g. `vectors.diff_means`) treat them like any other example set.
"""

from transformers import PreTrainedTokenizerBase

from ..config import DataConfig
from ..emotions import LABELS
from ..models import ModelAdapter
from ..texts import load_stories
from .base import Example, Rendered


class StoriesSet:
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.categories = list(LABELS)

    def examples(self) -> list[Example]:
        return [
            Example(id=s.id, category=s.emotion, split="val" if s.heldout else "train", text=s.text)
            for s in load_stories(self.cfg)
        ]

    def render(self, example: Example, tokenizer: PreTrainedTokenizerBase, adapter: ModelAdapter) -> Rendered:
        ids = tokenizer(example.text, add_special_tokens=False)["input_ids"]
        return Rendered(input_ids=ids, positions=list(range(min(self.cfg.split_token, len(ids)), len(ids))))
