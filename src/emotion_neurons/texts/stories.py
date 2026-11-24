"""The committed training stories.

- Training stories have a fixed order and a held-out split.
  - Every per-story array uses that order, so `stories.json` in a run records it.
"""

import json
from pathlib import Path

import torch
from jaxtyping import Bool, Int
from pydantic import BaseModel
from torch import Tensor

from ..config import StoriesData
from ..emotions import LABELS, emotion_index


class Story(BaseModel):
    id: str
    emotion: str
    topic_id: int
    story_idx: int
    text: str
    heldout: bool


class StoryRecord(BaseModel):
    """One row of a run's stories.json: a story without its text, plus its token count."""

    id: str
    emotion: str
    topic_id: int
    story_idx: int
    heldout: bool
    n_tokens: int


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _is_heldout(row: dict, cfg: StoriesData) -> bool:
    return row["story_idx"] in heldout_indices(cfg, row["emotion"])


def load_stories(cfg: StoriesData) -> list[Story]:
    """Sorted by (label order, topic, story_idx), so the order never depends on the filesystem."""
    rows = [row for path in sorted(Path(cfg.stories_dir).glob("*.jsonl")) for row in read_jsonl(path)]
    stories = [Story(**row, heldout=_is_heldout(row, cfg)) for row in rows]
    stories.sort(key=lambda s: (LABELS.index(s.emotion), s.topic_id, s.story_idx))
    ids = [s.id for s in stories]
    assert len(ids) == len(set(ids)), "duplicate story ids"
    return stories


def heldout_indices(cfg: StoriesData, emotion: str) -> list[int]:
    return cfg.heldout_neutral_idx if emotion == "neutral" else cfg.heldout_story_idx


def load_story_records(run_dir: Path) -> list[StoryRecord]:
    return [StoryRecord(**row) for row in json.loads((run_dir / "stories.json").read_text())["stories"]]


class StorySplit(BaseModel):
    """Per-story label index (-1 for neutral) and held-out flag, as tensors in run order."""

    model_config = {"arbitrary_types_allowed": True}

    label: Int[Tensor, " story"]
    heldout: Bool[Tensor, " story"]
    topic: Int[Tensor, " story"]

    @classmethod
    def of(cls, records: list[StoryRecord]) -> "StorySplit":
        return cls(
            label=torch.tensor([emotion_index(r.emotion) for r in records]),
            heldout=torch.tensor([r.heldout for r in records]),
            topic=torch.tensor([r.topic_id for r in records]),
        )

    @property
    def emotional(self) -> Bool[Tensor, " story"]:
        return self.label >= 0

    @property
    def neutral(self) -> Bool[Tensor, " story"]:
        return self.label < 0

    @property
    def train(self) -> Bool[Tensor, " story"]:
        return ~self.heldout
