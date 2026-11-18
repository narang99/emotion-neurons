"""The committed stories, in a fixed order, with the held-out split."""

import json
from dataclasses import dataclass
from pathlib import Path

from .config import EMOTIONS, DataConfig

LABELS = (*EMOTIONS, "neutral")
"""Row order of every per-story array: the 12 emotions in config order, then neutral."""


@dataclass(frozen=True)
class Story:
    id: str
    emotion: str
    topic_id: int
    story_idx: int
    text: str
    heldout: bool


def load_stories(cfg: DataConfig) -> list[Story]:
    """Sorted by (label order, topic, story_idx), so the order never depends on the filesystem."""
    rows = []
    for path in sorted(Path(cfg.stories_dir).glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    stories = []
    for r in rows:
        held = cfg.heldout_neutral_idx if r["emotion"] == "neutral" else cfg.heldout_story_idx
        stories.append(Story(r["id"], r["emotion"], r["topic_id"], r["story_idx"], r["text"], r["story_idx"] in held))
    stories.sort(key=lambda s: (LABELS.index(s.emotion), s.topic_id, s.story_idx))
    ids = [s.id for s in stories]
    assert len(ids) == len(set(ids)), "duplicate story ids"
    return stories
