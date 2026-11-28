"""Example sets for difference-of-means directions: emotion stories, benchmark instructions."""

from transformers import PreTrainedTokenizerBase

from ..config import DataConfig, InstructionsData, StoriesData
from ..models import ModelAdapter
from .base import Example, ExampleSet, Rendered, Split
from .instructions import InstructionSet, post_instruction_length
from .stories import StoriesSet


def render_split(
    examples_of: ExampleSet, split: str, tokenizer: PreTrainedTokenizerBase, adapter: ModelAdapter
) -> dict[str, list[Rendered]]:
    """The rendered examples of one split, per category."""
    rendered: dict[str, list[Rendered]] = {category: [] for category in examples_of.categories}
    for example in examples_of.examples():
        if example.split == split:
            rendered[example.category].append(examples_of.render(example, tokenizer, adapter))
    return rendered


def input_ids_per_category(rendered: dict[str, list[Rendered]]) -> dict[str, list[list[int]]]:
    return {category: input_ids(rows) for category, rows in rendered.items()}


def input_ids(rendered: list[Rendered]) -> list[list[int]]:
    return [r.input_ids for r in rendered]


def example_set(cfg: DataConfig) -> ExampleSet:
    match cfg:
        case StoriesData():
            return StoriesSet(cfg)
        case InstructionsData():
            return InstructionSet(cfg)


__all__ = [
    "Example",
    "ExampleSet",
    "InstructionSet",
    "Rendered",
    "Split",
    "StoriesSet",
    "example_set",
    "input_ids_per_category",
    "post_instruction_length",
    "render_split",
]
