"""Example sets for difference-of-means directions: emotion stories, benchmark instructions."""

from ..config import DataConfig, InstructionsData, StoriesData
from .base import Example, ExampleSet, Rendered, Split, examples_in
from .instructions import InstructionSet, post_instruction_length
from .stories import StoriesSet


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
    "examples_in",
    "post_instruction_length",
]
