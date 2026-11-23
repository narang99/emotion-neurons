"""Extraction runs: a data source and a method, the method's steps run in order.

- `ExtractionMethod` is the protocol a method implements, one method per step:
  - `EmotionVectorsMethod` (`extraction.emotion_vectors`): its `select` does nothing;
  - `DirectionMethod` (`extraction.direction`).
- A new method is a config class in the `Extraction` union, a method class, and one case in `method_for`.
- Each step reads what the step before it saved, so `steps` in the config reruns any part of a run.
"""

from typing import Protocol, get_args

import torch

from ..config import DirectionExtraction, EmotionVectorsExtraction, ExtractionRunConfig, Step
from .direction import DirectionMethod
from .emotion_vectors import EmotionVectorsMethod

STEP_ORDER: tuple[Step, ...] = get_args(Step)


class ExtractionMethod(Protocol):
    def capture(self) -> None:
        """GPU pass over the examples; saves sums or means to runs/<run_id>/."""

    def build(self) -> None:
        """The vectors or candidates, from what capture saved."""

    def select(self) -> None:
        """Picks one candidate."""

    def check(self) -> None:
        """Tests the result: nearest-vector accuracy, or generation."""

    def tokens(self) -> None:
        """Per-token scores for the report's token views."""

    def report(self) -> None:
        """results/<run_id>/report.html, from the saved results only."""


def method_for(cfg: ExtractionRunConfig) -> ExtractionMethod:
    match cfg.extraction:
        case EmotionVectorsExtraction():
            return EmotionVectorsMethod(cfg)
        case DirectionExtraction():
            return DirectionMethod(cfg)


def run_extraction(cfg: ExtractionRunConfig) -> None:
    torch.set_grad_enabled(False)
    method = method_for(cfg)
    steps = {
        "capture": method.capture,
        "build": method.build,
        "select": method.select,
        "check": method.check,
        "tokens": method.tokens,
        "report": method.report,
    }
    for step in cfg.steps or STEP_ORDER:
        print(f"[step] {cfg.extraction.kind}: {step}", flush=True)
        steps[step]()


__all__ = ["STEP_ORDER", "ExtractionMethod", "method_for", "run_extraction"]
