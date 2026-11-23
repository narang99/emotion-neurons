"""Extraction runs: a data source and a method, the method's steps run in order.

- `STEPS` maps a method kind (`config.Extraction`) to its named steps.
  - emotion_vectors: capture, build, check, tokens, report;
  - direction: capture, build, select, check, tokens, report.
  - tokens and report are shared (`extraction.outputs`): the token views and the HTML report.
- Each step reads what the step before it saved, so `steps` in the config reruns any part of a run.
"""

from collections.abc import Callable

import torch

from ..config import ExtractionRunConfig, Step
from . import direction, emotion_vectors, outputs

STEPS: dict[str, dict[Step, Callable[[ExtractionRunConfig], None]]] = {
    "emotion_vectors": emotion_vectors.STEPS | outputs.STEPS,
    "direction": direction.STEPS | outputs.STEPS,
}


def steps_to_run(cfg: ExtractionRunConfig) -> list[Step]:
    steps = STEPS[cfg.extraction.kind]
    unknown = [s for s in cfg.steps or [] if s not in steps]
    assert not unknown, f"{cfg.extraction.kind} has no steps {unknown}; its steps are {list(steps)}"
    return [s for s in steps if cfg.steps is None or s in cfg.steps]


def run_extraction(cfg: ExtractionRunConfig) -> None:
    torch.set_grad_enabled(False)
    for step in steps_to_run(cfg):
        print(f"[step] {cfg.extraction.kind}: {step}", flush=True)
        STEPS[cfg.extraction.kind][step](cfg)


__all__ = ["STEPS", "run_extraction", "steps_to_run"]
