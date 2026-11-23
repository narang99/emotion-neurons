"""HTML reports of saved results: a builder reads the results into typed data, a Jinja template renders it.

- `REPORTS` maps an extraction kind to its builder and template. `scripts/report.py` runs the config's one.
- A new report is a builder, a template and one entry here.
"""

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel

from ..config import ExtractionRunConfig
from .direction import build_direction_report
from .emotion_vectors import build_emotion_vectors_report
from .render import render, write_report


@dataclass(frozen=True)
class ReportKind:
    build: Callable[[ExtractionRunConfig], BaseModel]
    template: str


REPORTS: dict[str, ReportKind] = {
    "emotion_vectors": ReportKind(build_emotion_vectors_report, "emotion_vectors.html"),
    "direction": ReportKind(build_direction_report, "direction.html"),
}

__all__ = ["REPORTS", "ReportKind", "render", "write_report"]
