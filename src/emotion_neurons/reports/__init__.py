"""HTML reports of saved results: a builder reads the results into typed data, a Jinja template renders it.

- Each extraction method's `report` step calls its builder and template (`emotion_neurons.extraction`).
"""

from .direction import build_direction_report
from .emotion_vectors import build_emotion_vectors_report
from .render import render, write_report

__all__ = ["build_direction_report", "build_emotion_vectors_report", "render", "write_report"]
