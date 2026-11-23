"""Emotion vectors or a direction, end to end: extraction, checks, token views and the HTML report. GPU.

    python scripts/extract.py --config experiments/00-vector-extraction/stage2_l25.yaml
    python scripts/extract.py --config experiments/00-vector-extraction/refusal_qwen1.yaml
    python scripts/extract.py --config experiments/00-vector-extraction/stage2_l25.yaml --steps build,check,report --device cpu

- The config's `extraction.kind` picks the steps (`emotion_neurons.extraction`):
  - emotion_vectors: capture, build, check, tokens, report;
  - direction: capture, build, select, check, tokens, report.
- Large outputs go to runs/<run_id>/, the vectors and the report to results/<run_id>/.
"""

import sys

from emotion_neurons.config import ExtractionRunConfig, load_run_config
from emotion_neurons.extraction import run_extraction

if __name__ == "__main__":
    run_extraction(load_run_config(ExtractionRunConfig, sys.argv[1:]))
