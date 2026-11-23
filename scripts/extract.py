"""Emotion vectors or a direction: the steps of the config's extraction method (`emotion_neurons.extraction`).

    python scripts/extract.py --config experiments/00-vector-extraction/stage2_l25.yaml       # GPU for capture
    python scripts/extract.py --config experiments/00-vector-extraction/stage2_l25.yaml --steps build,check --device cpu
    python scripts/extract.py --config experiments/00-vector-extraction/refusal_qwen1.yaml    # GPU

- emotion_vectors: capture, build, check. direction: capture, build, select, check.
"""

import sys

from emotion_neurons.config import ExtractionRunConfig, load_run_config
from emotion_neurons.extraction import run_extraction

if __name__ == "__main__":
    run_extraction(load_run_config(ExtractionRunConfig, sys.argv[1:]))
