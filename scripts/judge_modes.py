"""Blind scene judgements and mode names for every cluster of a modes run, by cheap models on OpenRouter. CPU.

    python scripts/judge_modes.py --config configs/harvest_<emotion>_l25.yaml

- See `emotion_neurons.judge.run_judge`. The end-to-end run with checkpoints is `scripts/pipeline.py`.
"""

import sys

from emotion_neurons.config import ModesRunConfig, load_run_config
from emotion_neurons.judge import run_judge

if __name__ == "__main__":
    run_judge(load_run_config(ModesRunConfig, sys.argv[1:]))
