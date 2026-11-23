"""Harvest both poles of the chosen neurons over general text. GPU.

    python scripts/harvest.py --config configs/harvest_<emotion>_l25.yaml

- See `emotion_neurons.harvest.run_harvest`. The end-to-end run with checkpoints is `scripts/pipeline.py`.
"""

import sys

import torch

from emotion_neurons.config import ModesRunConfig, load_run_config
from emotion_neurons.harvest import run_harvest

torch.set_grad_enabled(False)

if __name__ == "__main__":
    run_harvest(load_run_config(ModesRunConfig, sys.argv[1:]))
