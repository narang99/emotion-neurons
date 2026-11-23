"""Modes of each harvested neuron at both poles, by HDBSCAN on the MLP input at its hits. GPU for the inputs.

    python scripts/modes.py --config configs/harvest_<emotion>_l25.yaml

- See `emotion_neurons.modes.run_modes`. The end-to-end run with checkpoints is `scripts/pipeline.py`.
"""

import sys

import torch

from emotion_neurons.config import ModesRunConfig, load_run_config
from emotion_neurons.modes import run_modes

torch.set_grad_enabled(False)

if __name__ == "__main__":
    run_modes(load_run_config(ModesRunConfig, sys.argv[1:]))
