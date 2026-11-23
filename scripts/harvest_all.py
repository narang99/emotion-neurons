"""Every neuron of one MLP: inputs once, then poles, per-band hits and modes per chunk, into a store. GPU.

    python scripts/harvest_all.py --config configs/harvest_all_qwen1_l17.yaml

- See `emotion_neurons.harvest_all`. Judges and reports read runs/<run_id>/store/ without the model.
"""

import sys

import torch

from emotion_neurons.config import ModesRunConfig, load_run_config
from emotion_neurons.harvest_all import run_harvest_all

torch.set_grad_enabled(False)

if __name__ == "__main__":
    run_harvest_all(load_run_config(ModesRunConfig, sys.argv[1:]))
