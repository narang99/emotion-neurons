"""The modes report: one page with a card per harvested neuron. CPU.

    python scripts/modes_report.py --config configs/harvest_<emotion>_l25.yaml

- See `emotion_neurons.modes_report.run_report`. The end-to-end run with checkpoints is `scripts/pipeline.py`.
"""

import sys

import torch

from emotion_neurons.config import ModesRunConfig, load_run_config
from emotion_neurons.modes_report import run_report

torch.set_grad_enabled(False)

if __name__ == "__main__":
    run_report(load_run_config(ModesRunConfig, sys.argv[1:]))
