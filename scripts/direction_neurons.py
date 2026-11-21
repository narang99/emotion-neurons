"""The neuron set behind a selected direction, in the greedy-set format the modes pipeline reads. GPU.

    python scripts/direction_neurons.py --config configs/refusal_qwen1.yaml

- Needs results/<run_id>/{direction.json, direction.safetensors} (scripts/direction.py).
- See `emotion_neurons.directions.run_direction_neurons`.
"""

import json
import sys

from emotion_neurons.config import DirectionRunConfig, load_run_config
from emotion_neurons.directions import run_direction_neurons

if __name__ == "__main__":
    print(json.dumps(run_direction_neurons(load_run_config(DirectionRunConfig, sys.argv[1:])), indent=1))
