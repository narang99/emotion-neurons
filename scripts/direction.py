"""A difference-of-means direction on instruction data, selected by its effect and checked by generation. GPU.

    python scripts/direction.py --config configs/refusal_qwen1.yaml

- See `emotion_neurons.directions`. Writes results/<run_id>/{direction.json, direction.safetensors, replies.json}.
"""

import sys

from emotion_neurons.config import DirectionRunConfig, load_run_config
from emotion_neurons.directions import run_direction

if __name__ == "__main__":
    summary = run_direction(load_run_config(DirectionRunConfig, sys.argv[1:]))
    print({k: summary[k] for k in ("choice", "refusal_rates", "seconds")})
