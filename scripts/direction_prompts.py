"""What a direction's top neurons do at the position the direction is read from: prompts at their extremes. CPU.

    python scripts/direction_prompts.py --config configs/refusal_qwen1.yaml --harvest-run harvest_refusal_qwen1

- Needs results/<run_id>/{neurons/<concept>.csv, neuron_acts.safetensors, direction.json, neurons.json} and the
  harvest run's cards.json (for lens and clusters). See `emotion_neurons.direction_page`.
"""

import argparse
import sys

from emotion_neurons.config import DirectionRunConfig, load_run_config
from emotion_neurons.direction_page import build_prompt_view

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--harvest-run", required=True)
    parser.add_argument("--top", type=int, default=40)
    known, rest = parser.parse_known_args(sys.argv[1:])
    summary = build_prompt_view(load_run_config(DirectionRunConfig, rest), known.harvest_run, known.top)
    print(
        f"{summary['separating_neurons']} of {summary['top_n']} top neurons separate held-out prompts >= 80%; "
        f"top {summary['top_n']} carry {summary['share_of_top']}% of the write"
    )
    for v in summary["neurons"][:12]:
        print(f"  #{v['rank']:>2} n{v['neuron']:<5} {v['share']:5.2f}% sep {100 * v['separation']:3.0f}% | {v['label']}")
