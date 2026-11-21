"""Themes of meaning that club together to build an emotion's direction, from a modes run's cluster labels. CPU.

    python scripts/themes.py --config configs/harvest_<emotion>_l25.yaml

- Needs results/<run_id>/cards.json (written by the report stage) and an OpenRouter key.
- See `emotion_neurons.themes`. Writes results/<run_id>/themes.json; steps are cached in results/<run_id>/themes/.
"""

import sys

from emotion_neurons.config import ModesRunConfig, load_run_config
from emotion_neurons.themes import run_themes

if __name__ == "__main__":
    for row in run_themes(load_run_config(ModesRunConfig, sys.argv[1:])):
        print(
            f"{row['push_pct']:6.2f}% | {row['clusters']:4d} clusters, {row['neurons']:3d} neurons | {row['theme']} ({row['kind']})"
        )
