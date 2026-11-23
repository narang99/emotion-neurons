"""Remove a theme's neurons from an emotion's write; held-out scores of both vectors. CPU.

    python scripts/theme_removal.py --config configs/remove_dread_desperate_l25.yaml

- See `emotion_neurons.theme_removal`. Writes results/<run_id>/{vectors.safetensors, removal.json}.
"""

import json
import sys

from emotion_neurons.config import ThemeRemovalRunConfig, load_run_config
from emotion_neurons.theme_removal import run_theme_removal

if __name__ == "__main__":
    summary = run_theme_removal(load_run_config(ThemeRemovalRunConfig, sys.argv[1:]))
    print(json.dumps({k: v for k, v in summary.items() if k != "removed"}, indent=1))
    print(f"removed {len(summary['removed'])} neurons")
