"""Page of per-token cosines with an emotion vector and its theme-removed version, side by side. CPU.

    uv run --group viz python scripts/token_report.py --config configs/remove_dread_desperate_l25.yaml

- Needs results/<run_id>/{token_scores.json, removal.json}. See `emotion_neurons.token_page`.
- Writes results/<run_id>/token_report.html, and token_rankings.html when token_scores_all.json exists
  (two rankings over all stories: biggest single-token drop, total change).
"""

import sys

from emotion_neurons.config import ThemeRemovalRunConfig, load_run_config
from emotion_neurons.token_page import build_page, build_ranked_page

if __name__ == "__main__":
    cfg = load_run_config(ThemeRemovalRunConfig, sys.argv[1:])
    pages = {"token_report.html": build_page(cfg.result_dir)}
    if (cfg.result_dir / "token_scores_all.json").exists():
        pages["token_rankings.html"] = build_ranked_page(cfg.result_dir)
    for name, page in pages.items():
        (cfg.result_dir / name).write_text(page)
        print(f"[saved] {cfg.result_dir}/{name} ({len(page) / 1e6:.1f} MB)")
