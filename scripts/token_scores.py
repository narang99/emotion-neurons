"""Per-token cosines of held-out stories with an emotion vector and its theme-removed version. GPU.

    python scripts/token_scores.py --config configs/remove_dread_desperate_l25.yaml

- Needs results/<run_id>/vectors.safetensors (scripts/theme_removal.py).
- `--tokens.stories all` scores every story (train and held-out) into token_scores_all.json; the default,
  held-out only, writes token_scores.json. See `emotion_neurons.theme_removal.story_token_scores`.
"""

import json
import sys
import time

import torch
from safetensors.torch import load_file

from emotion_neurons.config import ThemeRemovalRunConfig, load_run_config
from emotion_neurons.models import load_model
from emotion_neurons.theme_removal import story_texts, story_token_scores

torch.set_grad_enabled(False)

if __name__ == "__main__":
    cfg = load_run_config(ThemeRemovalRunConfig, sys.argv[1:])
    t0 = time.time()
    vectors = load_file(str(cfg.result_dir / "vectors.safetensors"))
    picked = story_texts(cfg)
    model, tokenizer = load_model(cfg.model, cfg.device)
    rows = story_token_scores(model, tokenizer, cfg, vectors, [s["text"] for s in picked])
    stories = [{k: s[k] for k in ("id", "label", "split")} | row for s, row in zip(picked, rows)]
    name = "token_scores.json" if cfg.tokens.stories == "heldout" else f"token_scores_{cfg.tokens.stories}.json"
    (cfg.result_dir / name).write_text(json.dumps({"emotion": cfg.emotion, "stories": stories}))
    print(f"[saved] {cfg.result_dir}/{name}: {len(stories)} stories, {sum(len(s['tokens']) for s in stories)} tokens")
    print(f"[total] {time.time() - t0:.0f}s")
