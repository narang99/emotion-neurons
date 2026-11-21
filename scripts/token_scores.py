"""Per-token cosines of held-out stories with an emotion vector and its theme-removed version. GPU.

    python scripts/token_scores.py --config configs/remove_dread_desperate_l25.yaml

- Needs results/<run_id>/vectors.safetensors (scripts/theme_removal.py).
- See `emotion_neurons.theme_removal.story_token_scores`. Writes results/<run_id>/token_scores.json.
"""

import json
import sys
import time

import torch
from safetensors.torch import load_file

from emotion_neurons.config import ThemeRemovalRunConfig, load_run_config
from emotion_neurons.models import load_model
from emotion_neurons.theme_removal import heldout_texts, story_token_scores

torch.set_grad_enabled(False)

if __name__ == "__main__":
    cfg = load_run_config(ThemeRemovalRunConfig, sys.argv[1:])
    t0 = time.time()
    vectors = load_file(str(cfg.result_dir / "vectors.safetensors"))
    texts, labels, ids = heldout_texts(cfg)
    model, tokenizer = load_model(cfg.model, cfg.device)
    rows = story_token_scores(model, tokenizer, cfg, vectors, texts)
    stories = [{"id": i, "label": label, **row} for i, label, row in zip(ids, labels, rows)]
    (cfg.result_dir / "token_scores.json").write_text(json.dumps({"emotion": cfg.emotion, "stories": stories}))
    print(f"[saved] {cfg.result_dir}/token_scores.json: {len(stories)} stories, {sum(len(s['tokens']) for s in stories)} tokens")
    print(f"[total] {time.time() - t0:.0f}s")
