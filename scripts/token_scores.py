"""Per-token scores of held-out stories or val prompts against a run's saved vectors, for the report. GPU.

    python scripts/token_scores.py emotion_vectors --config <vectors config>.yaml
    python scripts/token_scores.py direction --config <direction config>.yaml

- Writes results/<run_id>/token_scores.{json,safetensors}. See `emotion_neurons.token_scores`.
"""

import sys
import time

import torch

from emotion_neurons.config import DirectionRunConfig, VectorsRunConfig, load_run_config
from emotion_neurons.token_scores import direction_token_scores, save_token_scores, story_token_scores

torch.set_grad_enabled(False)

KINDS = {
    "emotion_vectors": (VectorsRunConfig, story_token_scores),
    "direction": (DirectionRunConfig, direction_token_scores),
}


def main():
    kind, argv = sys.argv[1], sys.argv[2:]
    assert kind in KINDS, f"unknown kind {kind!r}; one of {sorted(KINDS)}"
    config_class, score = KINDS[kind]
    cfg = load_run_config(config_class, argv)
    t0 = time.time()
    scores, values = score(cfg)
    save_token_scores(cfg.result_dir, scores, values)
    print(
        f"[saved] {cfg.result_dir}/token_scores.*: {len(scores.texts)} texts, {values.shape[0]} tokens, {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
