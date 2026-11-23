"""Per-token scores of held-out stories or val prompts against a run's saved vectors, for the report. GPU.

    python scripts/token_scores.py --config <extraction config>.yaml

- Writes results/<run_id>/token_scores.{json,safetensors}. See `emotion_neurons.token_scores`.
"""

import sys
import time

import torch

from emotion_neurons.config import ExtractionRunConfig, load_run_config
from emotion_neurons.token_scores import TOKEN_SCORES, save_token_scores

torch.set_grad_enabled(False)


def main():
    cfg = load_run_config(ExtractionRunConfig, sys.argv[1:])
    t0 = time.time()
    scores, values = TOKEN_SCORES[cfg.extraction.kind](cfg)
    save_token_scores(cfg.result_dir, scores, values)
    print(
        f"[saved] {cfg.result_dir}/token_scores.*: {len(scores.texts)} texts, {values.shape[0]} tokens, {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
