"""The steps every method ends with: per-token scores for the token views, then the HTML report.

- tokens (GPU): results/<run_id>/token_scores.{json,safetensors} (`emotion_neurons.token_scores`).
- report (CPU): results/<run_id>/report.html (`emotion_neurons.reports`), from the saved results only.
"""

import time

from ..config import ExtractionRunConfig
from ..reports import REPORTS, write_report
from ..token_scores import TOKEN_SCORES, save_token_scores


def tokens(cfg: ExtractionRunConfig) -> None:
    t0 = time.time()
    scores, values = TOKEN_SCORES[cfg.extraction.kind](cfg)
    save_token_scores(cfg.result_dir, scores, values)
    print(
        f"[saved] {cfg.result_dir}/token_scores.*: {len(scores.texts)} texts, {values.shape[0]} tokens, {time.time() - t0:.0f}s"
    )


def report(cfg: ExtractionRunConfig) -> None:
    kind = REPORTS[cfg.extraction.kind]
    print(f"[saved] {write_report(kind.template, kind.build(cfg), cfg.result_dir / 'report.html')}")


STEPS = {"tokens": tokens, "report": report}
