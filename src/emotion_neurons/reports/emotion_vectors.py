"""Report data for one run of emotion vectors: read only from its saved token scores (`token_scores`).

- One section per vector, with only that vector's cosines:
  - a heatmap: per story label (12 emotions and neutral), p50, p75, p90 and max of the token cosines from
    `split_token` on, over that label's held-out stories. Diverging colours, zero gray, as on the cards;
  - the top-k held-out stories by their mean token cosine from `split_token` on.
"""

import torch
from pydantic import BaseModel

from ..config import ExtractionRunConfig
from ..emotions import LABELS
from ..token_scores import ScoreIndex, Scores, StoryScores, load_token_scores
from .heatmap import Heatmap, heatmap
from .token_views import TokenView, scores_of, vector_view

QUANTILES = {"p50": 0.5, "p75": 0.75, "p90": 0.9, "max": 1.0}


class VectorSection(BaseModel):
    emotion: str
    quantiles: Heatmap
    views: list[TokenView]


class EmotionVectorsReport(BaseModel):
    title: str
    sections: list[VectorSection]


def late_tokens_by_label(index: ScoreIndex, scores: Scores, split_token: int) -> dict[str, Scores]:
    """Token scores from `split_token` on, of every held-out story of each label, in label order."""
    by_label: dict[str, list[Scores]] = {label: [] for label in LABELS}
    for t in index.texts:
        by_label[t.label].append(scores_of(t, scores)[split_token:])
    return {label: torch.cat(rows) for label, rows in by_label.items()}


def quantile_row(late: dict[str, Scores], e: int, q: float) -> list[float]:
    """One quantile of probe `e`'s token scores, for each story label."""
    return [float(scores[:, e].quantile(q)) for scores in late.values()]


def quantile_heatmap(late: dict[str, Scores], e: int) -> Heatmap:
    grid = [quantile_row(late, e, q) for q in QUANTILES.values()]
    return heatmap("", grid, list(QUANTILES), list(late), scale="diverging", fmt=".3f")


def story_means(index: ScoreIndex, scores: Scores, split_token: int) -> StoryScores:
    return torch.stack([scores_of(t, scores)[split_token:].mean(0) for t in index.texts])


def vector_section(
    index: ScoreIndex, scores: Scores, late: dict[str, Scores], means: StoryScores, e: int, top_k: int
) -> VectorSection:
    top = means[:, e].topk(top_k).indices.tolist()
    texts = [index.texts[i] for i in top]
    span = max(float(scores_of(t, scores)[:, e].abs().max()) for t in texts)
    views = [vector_view(t, scores_of(t, scores)[:, e], float(means[i, e]), span) for t, i in zip(texts, top)]
    return VectorSection(emotion=index.probes[e], quantiles=quantile_heatmap(late, e), views=views)


def build_emotion_vectors_report(cfg: ExtractionRunConfig) -> EmotionVectorsReport:
    saved = load_token_scores(cfg.result_dir)
    assert saved is not None, "no token scores: run the tokens step first"
    index, scores = saved
    split_token = cfg.stories.split_token
    means = story_means(index, scores, split_token)
    late = late_tokens_by_label(index, scores, split_token)
    top_k = cfg.emotion_vectors.tokens.top_k
    return EmotionVectorsReport(
        title=f"Emotion vectors: {cfg.model.name}, layer {cfg.emotion_vectors.extract.layer}",
        sections=[vector_section(index, scores, late, means, e, top_k) for e in range(len(index.probes))],
    )
