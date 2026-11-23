"""Report data for one run of emotion vectors: read only from its saved token scores (`token_scores`).

- Quantiles: per vector, p50, p75, p90 and max of the token cosines of every held-out story, from `split_token` on.
- Sections: per vector, the top-k held-out stories by their mean token cosine from `split_token` on.
"""

import torch
from pydantic import BaseModel

from ..config import ExtractionRunConfig
from ..token_scores import Scores, TokenScores, load_token_scores
from .token_views import TokenView, scores_of, vector_view

QUANTILES = {"p50": 0.5, "p75": 0.75, "p90": 0.9, "max": 1.0}


class QuantileRow(BaseModel):
    name: str
    values: list[float]


class VectorSection(BaseModel):
    emotion: str
    views: list[TokenView]


class EmotionVectorsReport(BaseModel):
    title: str
    emotions: list[str]
    quantiles: list[QuantileRow]
    sections: list[VectorSection]


def late_tokens(scores: TokenScores, values: Scores, split_token: int) -> Scores:
    return torch.cat([scores_of(t, values)[split_token:] for t in scores.texts])


def quantile_rows(late: Scores) -> list[QuantileRow]:
    return [QuantileRow(name=name, values=late.quantile(q, dim=0).tolist()) for name, q in QUANTILES.items()]


def story_means(scores: TokenScores, values: Scores, split_token: int) -> Scores:
    return torch.stack([scores_of(t, values)[split_token:].mean(0) for t in scores.texts])


def vector_section(scores: TokenScores, values: Scores, means: Scores, e: int, top_k: int) -> VectorSection:
    top = means[:, e].topk(top_k).indices.tolist()
    texts = [scores.texts[i] for i in top]
    span = max(float(scores_of(t, values)[:, e].abs().max()) for t in texts)
    views = [vector_view(t, scores_of(t, values)[:, e], float(means[i, e]), span) for t, i in zip(texts, top)]
    return VectorSection(emotion=scores.series[e], views=views)


def build_emotion_vectors_report(cfg: ExtractionRunConfig) -> EmotionVectorsReport:
    saved = load_token_scores(cfg.result_dir)
    assert saved is not None, "no token scores: run the tokens step first"
    scores, values = saved
    split_token = cfg.stories.split_token
    means = story_means(scores, values, split_token)
    top_k = cfg.emotion_vectors.tokens.top_k
    return EmotionVectorsReport(
        title=f"Emotion vectors: {cfg.model.name}, layer {cfg.emotion_vectors.extract.layer}",
        emotions=scores.series,
        quantiles=quantile_rows(late_tokens(scores, values, split_token)),
        sections=[vector_section(scores, values, means, e, top_k) for e in range(len(scores.series))],
    )
