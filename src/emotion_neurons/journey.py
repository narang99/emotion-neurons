"""How each emotion vector is built across depth, from per-story means of every layer.

- The stream after layer l is a sum: `resid_post_l = embed + sum_{k <= l} (attention_k + mlp_k)`.
- The emotion contrast (an emotion's train mean minus the 12-emotion mean) is linear, so it splits the same way.
  - `v_l = c(embed) + sum_{k <= l} (c(attention_k) + c(mlp_k))`, exactly.
- Read along the vector's own unit direction at layer l, each layer's attention and MLP write is a signed share of
  `|v_l|`.
- Denoised vectors remove layer l's neutral PCs of resid_post (the emotions paper's recipe).
  - The writes need no projection for their share along a denoised unit vector.
  - That unit vector is already orthogonal to the PCs, so `P w . u = w . u`.
"""

import torch
from jaxtyping import Float
from pydantic import BaseModel
from torch import Tensor

from .capture import LayerMeans
from .config import VectorsConfig
from .emotions import EMOTIONS
from .texts import StorySplit
from .vectors import RECIPES, EmotionVectors, cosine_matrix, emotion_means, remove, unit
from .vectors.checks import nearest_vector_scores

Contrast = Float[Tensor, "emotion d_model"]


def contrast(x: Float[Tensor, "story d_model"], split: StorySplit) -> Contrast:
    mu = emotion_means(x, split, split.train)
    return mu - mu.mean(0)


class LayerReading(BaseModel):
    """One layer, one value per emotion (lists in EMOTIONS order)."""

    layer: int
    norm_raw: list[float]
    norm_denoised: list[float]
    attention_along: list[float]
    mlp_along: list[float]
    mlp_cos: list[float]
    cos_previous: list[float]
    heldout_accuracy: float


def along_share(write: Contrast, vector: Contrast) -> list[float]:
    """Each emotion's write projected on its own unit vector, as a share of that vector's length."""
    return ((write * unit(vector)).sum(1) / vector.norm(dim=1)).tolist()


def diagonal_cos(a: Contrast, b: Contrast) -> list[float]:
    return torch.diagonal(cosine_matrix(a, b)).tolist()


def layer_vectors(means: LayerMeans, layer: int, split: StorySplit, cfg: VectorsConfig) -> EmotionVectors:
    vectors, _ = RECIPES[cfg.recipe]({"resid_post": means.site(layer, "resid_post")}, split, cfg)
    return vectors


def heldout_accuracy(means: LayerMeans, layer: int, split: StorySplit, vectors: EmotionVectors) -> float:
    """12-way nearest-vector label of the held-out stories, denoised."""
    scores = nearest_vector_scores(means.site(layer, "resid_post"), split, vectors.neutral_pcs("resid_post").double())
    held = split.heldout & split.emotional
    return float((scores.argmax(1) == split.label)[held].double().mean())


def read_layers(means: LayerMeans, split: StorySplit, cfg: VectorsConfig) -> tuple[list[LayerReading], dict[int, Contrast]]:
    """Per-layer readings, and the denoised vectors at every layer (for later comparisons)."""
    readings, denoised = [], {}
    previous = None
    for layer in range(means.n_layers):
        vectors = layer_vectors(means, layer, split, cfg)
        raw, den = vectors.get("resid_post", "mean", "raw"), vectors.get("resid_post", "mean", "denoised")
        attention, mlp = contrast(means.attention_write(layer), split), contrast(means.site(layer, "mlp_out"), split)
        readings.append(
            LayerReading(
                layer=layer,
                norm_raw=raw.norm(dim=1).tolist(),
                norm_denoised=den.norm(dim=1).tolist(),
                attention_along=along_share(attention, den),
                mlp_along=along_share(mlp, den),
                mlp_cos=diagonal_cos(remove(mlp, vectors.neutral_pcs("resid_post")), den),
                cos_previous=diagonal_cos(den, previous) if previous is not None else [1.0] * len(EMOTIONS),
                heldout_accuracy=heldout_accuracy(means, layer, split, vectors),
            )
        )
        denoised[layer] = den
        previous = den
        print(f"[layer {layer:2d}] |v| {raw.norm(dim=1).mean():8.2f} | attention {sum(readings[-1].attention_along) / 12:+.3f} "
              f"mlp {sum(readings[-1].mlp_along) / 12:+.3f} of |v| | held-out acc {readings[-1].heldout_accuracy:.2f}")  # fmt: skip
    return readings, denoised


def cos_to(denoised: dict[int, Contrast], reference: int) -> dict[int, list[float]]:
    return {layer: diagonal_cos(v, denoised[reference]) for layer, v in denoised.items()}


class BuildUp(BaseModel):
    """Shares of a reference layer's denoised vector, per emotion, written by each earlier component.

    - `embed + sum(attention) + sum(mlp)` is 1 for every emotion, exactly.
    """

    reference: int
    embed: list[float]
    attention: list[list[float]]
    mlp: list[list[float]]


def build_up(means: LayerMeans, split: StorySplit, reference_vector: Contrast, reference: int) -> BuildUp:
    attention, mlp = [], []
    for layer in range(reference + 1):
        attention.append(along_share(contrast(means.attention_write(layer), split), reference_vector))
        mlp.append(along_share(contrast(means.site(layer, "mlp_out"), split), reference_vector))
    embed = along_share(contrast(means.embed(), split), reference_vector)
    return BuildUp(reference=reference, embed=embed, attention=attention, mlp=mlp)
