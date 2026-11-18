"""Checks of a site's vectors that need only the story means.

- nearest-vector labels: each emotional story goes to the emotion whose vector is closest in cosine;
- split-half cosine: vectors from topics 0-9 against topics 10-19;
- emotion x emotion cosine.
"""

import torch
from jaxtyping import Float
from torch import Tensor

from ..emotions import EMOTIONS
from ..texts import StorySplit
from .denoise import remove
from .emotion_vectors import cosine_matrix
from .recipes import emotion_means


def nearest_vector_scores(
    x: Float[Tensor, "story dim"], split: StorySplit, pcs: Float[Tensor, "k dim"] | None = None
) -> Float[Tensor, "story emotion"]:
    """Cosine of each emotional story's mean (minus the grand mean) with each emotion vector.

    - A train story is left out of its own emotion's mean.
    - With `pcs`, both sides have the neutral PCs removed first.
    - Neutral stories keep zero rows.
    """
    mu = emotion_means(x, split, split.train)
    n_train = torch.stack([(split.train & (split.label == e)).sum() for e in range(len(EMOTIONS))]).double()
    scores = torch.zeros(len(x), len(EMOTIONS), dtype=torch.float64)
    for i in torch.nonzero(split.emotional).flatten().tolist():
        m = leave_out(mu, n_train, x[i], int(split.label[i])) if split.train[i] else mu.clone()
        grand = m.mean(0)
        story, vectors = (x[i] - grand)[None], m - grand
        if pcs is not None:
            story, vectors = remove(story, pcs), remove(vectors, pcs)
        scores[i] = cosine_matrix(story, vectors)[0]
    return scores


def leave_out(mu: Tensor, n: Tensor, story: Tensor, e: int) -> Tensor:
    m = mu.clone()
    m[e] = (mu[e] * n[e] - story) / (n[e] - 1)
    return m


def confusion(predicted: Tensor, split: StorySplit) -> Tensor:
    counts = torch.zeros(len(EMOTIONS), len(EMOTIONS), dtype=torch.int64)
    for i in torch.nonzero(split.emotional).flatten().tolist():
        counts[split.label[i], predicted[i]] += 1
    return counts


def split_half_cosine(x: Float[Tensor, "story dim"], split: StorySplit) -> list[float]:
    """Train stories of topics 0-9 against 10-19, mean baseline."""
    halves = []
    for low in (0, 10):
        among = split.train & (split.topic >= low) & (split.topic < low + 10)
        mu = emotion_means(x, split, among)
        halves.append(mu - mu.mean(0))
    return torch.diagonal(cosine_matrix(halves[0], halves[1])).tolist()
