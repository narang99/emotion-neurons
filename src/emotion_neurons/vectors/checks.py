"""Held-out accuracy of a site's vectors: each held-out emotional story goes to the emotion whose vector is closest in cosine.

- Story means and vectors are centred on the mean of the train emotion means, so the shared component does not decide.
- With `pcs`, both have the neutral PCs removed first (the denoised vectors).
"""

from jaxtyping import Float
from torch import Tensor

from ..texts import StorySplit
from .denoise import remove
from .emotion_vectors import cosine_matrix
from .recipes import emotion_means


def heldout_accuracy(x: Float[Tensor, "story dim"], split: StorySplit, pcs: Float[Tensor, "k dim"] | None = None) -> float:
    mu = emotion_means(x, split, split.train)
    grand = mu.mean(0)
    heldout = split.heldout & split.emotional
    stories, vectors = x[heldout] - grand, mu - grand
    if pcs is not None:
        stories, vectors = remove(stories, pcs), remove(vectors, pcs)
    predicted = cosine_matrix(stories, vectors).argmax(1)
    return (predicted == split.label[heldout]).double().mean().item()
