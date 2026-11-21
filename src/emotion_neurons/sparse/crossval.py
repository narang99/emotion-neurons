"""Where to stop a set: held-out early stopping over 6 folds.

- Folds: each emotion has stories with story_idx 0-5 (20 per index, one per topic). Fold k holds out story_idx k,
  and neutral stories 2k and 2k+1 (the folds of tag `exp/crossval`; the stage-2 split is fold 5).
- Held-out error of a set S, picked on the train folds:
  `|d_held - sum_{n in S} a_train[n] w_n| / |d_held|`.
  - The set predicts the held-out write with the shifts measured on the train folds.
  - A neuron whose train shift is noise makes the prediction worse, so the error has a minimum along the set size.
  - With held-out shifts instead, the full set would rebuild d_held exactly, and the error would only fall.
- The stop is the set size (greedy) or the lam (relaxed) where the error, averaged over folds, is lowest.
"""

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from ..texts import StorySplit

W_Down = Float[Tensor, "d_model d_mlp"]


def fold_split(base: StorySplit, story_idx: Int[Tensor, " story"], k: int) -> StorySplit:
    held = torch.where(base.emotional, story_idx == k, (story_idx == 2 * k) | (story_idx == 2 * k + 1))
    return StorySplit(label=base.label, heldout=held, topic=base.topic)


def no_heldout(base: StorySplit) -> StorySplit:
    """Every story is train: for the final set, fitted on all stories."""
    return StorySplit(label=base.label, heldout=torch.zeros_like(base.heldout), topic=base.topic)


def set_error(
    W: W_Down, shift: Float[Tensor, " d_mlp"], members: Bool[Tensor, " d_mlp"], target: Float[Tensor, " d_model"]
) -> float:
    return float((target - W[:, members] @ shift[members]).norm() / target.norm())


def prefix_errors(
    W: W_Down, shift: Float[Tensor, " d_mlp"], order: Int[Tensor, " step"], target: Float[Tensor, " d_model"]
) -> Float[Tensor, " step"]:
    """set_error of every prefix of a greedy order, from the running sum of its terms."""
    built = torch.cumsum(W[:, order] * shift[order], dim=1)
    return (target[:, None] - built).norm(dim=0) / target.norm()
