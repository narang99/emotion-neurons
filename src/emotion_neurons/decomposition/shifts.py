"""Each neuron's activation shift per emotion, and how it splits the MLP's write.

- dA_e[n]: neuron n's mean activation on emotion e's stories minus its mean over the 12 emotions.
- The write splits exactly over neurons: `delta_e = W_down @ dA_e = sum_n u_n`, with `u_n = W_down[:, n] * dA_e[n]`.
- Keeping a set S holds every other neuron at its 12-emotion mean, so the write becomes `sum_{n in S} u_n`.
"""

from pathlib import Path

import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from ..capture import late_means
from ..emotions import EMOTIONS
from ..texts import StorySplit, load_story_records

Shift = Float[Tensor, "emotion d_mlp"]


class ActivationShifts(BaseModel):
    """dA on different story subsets, from the same pooled sums.

    - train: the stories the vectors are built from;
    - half_a, half_b: a random half of each emotion's train stories, and the rest (for cross-fitting);
    - heldout: the held-out stories (for transfer);
    - std_error: standard error of each emotion's train mean (the grand mean's own error is about 1/12 of it).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    train: Shift
    half_a: Shift
    half_b: Shift
    heldout: Shift
    std_error: Shift


def shift_on(x: Float[Tensor, "story d_mlp"], split: StorySplit, among: Bool[Tensor, " story"]) -> Shift:
    mu = torch.stack([x[among & (split.label == e)].mean(0) for e in range(len(EMOTIONS))])
    return mu - mu.mean(0)


def random_halves(split: StorySplit, seed: int) -> Bool[Tensor, " story"]:
    """True for the second half of each emotion's train stories, drawn per emotion."""
    second = torch.zeros(len(split.label), dtype=torch.bool)
    generator = torch.Generator().manual_seed(seed)
    for e in range(len(EMOTIONS)):
        idx = (split.train & (split.label == e)).nonzero().squeeze(1)
        second[idx[torch.randperm(len(idx), generator=generator)[len(idx) // 2 :]]] = True
    return second


def shifts_from_means(x: Float[Tensor, "story d_mlp"], split: StorySplit, seed: int) -> ActivationShifts:
    second = random_halves(split, seed)
    train = split.train

    def std_error(e: int) -> Tensor:
        stories = train & (split.label == e)
        return x[stories].std(0) / stories.sum() ** 0.5

    return ActivationShifts(
        train=shift_on(x, split, train),
        half_a=shift_on(x, split, train & ~second),
        half_b=shift_on(x, split, train & second),
        heldout=shift_on(x, split, split.heldout),
        std_error=torch.stack([std_error(e) for e in range(len(EMOTIONS))]),
    )


def activation_shifts(run_dir: Path, seed: int) -> ActivationShifts:
    """From a single-layer run's pooled sums."""
    split = StorySplit.of(load_story_records(run_dir))
    return shifts_from_means(late_means(run_dir)["mlp_act"].float(), split, seed)


def neuron_terms(W_down: Float[Tensor, "d_model d_mlp"], shift: Float[Tensor, " d_mlp"]) -> Float[Tensor, "d_model d_mlp"]:
    """Column n is u_n, neuron n's share of the write."""
    return W_down * shift[None, :]
