"""Signal and noise neurons along one direction, from raw story-level pushes.

- Neuron n pushes the MLP output along a unit direction v by `p_n(t) = a_n(t) (w_n . v)` at token t. No mean is removed.
- A story's push is the mean over its tokens: `P_s[n] = abar_s[n] (w_n . v)`. The story is the unit of evidence,
  since tokens in one story are not independent.
- Over one emotion's train stories, each neuron has a mean push and a standard error.
  - `signal`: the mean push is not zero (|t| > t_cut).
  - `noise`: the rest. The hypothesis: their pushes cancel, so their sum is about zero.
  - Signal neurons split by the other stories (the 11 other emotions and neutral):
    - `specific`: the mean push differs from the other stories' mean push;
    - `constant`: it does not, so the neuron pushes the same way on any story (an offset).
"""

import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor

StoryPush = Float[Tensor, "story neuron"]
Mask = Bool[Tensor, " neuron"]


class Groups(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: Mask
    specific: Mask
    constant: Mask
    noise: Mask
    mean: Float[Tensor, " neuron"]
    mean_other: Float[Tensor, " neuron"]
    t: Float[Tensor, " neuron"]
    t_vs_other: Float[Tensor, " neuron"]


def mean_and_se(x: StoryPush) -> tuple[Float[Tensor, " neuron"], Float[Tensor, " neuron"]]:
    return x.mean(0), x.std(0) / len(x) ** 0.5


def classify(own: StoryPush, other: StoryPush, t_cut: float) -> Groups:
    m, se = mean_and_se(own)
    mo, seo = mean_and_se(other)
    t = m / se.clamp_min(1e-12)
    t_other = (m - mo) / (se**2 + seo**2).sqrt().clamp_min(1e-12)
    signal = t.abs() > t_cut
    specific = signal & (t_other.abs() > t_cut)
    return Groups(
        signal=signal, specific=specific, constant=signal & ~specific, noise=~signal,
        mean=m, mean_other=mo, t=t, t_vs_other=t_other,
    )  # fmt: skip


def neurons_for_mass(values: Float[Tensor, " neuron"], members: Mask, mass: float) -> int | None:
    """Fewest members, largest |value| first, whose values add up to `mass` of the sum over all neurons."""
    total = float(values.sum())
    v = values[members]
    cum = v[v.abs().argsort(descending=True)].cumsum(0)
    hit = (cum >= mass * total) if total > 0 else (cum <= mass * total)
    return int(hit.float().argmax()) + 1 if hit.any() else None


def noise_spread(train: StoryPush, test: StoryPush, noise: Mask) -> dict[str, float]:
    """The noise sum's mean and spread over stories, against independent noise (sqrt of the summed variances)."""
    predicted = float(train[:, noise].var(0).sum().sqrt())
    return {
        "mean_train": float(train[:, noise].sum(1).mean()),
        "mean_test": float(test[:, noise].sum(1).mean()),
        "sd_train": float(train[:, noise].sum(1).std()),
        "sd_if_independent": predicted,
    }
