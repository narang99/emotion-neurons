"""Greedy set search: add the neuron that rebuilds the most of the write that is still missing.

- The write splits over neurons: `d = sum_n u_n`, with `u_n = a[n] w_n`.
- For a set S, the missing part is `r = d - sum_{n in S} u_n`. It is the sum of the neurons outside S.
- Rebuilt share: `1 - |r|^2 / |d|^2`. 0 for the empty set, 1 for all neurons.
- Adding j changes `|r|^2` by `-(2 r.u_j - |u_j|^2)`. Greedy adds the j with the largest gain, and stops when no j helps.
- Kept up to date without the Gram matrix of the terms: `t = U^T s` grows by `a * WtW[j] * a[j]`,
  `r.u_j = d.u_j - t_j`, and `|u_j|^2 = a_j^2 |w_j|^2`.
"""

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class GreedyRun(BaseModel):
    """- order: the neuron added at each step;
    - rebuilt: the share of d rebuilt after each step.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    order: Int[Tensor, " step"]
    rebuilt: Float[Tensor, " step"]


def greedy_set(
    W: Float[Tensor, "d_model neuron"], WtW: Float[Tensor, "neuron neuron"], shift: Float[Tensor, " neuron"], steps: int
) -> GreedyRun:
    d = W @ shift
    d_norm2 = float(d @ d)
    c = shift * (W.T @ d)
    diag = shift**2 * (W * W).sum(0)
    t = torch.zeros_like(shift)
    chosen = torch.zeros_like(shift, dtype=torch.bool)
    r_norm2 = d_norm2
    order, rebuilt = [], []
    for _ in range(steps):
        gain = (2 * (c - t) - diag).masked_fill(chosen, -float("inf"))
        j = int(gain.argmax())
        if gain[j] <= 0:
            break
        r_norm2 -= float(gain[j])
        t += shift * WtW[j] * shift[j]
        chosen[j] = True
        order.append(j)
        rebuilt.append(1 - max(r_norm2, 0.0) / d_norm2)
    return GreedyRun(order=torch.tensor(order, dtype=torch.long), rebuilt=torch.tensor(rebuilt, dtype=torch.float64))


def first_reaching(rebuilt: Float[Tensor, " step"], share: float) -> int | None:
    """The set size where the rebuilt share first reaches `share`."""
    hit = rebuilt >= share
    return int(hit.float().argmax()) + 1 if hit.any() else None
