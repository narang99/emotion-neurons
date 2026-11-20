"""Sparse sets that rebuild a write in direction and length: greedy on the leftover, not on the cosine.

- The write splits over neurons: `d = sum_n u_n`, with `u_n = shift[n] w_n`.
- For a set S, the leftover is `r = d - sum_{n in S} u_n`. It is the sum of the neurons outside S.
  - A small leftover means the neurons outside S cancel each other, as a whole vector.
- Adding j changes `|r|^2` by `-(2 r.u_j - |u_j|^2)`. Greedy adds the j with the largest gain; it stops when no j helps.
- Kept up to date without building the Gram matrix of the terms: `t = U^T s` grows by `shift * WtW[j] * shift[j]`,
  `r.u_j = d.u_j - t_j`, `|u_j|^2 = shift_j^2 |w_j|^2`.
"""

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class ResidualGreedy(BaseModel):
    """- order: the neuron added at each step;
    - leftover: |r| / |d| after each step.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    order: Int[Tensor, " step"]
    leftover: Float[Tensor, " step"]


def residual_greedy(
    W: Float[Tensor, "d_model neuron"], WtW: Float[Tensor, "neuron neuron"], shift: Float[Tensor, " neuron"], steps: int
) -> ResidualGreedy:
    d = W @ shift
    d_norm2 = float(d @ d)
    c = shift * (W.T @ d)  # d . u_n
    diag = shift**2 * (W * W).sum(0)  # |u_n|^2
    t = torch.zeros_like(shift)
    chosen = torch.zeros_like(shift, dtype=torch.bool)
    r_norm2 = d_norm2
    order, leftover = [], []
    for _ in range(steps):
        gain = (2 * (c - t) - diag).masked_fill(chosen, -float("inf"))
        j = int(gain.argmax())
        if gain[j] <= 0:
            break
        r_norm2 -= float(gain[j])
        t += shift * WtW[j] * shift[j]
        chosen[j] = True
        order.append(j)
        leftover.append(max(r_norm2, 0.0) ** 0.5 / d_norm2**0.5)
    return ResidualGreedy(order=torch.tensor(order, dtype=torch.long), leftover=torch.tensor(leftover))


def first_below(leftover: Float[Tensor, " step"], level: float) -> int | None:
    hit = leftover <= level
    return int(hit.float().argmax()) + 1 if hit.any() else None
