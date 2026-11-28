"""Greedy set search: add the neuron that rebuilds the most of the target that is still missing.

- Each neuron has a part `u_n` in d_model. The target is a vector `d`, e.g. a direction.
- For a set S, the missing part is `m = d - sum_{n in S} u_n`.
- Rebuilt share: `1 - |m|^2 / |d|^2`. 0 for the empty set.
- Adding j changes `|m|^2` by `-(2 m.u_j - |u_j|^2)`. Greedy adds the j with the largest gain, and stops when no j helps.
- `m` is kept explicitly, so each step is one product `U^T m`.
  - The archive version kept a neuron-by-neuron Gram matrix instead. For 18 MLPs (198k neurons) that needs 300 GB.
"""

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class GreedyRun(BaseModel):
    """- order: the neuron added at each step;
    - rebuilt: the share of the target rebuilt after each step.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    order: Int[Tensor, " step"]
    rebuilt: Float[Tensor, " step"]


def greedy_set(parts: Float[Tensor, "d_model neuron"], target: Float[Tensor, " d_model"], steps: int) -> GreedyRun:
    norms2 = (parts * parts).sum(0)
    missing = target.clone()
    target_norm2 = float(target @ target)
    chosen = torch.zeros(parts.shape[1], dtype=torch.bool, device=parts.device)
    order, rebuilt = [], []
    for _ in range(steps):
        gain = (2 * (parts.T @ missing) - norms2).masked_fill(chosen, -float("inf"))
        j = int(gain.argmax())
        if gain[j] <= 0:
            break
        missing -= parts[:, j]
        chosen[j] = True
        order.append(j)
        rebuilt.append(1 - float(missing @ missing) / target_norm2)
    return GreedyRun(order=torch.tensor(order, dtype=torch.long), rebuilt=torch.tensor(rebuilt, dtype=torch.float64))


def first_reaching(rebuilt: Float[Tensor, " step"], share: float) -> int | None:
    """The set size where the rebuilt share first reaches `share`."""
    hit = rebuilt >= share
    return int(hit.float().argmax()) + 1 if hit.any() else None
