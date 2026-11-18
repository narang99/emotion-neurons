"""The first minimum-set search: greedy on the neuron terms directly, with no held-out stop.

- Superseded by `sparse.greedy` with a held-out stop, but it picked the top 64 harvested in `top64_l25`.
  - It works on the columns u_n instead of a Gram matrix, so it is kept as it was to reproduce that selection.
"""

import torch
from jaxtyping import Float
from torch import Tensor

THRESHOLDS = (0.9, 0.95, 0.99)


def greedy_on_terms(
    U: Float[Tensor, "d_model neuron"], d: Float[Tensor, " d_model"], steps: int
) -> tuple[list[int], list[float]]:
    """Stops early once the last threshold is reached."""
    c = U.T @ (d / d.norm())
    norm2 = (U * U).sum(0)
    s = torch.zeros_like(d)
    s_dot_d = s_norm2 = 0.0
    used = torch.zeros(U.shape[1], dtype=torch.bool)
    order, cos = [], []
    for _ in range(steps):
        along = s_dot_d + c
        score = along / (s_norm2 + 2 * (U.T @ s) + norm2).clamp_min(1e-12).sqrt()
        score[used] = -9
        n = int(score.argmax())
        used[n] = True
        s = s + U[:, n]
        s_dot_d, s_norm2 = float(along[n]), float(s @ s)
        order.append(n)
        cos.append(s_dot_d / s_norm2**0.5)
        if cos[-1] >= THRESHOLDS[-1]:
            break
    return order, cos


def first_reaching(cos: Float[Tensor, " k"]) -> list[int | None]:
    """The smallest prefix size reaching each threshold."""
    return [int((cos < t).sum()) + 1 if (cos >= t).any() else None for t in THRESHOLDS]


def prefix_cos(U: Float[Tensor, "d_model neuron"], d: Float[Tensor, " d_model"], order: Tensor) -> Float[Tensor, " k"]:
    return torch.nn.functional.cosine_similarity(U[:, order].cumsum(1), d[:, None], dim=0)


def round_robin(orders: dict[str, list[int]], total: int) -> tuple[list[int], dict[str, list[int]], int]:
    """Walk down every emotion's order in turn until `total` distinct neurons; a shared neuron counts for each."""
    selected: list[int] = []
    groups: dict[str, list[int]] = {emotion: [] for emotion in orders}
    rank = 0
    while len(selected) < total:
        for emotion, order in orders.items():
            n = order[rank]
            if n not in selected and len(selected) < total:
                selected.append(n)
            if n in selected:
                groups[emotion].append(n)
        rank += 1
    return selected, groups, rank
