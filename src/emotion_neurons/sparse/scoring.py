"""Scores for a 0/1 set S of neurons against one emotion's gap write, on the stories it was picked on.

- leftover: `|d - sum_S u_n| / |d|`, the sum of the neurons outside S, relative to d.
"""

import torch
from jaxtyping import Bool, Float
from torch import Tensor

W_Down = Float[Tensor, "d_model d_mlp"]
Members = Bool[Tensor, " d_mlp"]


def set_leftover(W: W_Down, shift: Float[Tensor, " d_mlp"], members: Members) -> float:
    d = W @ shift
    return float((W[:, ~members] @ shift[~members]).norm() / d.norm())


def random_members(n: int, k: int, generator: torch.Generator) -> Members:
    members = torch.zeros(n, dtype=torch.bool)
    members[torch.randperm(n, generator=generator)[:k]] = True
    return members
