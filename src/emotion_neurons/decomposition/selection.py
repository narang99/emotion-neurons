"""Ways to order neurons for one emotion's write, each as a Gram matrix and alignments for `greedy`.

- `WtW = W^T W` is shared by every emotion; each Gram is `WtW * a a^T` for some activation shift a.
- plain: against the train write `d = W a`.
- cross-fit: the alignment `s . d` is taken across two halves of the train stories.
  - A neuron's own story noise then cannot raise its score.
  - The norm `|s|^2` is the mean of the halves' own norms.
    - With the full-train norm, a neuron near zero overall but of opposite sign in the halves got a tiny norm and a
      huge score, and a cross-half Gram is indefinite.
- shrunk: select against `W (a * max(0, 1 - kappa / t^2))`, with `t = a / its standard error`.
  - The target leans on shifts that reproduce across stories.
  - The set's own write is unchanged; only the selection changes.
"""

import torch
from jaxtyping import Float
from torch import Tensor

from .sparse import Alignment, Gram

Matrix = Float[Tensor, "d_model neuron"]
Vector = Float[Tensor, " neuron"]


def gram(WtW: Float[Tensor, "neuron neuron"], a: Vector) -> Gram:
    return WtW * a[:, None] * a[None, :]


def plain(W: Matrix, a: Vector) -> Alignment:
    d = W @ a
    return a * (W.T @ d) / d.norm()


def crossfit(
    WtW: Float[Tensor, "neuron neuron"], a_half: Vector, b_half: Vector, d: Float[Tensor, " d_model"]
) -> tuple[Gram, Alignment]:
    alignment = (a_half * (WtW @ b_half) + b_half * (WtW @ a_half)) / 2 / d.norm()
    G = WtW * a_half[:, None] * a_half[None, :]
    G += WtW * b_half[:, None] * b_half[None, :]
    G /= 2
    return G, alignment


def shrunk(W: Matrix, a: Vector, std_error: Vector, kappa: float) -> Alignment:
    t2 = (a / std_error) ** 2
    d_shrunk = W @ (a * (1 - kappa / t2).clamp(min=0))
    return a * (W.T @ d_shrunk) / d_shrunk.norm()


def by_alignment(c: Alignment, steps: int) -> list[int]:
    return c.argsort(descending=True)[:steps].tolist()


def random_order(n: int, steps: int, seed: int) -> list[int]:
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed))[:steps].tolist()
