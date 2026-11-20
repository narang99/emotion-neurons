"""Greedy sparse sets on raw activations, one per token, batched over tokens.

- At token t the MLP writes `sum_n a_n(t) w_n`, with `w_n` the n-th column of `W_down`. No mean is removed.
- The terms are `u_n(t) = a_n(t) w_n`. Greedy adds, at each step, the neuron that raises `cos(sum_{n in S} u_n, d)` most.
  - It is the same search as `sparse.greedy`, with `a(t)` in place of the mean shift `dA_e`.
- The Gram matrix of the terms is `WtW * a a^T`. It is never built per token:
  - `t = U^T s` is updated by `a * WtW[j] * a_j` when j is added;
  - `|u_n|^2 = a_n^2 |w_n|^2`, and `u_n . d_hat = a_n (w_n . d_hat)`.
"""

import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict
from torch import Tensor

Acts = Float[Tensor, "token neuron"]


class RawGreedy(BaseModel):
    """- order: the neuron added at each step, per token;
    - cos: the set's cosine with the target after each step;
    - full_cos: the cosine of the whole MLP output with the target.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    order: Int[Tensor, "token step"]
    cos: Float[Tensor, "token step"]
    full_cos: Float[Tensor, " token"]


def raw_greedy(
    W: Float[Tensor, "d_model neuron"],
    WtW: Float[Tensor, "neuron neuron"],
    a: Acts,
    d: Float[Tensor, "token d_model"],
    steps: int,
    allowed: Bool[Tensor, "token neuron"] | None = None,
) -> RawGreedy:
    """`d` is each token's target direction (rows need not be unit).

    - `allowed`: the candidates per token (default: all). Once a token has none left, its later steps add nothing
      (order -1, cosine unchanged).
    """
    d_hat = d / d.norm(dim=1, keepdim=True)
    c = a * (d_hat @ W)  # u_n . d_hat
    diag = a**2 * (W * W).sum(0)  # |u_n|^2
    full = a @ W.T
    full_cos = (full * d_hat).sum(1) / full.norm(dim=1)

    rows = torch.arange(a.shape[0], device=a.device)
    alpha = torch.zeros(a.shape[0], device=a.device)  # s . d_hat
    beta = torch.zeros_like(alpha)  # |s|^2
    t = torch.zeros_like(a)  # U^T s
    chosen = torch.zeros_like(a, dtype=torch.bool) if allowed is None else ~allowed
    order, cos = [], []
    for _ in range(steps):
        norm2 = beta[:, None] + 2 * t + diag
        score = (alpha[:, None] + c) / norm2.clamp_min(1e-30).sqrt()
        score = score.masked_fill(chosen | (norm2 <= 0), -2.0)
        best, j = score.max(1)
        live = (best > -2.0).to(a.dtype)
        aj = a[rows, j] * live
        beta = beta + (2 * t[rows, j] + diag[rows, j]) * live
        alpha = alpha + c[rows, j] * live
        t = t + a * WtW[j] * aj[:, None]
        chosen[rows, j] = True
        order.append(torch.where(live.bool(), j, -1))
        cos.append(alpha / beta.clamp_min(1e-30).sqrt())
    return RawGreedy(order=torch.stack(order, 1), cos=torch.stack(cos, 1), full_cos=full_cos)


def first_reaching(cos: Float[Tensor, "token step"], threshold: float) -> Int[Tensor, " token"]:
    """The smallest set size reaching `threshold`, or 0 where the search never does."""
    reached = cos >= threshold
    return torch.where(reached.any(1), reached.float().argmax(1) + 1, torch.zeros_like(reached[:, 0], dtype=torch.long))
