"""Relaxed mask: a convex version of the 0/1 set search, with the same fixed coefficients.

- Each neuron keeps its measured shift `a[n]`. Only a mask `m[n]` in [0, 1] is solved for:
  `min_m 1/2 |d - sum_n m[n] a[n] w_n|^2 + lam * sum_n m[n]`.
  - `m[n] = 0`: the neuron sits at the other-story mean. `m[n] = 1`: it keeps its full shift.
  - `m[n] <= 1` stops a neuron from being scaled past what the model did.
    - Free coefficients (OMP, plain Lasso) would build d from shifts the data does not show.
  - With `m >= 0`, the L1 penalty is just `lam * sum(m)`.
- Units: the columns `a[n] w_n` and d are divided by `|d|`, so the leftover is relative and `lam` is comparable across emotions.
- Solver: FISTA (projected gradient with momentum) with adaptive restart. The projection onto [0, 1] is a clamp.
- A set is read from a mask as its support: every neuron with `m > 0`, each kept at its full shift (`m = 1`).
"""

import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor

Columns = Float[Tensor, "d_model neuron"]
Mask = Float[Tensor, " neuron"]
Target = Float[Tensor, " d_model"]


class MaskSolution(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    lam: float
    mask: Mask
    iterations: int
    stationarity: float  # max |m - prox(m)|, in mask units; 0 at the exact optimum


def scaled_columns(W: Columns, shift: Float[Tensor, " neuron"]) -> tuple[Columns, Target]:
    """Columns `a[n] w_n / |d|` and the unit target `d / |d|`."""
    d = W @ shift
    norm = d.norm()
    return W * (shift / norm)[None, :], d / norm


def lipschitz(B: Columns, iterations: int = 100) -> float:
    """Largest eigenvalue of `B^T B`, by power iteration, with a 1% margin."""
    v = torch.randn(B.shape[1], generator=torch.Generator().manual_seed(0), dtype=B.dtype)
    for _ in range(iterations):
        v = B.T @ (B @ v)
        v /= v.norm()
    return 1.01 * float(v @ (B.T @ (B @ v)))


def lambda_max(B: Columns, d: Target) -> float:
    """The smallest lam at which m = 0 is optimal: the gradient at 0 is `-B^T d`."""
    return float((B.T @ d).max())


def prox_step(B: Columns, d: Target, m: Mask, lam: float, L: float) -> Mask:
    gradient = B.T @ (B @ m - d)
    return (m - (gradient + lam) / L).clamp(0, 1)


def solve_mask(B: Columns, d: Target, lam: float, L: float, start: Mask, max_iterations: int, tolerance: float) -> MaskSolution:
    m, y, t = start.clone(), start.clone(), 1.0
    stationarity = float("inf")
    for i in range(1, max_iterations + 1):
        m_next = prox_step(B, d, y, lam, L)
        if float((y - m_next) @ (m_next - m)) > 0:  # momentum points uphill: restart it
            y, t = m.clone(), 1.0
            continue
        t_next = (1 + (1 + 4 * t * t) ** 0.5) / 2
        y = m_next + (t - 1) / t_next * (m_next - m)
        m, t = m_next, t_next
        if i % 25 == 0:
            stationarity = float((m - prox_step(B, d, m, lam, L)).abs().max())
            if stationarity < tolerance:
                break
    return MaskSolution(lam=lam, mask=m, iterations=i, stationarity=stationarity)


def leftover(B: Columns, d: Target, m: Mask) -> float:
    """|d - B m| / |d|; d is a unit vector here."""
    return float((d - B @ m).norm())


def support(m: Mask) -> Mask:
    return (m > 0).to(m.dtype)


def lambda_path(
    B: Columns, d: Target, ratios: list[float], L: float, max_iterations: int, tolerance: float
) -> list[MaskSolution]:
    """Masks for `lam = ratio * lambda_max`, large to small, each warm-started from the one before."""
    top = lambda_max(B, d)
    m = torch.zeros(B.shape[1], dtype=B.dtype)
    path = []
    for ratio in ratios:
        solution = solve_mask(B, d, ratio * top, L, m, max_iterations, tolerance)
        m = solution.mask
        path.append(solution)
    return path
