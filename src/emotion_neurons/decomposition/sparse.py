"""Sparse neuron sets that keep the direction of a write: greedy selection, swaps and shrinking.

- A set S writes `s = sum_{n in S} u_n`, scored by `cos(s, d)` against a target write d.
- Every move is exact and cheap, from the Gram matrix `G = U^T U` and the alignments `c_n = u_n . d_hat`.
  - `SetState` keeps `t = U^T s`, `s . d_hat` and `|s|^2` up to date under add and remove.
  - So the cosine after adding any neuron j is `(s.d_hat + c_j) / sqrt(|s|^2 + 2 t_j + G_jj)`, for all j at once.
"""

import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel
from torch import Tensor

Gram = Float[Tensor, "neuron neuron"]
Alignment = Float[Tensor, " neuron"]

EXCLUDED = -2.0
"""Score of a neuron that cannot be chosen; below any cosine."""


class SetState:
    def __init__(self, G: Gram, c: Alignment, members: list[int]):
        self.G, self.c, self.diag = G, c, G.diagonal()
        self.mask: Bool[Tensor, " neuron"] = torch.zeros(G.shape[0], dtype=torch.bool)
        self.mask[members] = True
        idx = torch.tensor(members, dtype=torch.long)
        self.t: Float[Tensor, " neuron"] = G[idx].sum(0) if members else torch.zeros(G.shape[0], dtype=G.dtype)
        self.s_dot_d = float(c[idx].sum()) if members else 0.0
        self.s_norm2 = float(self.t[idx].sum()) if members else 0.0

    @property
    def cos(self) -> float:
        return self.s_dot_d / max(self.s_norm2, 1e-30) ** 0.5

    def add(self, j: int) -> None:
        self.s_norm2 += 2 * float(self.t[j]) + float(self.diag[j])
        self.t += self.G[j]
        self.s_dot_d += float(self.c[j])
        self.mask[j] = True

    def remove(self, i: int) -> None:
        self.s_norm2 += -2 * float(self.t[i]) + float(self.diag[i])
        self.t -= self.G[i]
        self.s_dot_d -= float(self.c[i])
        self.mask[i] = False

    def members(self) -> Int[Tensor, " member"]:
        return self.mask.nonzero().squeeze(1)

    def cos_after_adding(self) -> Float[Tensor, " neuron"]:
        norm2 = self.s_norm2 + 2 * self.t + self.diag
        score = (self.s_dot_d + self.c) / norm2.clamp_min(1e-30).sqrt()
        return score.masked_fill(self.mask | (norm2 <= 0), EXCLUDED)

    def cos_after_removing(self) -> tuple[Int[Tensor, " member"], Float[Tensor, " member"]]:
        S = self.members()
        return S, (self.s_dot_d - self.c[S]) / (self.s_norm2 - 2 * self.t[S] + self.diag[S]).clamp_min(1e-30).sqrt()

    def best_swap(self) -> tuple[int, int, float]:
        """(member out, neuron in, cosine after the swap) maximising the cosine."""
        S = self.members()
        s_dot_d = self.s_dot_d - self.c[S][:, None] + self.c[None, :]
        norm2 = (self.s_norm2 - 2 * self.t[S] + self.diag[S])[:, None] + (2 * self.t + self.diag)[None, :] - 2 * self.G[S]
        score = (s_dot_d / norm2.clamp_min(1e-30).sqrt()).masked_fill_(self.mask[None, :], EXCLUDED)
        k = int(score.argmax())
        out, into = int(S[k // score.shape[1]]), k % score.shape[1]
        return out, into, float(score.view(-1)[k])


def greedy(G: Gram, c: Alignment, steps: int) -> tuple[list[int], list[float]]:
    """Forward selection: each step adds the neuron that raises the cosine most."""
    state = SetState(G, c, [])
    order, cos = [], []
    for _ in range(steps):
        j = int(state.cos_after_adding().argmax())
        state.add(j)
        order.append(j)
        cos.append(state.cos)
    return order, cos


def swap_until_stuck(state: SetState, max_swaps: int, stop_at: float | None = None) -> int:
    """Best-improvement 1-swaps at fixed size, until none raises the cosine (or it reaches `stop_at`)."""
    n = 0
    while n < max_swaps and (stop_at is None or state.cos < stop_at):
        out, into, cos = state.best_swap()
        if cos <= state.cos + 1e-7:
            break
        state.remove(out)
        state.add(into)
        n += 1
    return n


def shrink(G: Gram, c: Alignment, start: list[int], tau: float, max_swaps: int) -> list[int]:
    """Drop the cheapest member while the cosine stays above `tau`, swapping back above it when it falls under."""
    state = SetState(G, c, start)
    best = state.members().tolist()
    while True:
        S, cos = state.cos_after_removing()
        state.remove(int(S[cos.argmax()]))
        if state.cos < tau:
            swap_until_stuck(state, max_swaps, stop_at=tau)
        if state.cos < tau:
            return best
        best = state.members().tolist()


class SetMetrics(BaseModel):
    """- cos: with the target write, train stories.
    - along: share of the target's length the set carries along it.
    - rel_err: |d - s| / |d|.
    - transfer: the set's write on the held-out stories against the train target.
    """

    k: int
    cos: float
    along: float
    rel_err: float
    transfer: float


def set_metrics(
    W: Float[Tensor, "d_model neuron"], shift: Float[Tensor, " neuron"], d: Float[Tensor, " d_model"],
    heldout_shift: Float[Tensor, " neuron"], members: list[int],
) -> SetMetrics:  # fmt: skip
    idx = torch.tensor(members, dtype=torch.long)
    s, s_heldout = W[:, idx] @ shift[idx], W[:, idx] @ heldout_shift[idx]
    cos = torch.nn.functional.cosine_similarity
    return SetMetrics(
        k=len(members), cos=float(cos(s, d, dim=0)), along=float(s @ d / (d @ d)),
        rel_err=float((d - s).norm() / d.norm()), transfer=float(cos(s_heldout, d, dim=0)),
    )  # fmt: skip


def prefix_curves(
    W: Float[Tensor, "d_model neuron"], shift: Float[Tensor, " neuron"], d: Float[Tensor, " d_model"],
    heldout_shift: Float[Tensor, " neuron"], order: list[int],
) -> tuple[Float[Tensor, " k"], Float[Tensor, " k"]]:  # fmt: skip
    """Train cosine and held-out transfer of every prefix of `order`."""
    o = torch.as_tensor(order)
    cos = torch.nn.functional.cosine_similarity
    train = cos((W[:, o] * shift[o]).cumsum(1), d[:, None], dim=0)
    transfer = cos((W[:, o] * heldout_shift[o]).cumsum(1), d[:, None], dim=0)
    return train, transfer
