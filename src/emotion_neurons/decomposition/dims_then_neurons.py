"""Stage 3's set: the write's main dimensions, then the neurons behind each dimension.

1. Dimensions of delta_e covering `dim_share` of its positive mass, and separately of its negative mass.
2. Per kept dimension d, the terms `W_down[d, n] * dA_e[n]`: the neurons covering `neuron_share` of the positive
   terms and, separately, of the negative terms.
3. The set is the union.

- At 0.9 / 0.9 the set is all 12288 neurons, because the write is dense.
  - So harvesting ranks neurons inside it by their net signed term over the kept dims (`rank_neurons`).
  - An earlier ranking clamped each dim's term before summing.
    - A large neuron pushing along the write in some dims and against it in others then ranked top on both poles.
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor

from ..emotions import EMOTIONS


def cover_count(sorted_values: Float[Tensor, "... n"], share: float) -> Int[Tensor, "..."]:
    """Rows sorted descending, all >= 0: how many leading entries first reach `share` of the row total."""
    total = sorted_values.cumsum(-1)
    return ((total < share * total[..., -1:]).sum(-1) + 1).clamp(max=sorted_values.shape[-1])


def dims_kept(delta: Float[Tensor, " d_model"], sign: int, share: float) -> Int[Tensor, " dim"]:
    v = (sign * delta).clamp(min=0)
    if v.sum() == 0:
        return torch.tensor([], dtype=torch.long)
    values, order = v.sort(descending=True)
    return order[: int(cover_count(values, share))]


def neurons_for(terms: Float[Tensor, "dim d_mlp"], sign: int, share: float) -> tuple[Int[Tensor, " neuron"], Int[Tensor, " dim"]]:
    """Per row, the neurons covering `share` of the `sign` terms: (their union, how many each row needed)."""
    pushing = (sign * terms).clamp(min=0)
    values, order = pushing.sort(dim=1, descending=True)
    k = cover_count(values, share)
    k = torch.where(pushing.sum(1) > 0, k, torch.zeros_like(k))
    keep = torch.arange(pushing.shape[1])[None, :] < k[:, None]
    return torch.unique(order[keep]), k


def rank_neurons(
    neuron_set: dict, W: Float[Tensor, "d_model d_mlp"], write: Float[Tensor, "emotion d_model"],
    shift: Float[Tensor, "emotion d_mlp"], per_group: int,
) -> dict[str, list[int]]:  # fmt: skip
    """Per (emotion, pole): the top neurons by net signed term over the kept dims.

    - pos: pushes the kept dims along the write; neg: against it.
    """
    groups = {}
    for e, emotion in enumerate(EMOTIONS):
        delta, dA = write[e], shift[e]
        dims = sorted({d for key in ("posdim/pos", "negdim/pos") for d in neuron_set["counts"][f"{emotion}/{key}"]["dims"]})
        dims = torch.tensor(dims, dtype=torch.long)
        aligned = W[dims] * dA[None, :] * torch.sign(delta[dims])[:, None]
        net = aligned.sum(0)
        for pole, sign in (("pos", 1), ("neg", -1)):
            score = sign * net
            top = score.argsort(descending=True)[:per_group]
            groups[f"{emotion}/{pole}"] = [int(n) for n in top if score[n] > 0]
        assert not set(groups[f"{emotion}/pos"]) & set(groups[f"{emotion}/neg"])
    return groups
