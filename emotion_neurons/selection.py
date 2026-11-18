"""Ranking the layer-L neurons behind the MLP's emotion write, from Stage 3's kept dimensions.

At dim_share = neuron_share = 0.9 the Stage 3 set is all 12288 neurons (the write is dense: ~1300 dims per pole,
~3000 neurons per dim), so the shortlist comes from a ranking within it:

score_e(n) = sum over kept dims d of sign(delta_e[d]) * W_down[d, n] * dA_e[n]

pos: the highest net score (pushes the kept dims along the emotion's write); neg: the lowest (pushes against it).
The sum is net: an earlier version clamped each dim's term before summing, so a large neuron pushing along the
write in some dims and against it in others ranked top on both poles (277 of 300 pos/neg slots shared).
"""

import torch

from .config import EMOTIONS


def rank_neurons(ns: dict, W: torch.Tensor, vec: dict[str, torch.Tensor], per_group: int) -> dict[str, list[int]]:
    groups = {}
    for e, emo in enumerate(EMOTIONS):
        delta, dA = vec["mlp_out/mean/raw"][e], vec["mlp_act/mean/raw"][e]
        dims = sorted({d for key in ("posdim/pos", "negdim/pos") for d in ns["counts"][f"{emo}/{key}"]["dims"]})
        dims = torch.tensor(dims, dtype=torch.long)
        aligned = W[dims] * dA[None, :] * torch.sign(delta[dims])[:, None]  # [n_dims, d_mlp], + = along the write
        net = aligned.sum(0)
        for pole, sign in (("pos", 1), ("neg", -1)):
            score = sign * net
            top = score.argsort(descending=True)[:per_group]
            groups[f"{emo}/{pole}"] = [int(n) for n in top if score[n] > 0]
        assert not set(groups[f"{emo}/pos"]) & set(groups[f"{emo}/neg"])
    return groups
