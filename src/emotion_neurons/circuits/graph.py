"""Circuits for one sequence: rank neurons by total effect, keep the top, and check by mean-ablating the rest.

- A node is a neuron (layer, index); its score is its total effect summed over the chosen tokens.
- Faithfulness of a kept set C (the usual definition):
  - `F = (s(C) - s(none)) / (s(all) - s(none))`;
  - `s(C)`: the score with every neuron outside C held at its 12-emotion mean activation, at the chosen tokens;
  - `s(none)`: every neuron of layers <= L held at its mean; `s(all)`: the model as it is.
- This forward pass is the real model: attention and norms are not frozen here.
"""

import json
from pathlib import Path

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
from transformers import PreTrainedModel

from ..emotions import EMOTIONS
from ..models.qwen3 import StopForward, decoder_layers, hooks, raise_stop
from .attribution import Attribution

Node = tuple[int, int]
Keep = dict[int, Bool[Tensor, " d_mlp"]]


def node_scores(attribution: Attribution, tokens: Bool[Tensor, " token"]) -> Float[Tensor, "layer d_mlp"]:
    """Effect against the mean activation, so a node's score predicts what mean-ablating it costs.

    - The effect against zero ranked neurons with large but constant activations first (29% of the top-1000 mass
      in layer 0), and mean-ablating those changes little: circuits of 100-1000 neurons had faithfulness -0.1.
    """
    return torch.stack([attribution.total_from_mean[layer][tokens].sum(0) for layer in sorted(attribution.total_from_mean)])


def top_nodes(scores: Float[Tensor, "layer d_mlp"], k: int) -> list[Node]:
    flat = scores.abs().flatten().topk(k).indices
    width = scores.shape[1]
    return [(int(i) // width, int(i) % width) for i in flat]


def keep_mask(nodes: list[Node], n_layers: int, d_mlp: int) -> Keep:
    keep = {layer: torch.zeros(d_mlp, dtype=torch.bool) for layer in range(n_layers)}
    for layer, n in nodes:
        keep[layer][n] = True
    return keep


def random_nodes(n_layers: int, d_mlp: int, k: int, seed: int) -> list[Node]:
    flat = torch.randperm(n_layers * d_mlp, generator=torch.Generator().manual_seed(seed))[:k]
    return [(int(i) // d_mlp, int(i) % d_mlp) for i in flat]


def mean_ablation_hook(mean: Float[Tensor, " d_mlp"], keep: Bool[Tensor, " d_mlp"], tokens: Bool[Tensor, " token"]):
    def hook(_module, args):
        a = args[0].clone()
        replace = tokens.to(a.device)[:, None] & ~keep.to(a.device)[None, :]
        a[0] = torch.where(replace, mean.to(a.device, a.dtype)[None, :], a[0])
        return (a,)

    return hook


@torch.no_grad()
def ablated_score(
    model: PreTrainedModel, input_ids: Int[Tensor, "1 token"], layer: int, unit: Float[Tensor, " d_model"],
    means: Float[Tensor, "layer d_mlp"], keep: Keep | None, tokens: Bool[Tensor, " token"],
) -> float:  # fmt: skip
    """Summed score over `tokens`; `keep=None` runs the model unchanged."""
    blocks = decoder_layers(model)[: layer + 1]
    out: dict[str, Tensor] = {}
    handles = [
        blocks[-1].register_forward_hook(lambda _m, _a, o: out.__setitem__("resid_post", o[0] if isinstance(o, tuple) else o)),
        blocks[-1].register_forward_hook(raise_stop),
    ]
    if keep is not None:
        handles += [
            b.mlp.down_proj.register_forward_pre_hook(mean_ablation_hook(means[i], keep[i], tokens)) for i, b in enumerate(blocks)
        ]
    with hooks(handles):
        try:
            model.model(input_ids=input_ids, use_cache=False)
        except StopForward:
            pass
    return float(
        (out["resid_post"][0].float() @ unit.to(out["resid_post"].device).float())[tokens.to(out["resid_post"].device)].sum()
    )


def faithfulness(full: float, none: float, kept: float) -> float:
    return (kept - none) / (full - none)


def emotion_membership(sparse_layers: Path, layers: int) -> dict[Node, list[str]]:
    """Which emotions' matching sets (from `sparse_layers.py`) each neuron is in."""
    data = json.loads(sparse_layers.read_text())["layers"]
    member: dict[Node, list[str]] = {}
    for layer in range(layers):
        for emotion in EMOTIONS:
            for n in data[str(layer)]["emotions"][emotion]["neurons"] or []:
                member.setdefault((layer, n), []).append(emotion)
    return member
