"""Edges into a neuron: how each earlier neuron's write moves its gate and up inputs (Goodfire's input edges).

- Reader m at layer l reads `x_hat = g * resid_mid_l / rms`, through two rows: `z_gate = w_gate_m . x_hat`, same for up.
- With `1 / rms` fixed at the token, `z` is linear in the stream, so it splits exactly over every earlier write.
  - Writer j of layer k < l writes `a_j W_down_k[:, j]`, so its share is `a_j(t) * V_j / rms(t)`.
  - `V = W_down_k^T (w_m * g)` is a virtual weight: one matrix-vector product, the same for every token.
- Attention and the embeddings are the rest of `z`; they are reported as one remainder, not split.
"""

import torch
from jaxtyping import Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from transformers import PreTrainedModel

from ..models.qwen3 import decoder_layers
from .attribution import Attribution

Lane = str
""""gate" or "up"."""


class InputEdges(BaseModel):
    """Per writer layer, each writer neuron's share of the reader's gate and up input, summed over chosen tokens."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reader_layer: int
    reader: int
    shares: dict[Lane, dict[int, Float[Tensor, " d_mlp"]]]
    total: dict[Lane, float]
    """The reader's input summed over the tokens, so neurons' shares plus the remainder equal it."""


def virtual_weights(model: PreTrainedModel, reader_layer: int, reader: int, lane: Lane) -> dict[int, Float[Tensor, " d_mlp"]]:
    blocks = decoder_layers(model)
    mlp = blocks[reader_layer].mlp
    row = (mlp.gate_proj if lane == "gate" else mlp.up_proj).weight[reader].detach().float()
    read = row * blocks[reader_layer].post_attention_layernorm.weight.detach().float()
    return {k: (blocks[k].mlp.down_proj.weight.detach().float().T @ read).cpu() for k in range(reader_layer)}


def input_edges(
    model: PreTrainedModel, attribution: Attribution, reader_layer: int, reader: int, tokens: torch.Tensor
) -> InputEdges:
    """`tokens` selects the token positions to sum over (a boolean mask or indices)."""
    shares, total = {}, {}
    scale = attribution.inverse_rms[reader_layer][tokens]
    for lane in ("gate", "up"):
        weights = virtual_weights(model, reader_layer, reader, lane)
        shares[lane] = {k: (attribution.activation[k][tokens] * scale[:, None]).sum(0) * V for k, V in weights.items()}
        total[lane] = reader_input(model, attribution, reader_layer, reader, lane, tokens)
    return InputEdges(reader_layer=reader_layer, reader=reader, shares=shares, total=total)


def reader_input(model, attribution: Attribution, reader_layer: int, reader: int, lane: Lane, tokens) -> float:
    """The reader's actual gate or up input, summed over the tokens."""
    mlp = decoder_layers(model)[reader_layer].mlp
    row = (mlp.gate_proj if lane == "gate" else mlp.up_proj).weight[reader].detach().float().cpu()
    return float((attribution.mlp_input[reader_layer][tokens] @ row).sum())


def neuron_share(edges: InputEdges, lane: Lane) -> float:
    """Share of the reader's input written by earlier MLP neurons; attention and embeddings write the rest."""
    return float(sum(v.sum() for v in edges.shares[lane].values())) / edges.total[lane]
