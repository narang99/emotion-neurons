"""Split one neuron's lane input at one token by where it came from: which input token, through which path.

- The stream before layer l's MLP is `embed + sum_k attn_k + sum_{k<l} mlp_k`, all at the same token t.
- Attention moves information between tokens. With the attention pattern A fixed, head h of layer k writes
  `W_O^h z_h(t)` with `z_h(t) = sum_s A_h(t, s) v_g(s)`, where `v_g(s) = W_V^g ln1_k(x_s)` and g is h's KV group.
- So the lane input `read . resid_mid(t) / rms(t)` (with `read = row * gain`) splits exactly into:
  - `embed`: the token's own embedding;
  - `mlp[k]`: each earlier MLP's write at t;
  - `head_source[k, h, s] = A_h(t, s) * (v_g(s) . r_h) / rms(t)`, with `r_h = (W_O^h)^T read`.
- Exact only with eager attention (it returns A). Qwen3 has no bias in v_proj and o_proj, so nothing is left over.
"""

from collections.abc import Callable

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from transformers import PreTrainedModel

from ..models.qwen3 import StopForward, decoder_layers, hooks, raise_stop
from .heads import Lane, head_count, head_reads, lane_read


class SourceSplit(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    lane_input: float
    """`read . resid_mid(t) / rms(t)`, measured."""
    embed: float
    mlp: Float[Tensor, " layer"]
    """Each earlier MLP's part (layers 0 .. l-1)."""
    head_source: Float[Tensor, "layer head source"]
    """Layers 0 .. l, heads, source tokens 0 .. t."""

    def rebuilt(self) -> float:
        return self.embed + float(self.mlp.sum()) + float(self.head_source.sum())

    def by_source(self) -> Float[Tensor, " source"]:
        """Everything that came through attention from each source token (all layers and heads)."""
        return self.head_source.sum((0, 1))


@torch.no_grad()
def split_by_source(
    model: PreTrainedModel, input_ids: Int[Tensor, "1 token"], layer: int, neuron: int, lane: Lane, t: int
) -> SourceSplit:
    """Needs `model.set_attn_implementation("eager")`, so attention returns its pattern."""
    blocks = decoder_layers(model)[: layer + 1]
    n_heads, head_dim = head_count(model)
    n_kv = model.config.num_key_value_heads
    store: dict = {"ln1": {}, "pattern": {}, "mlp": {}}

    def keep(kind: str, k: int) -> Callable:
        if kind == "pattern":
            return lambda _m, _a, out: store["pattern"].__setitem__(k, out[1][0, :, t, : t + 1].float())
        return lambda _m, _a, out: store[kind].__setitem__(k, out[0].float())  # batch 0: [token, d_model]

    handles = [blocks[0].register_forward_pre_hook(lambda _m, a: store.__setitem__("embed", a[0][0, t].float()))]
    for k, block in enumerate(blocks):
        handles += [
            block.input_layernorm.register_forward_hook(keep("ln1", k)),
            block.self_attn.register_forward_hook(keep("pattern", k)),
        ]
        if k < layer:
            handles.append(block.mlp.register_forward_hook(keep("mlp", k)))
    handles += [
        blocks[-1].post_attention_layernorm.register_forward_pre_hook(lambda _m, a: store.__setitem__("mid", a[0][0, t].float())),
        blocks[-1].post_attention_layernorm.register_forward_hook(raise_stop),
    ]
    with hooks(handles):
        try:
            model.model(input_ids=input_ids, use_cache=False)
        except StopForward:
            pass

    read = lane_read(model, layer, neuron, lane)
    eps = blocks[-1].post_attention_layernorm.variance_epsilon
    inv_rms = float(torch.rsqrt(store["mid"].pow(2).mean() + eps))
    head_source = torch.zeros(layer + 1, n_heads, t + 1)
    for k in range(layer + 1):
        r = head_reads(model, k, read)  # [head, head_dim]
        v = blocks[k].self_attn.v_proj(store["ln1"][k].to(blocks[k].self_attn.v_proj.weight.dtype)).float()[: t + 1]
        v = v.view(t + 1, n_kv, head_dim).repeat_interleave(n_heads // n_kv, dim=1)  # [source, head, head_dim]
        along = torch.einsum("shk,hk->hs", v, r)
        head_source[k] = (store["pattern"][k] * along * inv_rms).cpu()
    mlp = torch.tensor([float(store["mlp"][k][t] @ read) * inv_rms for k in range(layer)])
    return SourceSplit(
        lane_input=float(store["mid"] @ read) * inv_rms,
        embed=float(store["embed"] @ read) * inv_rms,
        mlp=mlp,
        head_source=head_source,
    )
