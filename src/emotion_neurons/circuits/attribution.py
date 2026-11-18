"""Each neuron's effect on an emotion score, from one backward pass with attention and norms frozen.

- Emotion score of token t at layer L: `s(t) = u . resid_post_L(t)`, with u the emotion's unit vector at layer L.
- The stream is a sum, so `s(t)` splits exactly into direct shares:
  - neuron n of layer l: `a_n(t) * (W_down_l[:, n] . u)`;
  - attention of layer l: its write `. u`; the embeddings: `embed(t) . u`.
- A neuron also acts through later neurons. Its total effect is measured as `a_n(t) * ds(t)/da_n(t)`:
  - attention outputs are detached, so attention is a fixed lump and paths stay at the same token;
  - each RMSNorm's scale `1 / rms` is detached, so a norm is the linear map it applied at that token;
  - SwiGLU stays as it is, so the total is first-order through `silu(gate) * up`.
- With attention frozen, `s(t)` depends on no other token's neurons, so one backward pass of `sum_t s(t)` gives
  every token's effects at once.
"""

from collections.abc import Callable

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from transformers import PreTrainedModel

from ..models.qwen3 import StopForward, decoder_layers, hooks, raise_stop

PerLayer = dict[int, Float[Tensor, "token d_mlp"]]


class Attribution(BaseModel):
    """One sequence, every layer up to L; tensors on the CPU in fp32."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    score: Float[Tensor, " token"]
    activation: PerLayer
    total: PerLayer
    direct: PerLayer
    attention_direct: Float[Tensor, "layer token"]
    embed_direct: Float[Tensor, " token"]
    inverse_rms: Float[Tensor, "layer token"]
    """1 / rms of each layer's stream before its MLP, for edges into that layer's neurons."""
    mlp_input: dict[int, Float[Tensor, "token d_model"]]
    """The normalised MLP input, so a neuron's gate and up inputs are `w . mlp_input`."""


def frozen_rms_norm(scales: list, outputs: list) -> Callable:
    """Recomputes the norm with its scale detached; the value is unchanged."""

    def hook(module: nn.Module, args: tuple, _out: Tensor) -> Tensor:
        x = args[0].float()
        inverse_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + module.variance_epsilon).detach()
        normalised = module.weight * (x * inverse_rms).to(args[0].dtype)
        scales.append(inverse_rms[..., 0])
        outputs.append(normalised.detach())
        return normalised

    return hook


def detach_attention(store: list) -> Callable:
    def hook(_module: nn.Module, _args: tuple, out: tuple) -> tuple:
        store.append(out[0].detach())
        return (out[0].detach(), *out[1:])

    return hook


def keep_activation(store: list) -> Callable:
    def hook(_module: nn.Module, args: tuple) -> None:
        args[0].retain_grad()
        store.append(args[0])

    return hook


def attribute(
    model: PreTrainedModel, input_ids: Int[Tensor, "1 token"], layer: int, unit: Float[Tensor, " d_model"]
) -> Attribution:
    """Effects on the emotion score at every token of one sequence (batch of one, no padding)."""
    layers = decoder_layers(model)[: layer + 1]
    acts, attention, inverse_rms, mlp_input = [], [], [], []
    out: dict[str, Tensor] = {}
    handles = []
    for block in layers:
        handles += [
            block.self_attn.register_forward_hook(detach_attention(attention)),
            block.post_attention_layernorm.register_forward_hook(frozen_rms_norm(inverse_rms, mlp_input)),
            block.mlp.down_proj.register_forward_pre_hook(keep_activation(acts)),
        ]
    handles += [
        layers[-1].register_forward_hook(lambda _m, _a, o: out.__setitem__("resid_post", o[0] if isinstance(o, tuple) else o)),
        layers[-1].register_forward_hook(raise_stop),
    ]
    with torch.enable_grad(), hooks(handles):
        embed = model.model.embed_tokens(input_ids).detach().requires_grad_(True)
        try:
            model.model(inputs_embeds=embed, use_cache=False)
        except StopForward:
            pass
        u = unit.to(out["resid_post"].device, torch.float32)
        score = out["resid_post"][0].float() @ u
        score.sum().backward()
    return collect(layers, acts, attention, inverse_rms, mlp_input, embed, u, score)


def collect(layers, acts, attention, inverse_rms, mlp_input, embed, u, score) -> Attribution:
    def along_unit(x: Tensor) -> Tensor:
        return (x[0].float() @ u).detach().cpu()

    activation, total, direct = {}, {}, {}
    for i, (block, a) in enumerate(zip(layers, acts)):
        value = (block.mlp.down_proj.weight.float().T @ u).detach()
        activation[i] = a[0].detach().float().cpu()
        total[i] = (a[0] * a.grad[0]).detach().float().cpu()
        direct[i] = (a[0].detach().float() * value).cpu()
    return Attribution(
        score=score.detach().cpu(),
        activation=activation,
        total=total,
        direct=direct,
        attention_direct=torch.stack([along_unit(x) for x in attention]),
        embed_direct=along_unit(embed),
        inverse_rms=torch.stack([r[0].float().cpu() for r in inverse_rms]),
        mlp_input={i: x[0].float().cpu() for i, x in enumerate(mlp_input)},
    )


def direct_sum(attribution: Attribution) -> Float[Tensor, " token"]:
    """Equals the score up to bf16 rounding: the split is exact."""
    neurons = sum(d.sum(1) for d in attribution.direct.values())
    return neurons + attribution.attention_direct.sum(0) + attribution.embed_direct
