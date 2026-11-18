"""What attention heads write into one MLP neuron, and a vocabulary read of the part the neuron uses.

- Head h of layer k writes `w_h(t) = W_O^h z_h(t)` into the stream (`z_h`: its 128 dims of the o_proj input).
- Neuron m of layer l reads one lane (gate or up) as `read . resid_mid / rms`, with `read = row * g` (g: norm gain).
- So the head's push on the lane is `z_h(t) . r_h / rms(t)`, with `r_h = (W_O^h)^T read`. This is exact.
  - The heads of layer k add up to that layer's attention part of the lane input.
- The neuron uses only the part of `z_h` along `r_h`. In the stream, that part points along `u_h = W_O^h r_h_hat`.
  - `u_h` is the direction of everything head h can write that this lane can read.
  - It does not depend on how the head's 128 dims are rotated.
- Geva's read of the used part: `U (g_final * u_h)`, the logit lens of `u_h`.
"""

from collections.abc import Callable

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from transformers import PreTrainedModel

from ..models.qwen3 import StopForward, decoder_layers, hooks, raise_stop

Lane = str


def head_count(model: PreTrainedModel) -> tuple[int, int]:
    config = model.config
    return config.num_attention_heads, config.head_dim


def lane_read(model: PreTrainedModel, layer: int, neuron: int, lane: Lane) -> Float[Tensor, " d_model"]:
    block = decoder_layers(model)[layer]
    row = (block.mlp.gate_proj if lane == "gate" else block.mlp.up_proj).weight[neuron].float()
    return row * block.post_attention_layernorm.weight.float()


def head_reads(model: PreTrainedModel, source_layer: int, read: Tensor) -> Float[Tensor, "head head_dim"]:
    """`r_h = (W_O^h)^T read` for every head of `source_layer`."""
    n_heads, head_dim = head_count(model)
    W_O = decoder_layers(model)[source_layer].self_attn.o_proj.weight.float()
    return (W_O.T @ read).view(n_heads, head_dim)


def used_directions(
    model: PreTrainedModel, source_layer: int, r: Float[Tensor, "head head_dim"]
) -> Float[Tensor, "head d_model"]:
    """`u_h = W_O^h r_h / |r_h|`: where in the stream the used part of each head points."""
    n_heads, head_dim = head_count(model)
    W_O = decoder_layers(model)[source_layer].self_attn.o_proj.weight.float().view(-1, n_heads, head_dim)
    unit = r / r.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return torch.einsum("dhk,hk->hd", W_O, unit)


class HeadReading(BaseModel):
    """One sequence, read up to a reader layer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    z: dict[int, Float[Tensor, "token d_model"]]
    """The o_proj input of each layer: the heads' outputs side by side."""
    inverse_rms: Float[Tensor, " token"]
    """1 / rms of the stream before the reader layer's MLP."""
    mlp_input: Float[Tensor, "token d_model"]
    """The reader layer's normalised MLP input, so a neuron's lane input is `row . mlp_input`."""
    activation: Float[Tensor, "token neuron"]


@torch.no_grad()
def read_heads(model: PreTrainedModel, input_ids: Int[Tensor, "1 token"], reader_layer: int, neurons: list[int]) -> HeadReading:
    blocks = decoder_layers(model)[: reader_layer + 1]
    z: dict[int, Tensor] = {}
    store: dict[str, Tensor] = {}

    def keep_z(layer: int) -> Callable:
        return lambda _m, args: z.__setitem__(layer, args[0][0].float())

    def keep_scale(module, args):
        x = args[0][0].float()
        store["inverse_rms"] = torch.rsqrt(x.pow(2).mean(-1) + module.variance_epsilon)

    handles = [block.self_attn.o_proj.register_forward_pre_hook(keep_z(k)) for k, block in enumerate(blocks)]
    handles += [
        blocks[-1].mlp.register_forward_pre_hook(lambda _m, a: store.__setitem__("mlp_input", a[0][0].float())),
        blocks[-1].post_attention_layernorm.register_forward_pre_hook(keep_scale),
        blocks[-1].mlp.down_proj.register_forward_pre_hook(
            lambda _m, a: store.__setitem__("activation", a[0][0][:, neurons].float())
        ),
        blocks[-1].register_forward_hook(raise_stop),
    ]
    with hooks(handles):
        try:
            model.model(input_ids=input_ids, use_cache=False)
        except StopForward:
            pass
    return HeadReading(z=z, **store)


def head_pushes(
    model: PreTrainedModel, z: Float[Tensor, "token d_model"], scale: Float[Tensor, " token"], source_layer: int, read: Tensor
) -> Float[Tensor, "token head"]:
    """Each head's push on the lane input at each token: `z_h . r_h / rms`."""
    n_heads, head_dim = head_count(model)
    r = head_reads(model, source_layer, read)
    return torch.einsum("thk,hk->th", z.view(-1, n_heads, head_dim), r) * scale[:, None]


@torch.no_grad()
def activation_without_heads(
    model: PreTrainedModel, input_ids: Tensor, reader_layer: int, neuron: int, removed: dict[int, list[int]], tokens: Tensor
) -> Float[Tensor, " hit"]:
    """The neuron's activation at `tokens` with the `removed` heads' o_proj inputs set to zero at those tokens."""
    n_heads, head_dim = head_count(model)
    blocks = decoder_layers(model)[: reader_layer + 1]
    out: dict[str, Tensor] = {}

    def zero_heads(heads: list[int]) -> Callable:
        def hook(_m, args):
            x = args[0].clone().view(1, -1, n_heads, head_dim)
            rows = tokens.to(x.device).nonzero().squeeze(1)
            x[0, rows[:, None], torch.tensor(heads, device=x.device)[None, :]] = 0
            return (x.view(1, -1, n_heads * head_dim),)

        return hook

    handles = [blocks[k].self_attn.o_proj.register_forward_pre_hook(zero_heads(h)) for k, h in removed.items() if h]
    handles += [
        blocks[-1].mlp.down_proj.register_forward_pre_hook(lambda _m, a: out.__setitem__("a", a[0][0, :, neuron].float())),
        blocks[-1].register_forward_hook(raise_stop),
    ]
    with hooks(handles):
        try:
            model.model(input_ids=input_ids, use_cache=False)
        except StopForward:
            pass
    return out["a"][tokens.to(out["a"].device)].cpu()
