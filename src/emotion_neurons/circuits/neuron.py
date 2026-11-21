"""The local circuit of one neuron: what writes its inputs, and which later neurons read its output.

- A neuron m of layer l reads `x_hat = g * resid_mid_l / rms` through two rows (gate, up).
- With `1 / rms` fixed at the token, each input splits exactly over the writes in the stream before it:
  - the embeddings, each earlier layer's attention write, and each earlier neuron `a_j W_down[:, j]`;
  - plus layer l's own attention write.
- A later neuron r of layer l' reads m's write the same way: its share of r's gate input is
  `a_m(t) / rms_l'(t) * (W_down_l[:, m] . (w_gate_r * g_l'))`, exactly.
- Local checks run the real model: hold the top writers at their mean, and see how far m's activation moves.
"""

from collections.abc import Callable

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from transformers import PreTrainedModel

from ..models.qwen3 import StopForward, decoder_layers, hooks, raise_stop

Lane = str


class Stream(BaseModel):
    """One sequence read up to a layer, on the GPU: per layer, what the MLP read and wrote, and attention's write."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embed: Float[Tensor, "token d_model"]
    attention: list[Float[Tensor, "token d_model"]]
    activation: list[Float[Tensor, "token d_mlp"]]
    mlp_input: list[Float[Tensor, "token d_model"]]
    inverse_rms: list[Float[Tensor, " token"]]


@torch.no_grad()
def read_stream(model: PreTrainedModel, input_ids: Int[Tensor, "1 token"], last_layer: int) -> Stream:
    blocks = decoder_layers(model)[: last_layer + 1]
    store: dict[str, list] = {"attention": [], "activation": [], "mlp_input": [], "inverse_rms": []}

    def on_attention(_m, _a, out):
        store["attention"].append(out[0][0].float())

    def on_norm(module: nn.Module, args: tuple, out: Tensor):
        x = args[0][0].float()
        store["inverse_rms"].append(torch.rsqrt(x.pow(2).mean(-1) + module.variance_epsilon))
        store["mlp_input"].append(out[0].float())

    def on_down(_m, args):
        store["activation"].append(args[0][0].float())

    embed: dict[str, Tensor] = {}
    handles = [blocks[0].register_forward_pre_hook(lambda _m, args: embed.__setitem__("x", args[0][0].float()))]
    for block in blocks:
        handles += [
            block.self_attn.register_forward_hook(on_attention),
            block.post_attention_layernorm.register_forward_hook(on_norm),
            block.mlp.down_proj.register_forward_pre_hook(on_down),
        ]
    handles.append(blocks[-1].register_forward_hook(raise_stop))
    with hooks(handles):
        try:
            model.model(input_ids=input_ids, use_cache=False)
        except StopForward:
            pass
    return Stream(embed=embed["x"], **store)


def lane_row(model: PreTrainedModel, layer: int, neuron: int, lane: Lane) -> Float[Tensor, " d_model"]:
    mlp = decoder_layers(model)[layer].mlp
    return (mlp.gate_proj if lane == "gate" else mlp.up_proj).weight[neuron].float()


def read_row(model: PreTrainedModel, layer: int, neuron: int, lane: Lane) -> Float[Tensor, " d_model"]:
    """The neuron's input row with the norm gain folded in: its input is `read . resid_mid / rms`."""
    return lane_row(model, layer, neuron, lane) * decoder_layers(model)[layer].post_attention_layernorm.weight.float()


class InputSplit(BaseModel):
    """One lane of one neuron, summed over the chosen tokens. The parts add up to `actual`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    actual: float
    embed: float
    attention: list[float]
    neurons: list[Float[Tensor, " d_mlp"]]

    def mlp_total(self) -> float:
        return float(sum(n.sum() for n in self.neurons))

    def rebuilt(self) -> float:
        return self.embed + sum(self.attention) + self.mlp_total()


@torch.no_grad()
def split_input(model: PreTrainedModel, stream: Stream, layer: int, neuron: int, lane: Lane, tokens: Tensor) -> InputSplit:
    read = read_row(model, layer, neuron, lane)
    scale = stream.inverse_rms[layer][tokens]
    blocks = decoder_layers(model)

    def along(x: Tensor) -> float:
        return float(((x[tokens] @ read) * scale).sum())

    neurons = [
        ((stream.activation[k][tokens] * scale[:, None]).sum(0) * (blocks[k].mlp.down_proj.weight.float().T @ read)).cpu()
        for k in range(layer)
    ]
    return InputSplit(
        actual=float((stream.mlp_input[layer][tokens] @ lane_row(model, layer, neuron, lane)).sum()),
        embed=along(stream.embed),
        attention=[along(stream.attention[k]) for k in range(layer + 1)],
        neurons=neurons,
    )


@torch.no_grad()
def reader_shares(
    model: PreTrainedModel, stream: Stream, layer: int, neuron: int, reader_layer: int, tokens: Tensor
) -> dict[Lane, Float[Tensor, " d_mlp"]]:
    """For every neuron of `reader_layer`, the share of its gate and up input written by `neuron`, summed over tokens."""
    block = decoder_layers(model)[reader_layer]
    write = decoder_layers(model)[layer].mlp.down_proj.weight[:, neuron].float()
    amount = float((stream.activation[layer][tokens, neuron] * stream.inverse_rms[reader_layer][tokens]).sum())
    gain = block.post_attention_layernorm.weight.float()
    return {
        lane: (amount * ((proj.weight.float() * gain) @ write)).cpu()
        for lane, proj in (("gate", block.mlp.gate_proj), ("up", block.mlp.up_proj))
    }


def lane_inputs(model: PreTrainedModel, stream: Stream, layer: int, tokens: Tensor) -> dict[Lane, Float[Tensor, "token d_mlp"]]:
    block = decoder_layers(model)[layer]
    x = stream.mlp_input[layer][tokens]
    return {"gate": x @ block.mlp.gate_proj.weight.float().T, "up": x @ block.mlp.up_proj.weight.float().T}


def first_order_change(gate: Tensor, up: Tensor, d_gate: Tensor, d_up: Tensor) -> Tensor:
    """Change of `silu(gate) * up` for small input changes: `silu'(gate) up d_gate + silu(gate) d_up`."""
    s = torch.sigmoid(gate)
    silu_slope = s * (1 + gate * (1 - s))
    return silu_slope * up * d_gate + torch.nn.functional.silu(gate) * d_up


def hold_at_mean(layer_means: Float[Tensor, "layer d_mlp"], held: dict[int, Tensor], tokens: Tensor) -> list:
    """Pre-hooks for down_proj that set the `held` neurons of each layer to their mean at the chosen tokens."""

    def hook_for(layer: int) -> Callable:
        def hook(_module, args):
            a = args[0].clone()
            idx = held[layer].to(a.device)
            rows = tokens.to(a.device).nonzero().squeeze(1)
            a[0, rows[:, None], idx[None, :]] = layer_means[layer, held[layer]].to(a.device, a.dtype)
            return (a,)

        return hook

    return [(layer, hook_for(layer)) for layer in held if len(held[layer])]


@torch.no_grad()
def activation_with_held(
    model: PreTrainedModel,
    input_ids: Tensor,
    layer: int,
    neuron: int,
    layer_means: Tensor,
    held: dict[int, Tensor],
    tokens: Tensor,
) -> Float[Tensor, " token"]:
    """The neuron's activation at the chosen tokens, with the `held` neurons set to their mean."""
    blocks = decoder_layers(model)[: layer + 1]
    out: dict[str, Tensor] = {}
    handles = [blocks[k].mlp.down_proj.register_forward_pre_hook(h) for k, h in hold_at_mean(layer_means, held, tokens)]
    handles += [
        blocks[layer].mlp.down_proj.register_forward_pre_hook(lambda _m, a: out.__setitem__("a", a[0][0, :, neuron].float())),
        blocks[layer].register_forward_hook(raise_stop),
    ]
    with hooks(handles):
        try:
            model.model(input_ids=input_ids, use_cache=False)
        except StopForward:
            pass
    return out["a"][tokens.to(out["a"].device)].cpu()
