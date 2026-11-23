"""Load Qwen3 and read its decoder layers with hooks.

- A Qwen3 decoder layer is `h = h + attn(ln1(h)); h = h + mlp(ln2(h))`, with `mlp = down(silu(gate(x)) * up(x))`.
- Single-layer reads stop the forward pass right after that layer, so later layers are never computed.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from transformers import PreTrainedModel

from ..emotions import HOOKED_SITES, Site

SiteActivations = dict[Site, Float[Tensor, "batch seq dim"]]


class StopForward(Exception):
    """Raised by the hook on the last layer a read needs."""


def decoder_layers(model: PreTrainedModel) -> nn.ModuleList:
    return model.model.layers


@contextmanager
def hooks(handles: list[torch.utils.hooks.RemovableHandle]) -> Iterator[None]:
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def raise_stop(*_args) -> None:
    raise StopForward


@torch.no_grad()
def forward_until_stop(
    model: PreTrainedModel, input_ids: Int[Tensor, "batch seq"], attention_mask: Int[Tensor, "batch seq"]
) -> None:
    try:
        model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    except StopForward:
        pass


def store_input(store: dict, key: str) -> Callable:
    return lambda _module, args: store.__setitem__(key, args[0].detach())


def store_output(store: dict, key: str) -> Callable:
    return lambda _module, _args, out: store.__setitem__(key, (out[0] if isinstance(out, tuple) else out).detach())


def read_layer(
    model: PreTrainedModel, layer: int, input_ids: Int[Tensor, "batch seq"], attention_mask: Int[Tensor, "batch seq"]
) -> SiteActivations:
    """resid_mid, mlp_act and resid_post of one layer, in bf16."""
    acts: SiteActivations = {}
    block = decoder_layers(model)[layer]
    handles = [
        block.post_attention_layernorm.register_forward_pre_hook(store_input(acts, "resid_mid")),
        block.mlp.down_proj.register_forward_pre_hook(store_input(acts, "mlp_act")),
        block.register_forward_hook(store_output(acts, "resid_post")),
        block.register_forward_hook(raise_stop),
    ]
    with hooks(handles):
        forward_until_stop(model, input_ids, attention_mask)
    missing = set(HOOKED_SITES) - acts.keys()
    assert not missing, f"hooks did not fire: {missing}"
    return acts


def read_mlp(
    model: PreTrainedModel,
    layer: int,
    input_ids: Int[Tensor, "batch seq"],
    attention_mask: Int[Tensor, "batch seq"],
    neurons: Int[Tensor, " neuron"],
) -> tuple[Float[Tensor, "batch seq d_model"], Float[Tensor, "batch seq neuron"]]:
    """The MLP input (post_attention_layernorm output) and the chosen neurons' activations, in fp32."""
    acts: dict[str, Tensor] = {}
    block = decoder_layers(model)[layer]
    handles = [
        block.mlp.register_forward_pre_hook(store_input(acts, "mlp_in")),
        block.mlp.down_proj.register_forward_pre_hook(store_input(acts, "mlp_act")),
        block.register_forward_hook(raise_stop),
    ]
    with hooks(handles):
        forward_until_stop(model, input_ids, attention_mask)
    return acts["mlp_in"].float(), acts["mlp_act"][:, :, neurons].float()


def read_resid_pre(
    model: PreTrainedModel, layer: int, input_ids: Int[Tensor, "batch seq"], attention_mask: Int[Tensor, "batch seq"]
) -> Float[Tensor, "batch seq d_model"]:
    """The residual stream at the input of one layer, in bf16. The forward pass stops there."""
    acts: dict[str, Tensor] = {}

    def store_and_stop(_module, args, kwargs) -> None:
        acts["resid_pre"] = (args[0] if args else kwargs["hidden_states"]).detach()
        raise StopForward

    handle = decoder_layers(model)[layer].register_forward_pre_hook(store_and_stop, with_kwargs=True)
    with hooks([handle]):
        forward_until_stop(model, input_ids, attention_mask)
    return acts["resid_pre"]
