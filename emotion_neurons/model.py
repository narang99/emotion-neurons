"""Load Qwen3 and record the layer-L sites, stopping the forward pass right after layer L."""

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class StopForward(Exception):
    """Raised by the layer-L hook: nothing after layer L is needed, so it is never computed."""


def load_model(name: str, device: str):
    token = os.environ.get("HF_TOKEN") or None
    tok = AutoTokenizer.from_pretrained(name, token=token)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map=device, token=token)
    model.eval()
    return model, tok


def decoder_layers(model):
    return model.model.layers


@contextmanager
def capture(model, layer: int, stop: bool = True) -> Iterator[dict[str, torch.Tensor]]:
    """Fills a dict with resid_mid, mlp_act and resid_post of `layer`, each [batch, seq, dim], in bf16.

    Qwen3 decoder layer: h = h + attn(ln1(h)); h = h + mlp(ln2(h)).
    resid_mid is the input to ln2 (post_attention_layernorm), resid_post is the layer's output.
    """
    acts: dict[str, torch.Tensor] = {}
    block = decoder_layers(model)[layer]

    def on_mid(_mod, args):
        acts["resid_mid"] = args[0].detach()

    def on_act(_mod, args):
        acts["mlp_act"] = args[0].detach()

    def on_post(_mod, _args, out):
        acts["resid_post"] = (out[0] if isinstance(out, tuple) else out).detach()
        if stop:
            raise StopForward

    handles = [
        block.post_attention_layernorm.register_forward_pre_hook(on_mid),
        block.mlp.down_proj.register_forward_pre_hook(on_act),
        block.register_forward_hook(on_post),
    ]
    try:
        yield acts
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def run_to_layer(model, layer: int, enc) -> dict[str, torch.Tensor]:
    """One forward pass of the base model up to `layer`; returns the three hooked sites."""
    with capture(model, layer, stop=True) as acts:
        try:
            model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False)
        except StopForward:
            pass
    missing = {"resid_mid", "mlp_act", "resid_post"} - acts.keys()
    assert not missing, f"hooks did not fire: {missing}"
    return acts
