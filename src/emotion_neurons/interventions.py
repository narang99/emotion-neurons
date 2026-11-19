"""Edit MLP neuron activations during a forward pass, and read what the edit changes downstream.

- An edit acts on the input to `down_proj` (the activation `silu(gate) * up`) of one layer, at every token.
  - `add`: `a[..., n] += shift[n]`, e.g. remove an emotion's mean shift, or swap it for another emotion's.
  - `hold`: `a[..., n] = value[n]`, e.g. mean ablation.
- The readout of one pass:
  - the mean `resid_post` over a token range at chosen layers, for emotion scores;
  - the logits at the last token, for the model's own answer.
"""

from collections.abc import Callable
from typing import Literal

import torch
from jaxtyping import Float, Int
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from transformers import PreTrainedModel

from .models.qwen3 import decoder_layers, hooks

EditKind = Literal["add", "hold"]


class NeuronEdit(BaseModel):
    """One edit of some neurons of one MLP layer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    layer: int
    neurons: Int[Tensor, " neuron"]
    values: Float[Tensor, " neuron"]
    kind: EditKind

    def hook(self) -> Callable:
        def edit(_module, args):
            a = args[0].clone()
            idx = self.neurons.to(a.device)
            v = self.values.to(a.device, a.dtype)
            a[..., idx] = a[..., idx] + v if self.kind == "add" else v.expand_as(a[..., idx])
            return (a,)

        return edit


class Readout(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stream_means: Float[Tensor, "layer d_model"]
    """Mean `resid_post` over the token range, at each read layer (in the order asked)."""
    last_logits: Float[Tensor, " vocab"]


@torch.no_grad()
def run_with_edits(
    model: PreTrainedModel,
    input_ids: Int[Tensor, "1 token"],
    edits: list[NeuronEdit],
    read_layers: list[int],
    token_range: tuple[int, int],
) -> Readout:
    start, stop = token_range
    means: dict[int, Tensor] = {}

    def keep_mean(layer: int) -> Callable:
        def hook(_m, _a, out):
            h = out[0] if isinstance(out, tuple) else out
            means[layer] = h[0, start:stop].float().mean(0)

        return hook

    blocks = decoder_layers(model)
    handles = [blocks[e.layer].mlp.down_proj.register_forward_pre_hook(e.hook()) for e in edits]
    handles += [blocks[k].register_forward_hook(keep_mean(k)) for k in read_layers]
    with hooks(handles):
        logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].float()
    return Readout(stream_means=torch.stack([means[k] for k in read_layers]), last_logits=logits)


@torch.no_grad()
def generate_with_edits(
    model: PreTrainedModel, input_ids: Int[Tensor, "1 token"], edits: list[NeuronEdit], new_tokens: int
) -> Int[Tensor, " new"]:
    """Greedy continuation with the edits active at every position, prompt and new tokens alike."""
    blocks = decoder_layers(model)
    handles = [blocks[e.layer].mlp.down_proj.register_forward_pre_hook(e.hook()) for e in edits]
    with hooks(handles):
        out = model.generate(input_ids=input_ids, max_new_tokens=new_tokens, do_sample=False, use_cache=True)
    return out[0, input_ids.shape[1] :]


def emotion_scores(
    stream_means: Float[Tensor, "story d_model"], centre: Float[Tensor, " d_model"], units: Float[Tensor, "emotion d_model"]
) -> Float[Tensor, "story emotion"]:
    """Projection of each story's centred mean stream on each emotion's unit vector."""
    return (stream_means - centre) @ units.T
