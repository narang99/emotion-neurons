"""Reading a neuron's value vector (its W_down column) through the unembedding: a logit lens.

- logits = U (g * v): the final RMSNorm gain g, then the unembedding U.
  - RMSNorm only rescales a single vector, so the token ranking is exact for v alone.
- In earlier layers the stream is not yet in the unembedding's basis, so the read gets weaker with depth from the end.
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor

from .models import load_cached_tensor


def unembedding(model: str) -> tuple[Float[Tensor, " d_model"], Float[Tensor, "vocab d_model"]]:
    """(final norm gain, lm_head), cached as bf16."""
    return load_cached_tensor(model, "model.norm.weight"), load_cached_tensor(model, "lm_head.weight")


def value_logits(
    values: Float[Tensor, "d_model neuron"], gain: Float[Tensor, " d_model"], U: Float[Tensor, "vocab d_model"]
) -> Float[Tensor, "vocab neuron"]:
    return U @ (gain[:, None] * values)


def vocab_ranks(logits: Float[Tensor, "vocab neuron"]) -> Int[Tensor, "vocab neuron"]:
    """0 for each column's top token."""
    rank = torch.empty_like(logits, dtype=torch.int32)
    vocab = logits.shape[0]
    for j in range(logits.shape[1]):
        rank[logits[:, j].argsort(descending=True), j] = torch.arange(vocab, dtype=torch.int32)
    return rank
