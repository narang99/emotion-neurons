"""Logit lens of residual directions: `W_U @ (gamma_final * x)`, as in tags exp/input_rows and exp/head_reads.

- Direct effect only: the layers after the one that writes x are ignored.
- The unembedding is cached as bf16 (`models.weights`); it is read in fp32 chunks, since bf16 matmul is slow on CPU.
"""

import torch
from jaxtyping import Float
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from .models.weights import cache_dir, load_tensor


def cached_bf16(model: str, name: str) -> Tensor:
    path = cache_dir() / f"{model.replace('/', '--')}.{name}.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(load_tensor(model, name).bfloat16(), path)
    return torch.load(path)


def unembedding(model: str, n_vocab: int) -> tuple[Tensor, Float[Tensor, " d_model"]]:
    """(lm_head [vocab, d_model] bf16, cut to the tokenizer's vocabulary; final norm gain, fp32)."""
    return cached_bf16(model, "lm_head.weight")[:n_vocab], cached_bf16(model, "model.norm.weight").float()


def top_token_ids(
    W_U: Tensor, gamma: Float[Tensor, " d_model"], directions: Float[Tensor, "d_model column"], k: int, chunk: int = 16384
) -> dict[str, Tensor]:
    """Top k vocabulary ids per column, pushed up ("up") and down ("down")."""
    scaled = gamma[:, None] * directions.float()
    best = {"up": ([], []), "down": ([], [])}
    for start in range(0, W_U.shape[0], chunk):
        logits = W_U[start : start + chunk].float() @ scaled
        for side, largest in (("up", True), ("down", False)):
            top = logits.topk(k, dim=0, largest=largest)
            best[side][0].append(top.values)
            best[side][1].append(top.indices + start)
    out = {}
    for side, largest in (("up", True), ("down", False)):
        values, ids = torch.cat(best[side][0]), torch.cat(best[side][1])
        out[side] = ids.gather(0, values.topk(k, dim=0, largest=largest).indices)
    return out


def decode_columns(tokenizer: PreTrainedTokenizerBase, ids: Tensor) -> list[list[str]]:
    return [[tokenizer.decode([i]) for i in ids[:, j].tolist()] for j in range(ids.shape[1])]
