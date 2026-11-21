"""The poles of every MLP neuron: its low and high quantiles (p3, p97) over general text.

1. Each corpus is streamed, tokenized and cut into windows (never across documents).
2. One pass to layer L reads `a_n = silu(gate_n) * up_n` for all neurons, at real tokens past `skip_first`.
3. Per neuron, quantile q is the k-th smallest value with k = int(q * n_tokens).

- The first tokens of each window are left out, because attention-sink effects dominate there.
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import PolesConfig
from .models import read_mlp
from .texts import corpus_windows

NEURON_CHUNK = 256
"""Neurons sorted at once on the GPU; 512 warned of memory on an L4."""


def collect_windows(cfg: PolesConfig, tokenizer: PreTrainedTokenizerBase) -> list[list[int]]:
    windows = []
    for corpus in cfg.corpora:
        start = len(windows)
        windows += [w for _, w in corpus_windows(corpus, tokenizer, cfg.window, cfg.min_window, cfg.seed)]
        print(f"[corpus] {corpus.name}: {len(windows) - start} windows, {sum(len(w) for w in windows[start:])} tokens")
    return windows


def padded_batch(windows: list[list[int]], pad_id: int) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
    longest = max(len(w) for w in windows)
    ids = torch.full((len(windows), longest), pad_id, dtype=torch.long)
    mask = torch.zeros((len(windows), longest), dtype=torch.long)
    for j, w in enumerate(windows):
        ids[j, : len(w)] = torch.tensor(w)
        mask[j, : len(w)] = 1
    return ids, mask


def corpus_activations(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, cfg: PolesConfig, layer: int
) -> Float[Tensor, "token neuron"]:
    """All neurons at every kept corpus token, in fp16 on the CPU (210k x 12288 is about 5 GB)."""
    windows = collect_windows(cfg, tokenizer)
    all_neurons = torch.arange(model.config.intermediate_size, device=model.device)
    rows = []
    for start in range(0, len(windows), cfg.batch_size):
        ids, mask = padded_batch(windows[start : start + cfg.batch_size], tokenizer.pad_token_id)
        _, a = read_mlp(model, layer, ids.to(model.device), mask.to(model.device), all_neurons)
        keep = mask.bool().to(a.device)
        keep[:, : cfg.skip_first] = False
        rows.append(a[keep].half().cpu())
    return torch.cat(rows)


def quantiles_per_neuron(
    acts: Float[Tensor, "token neuron"], quantiles: list[float], device: str | torch.device = "cpu"
) -> Float[Tensor, "quantile neuron"]:
    """The k-th smallest value per neuron, k = int(q * n_tokens), sorted in chunks of neurons."""
    ranks = torch.tensor([max(1, int(q * len(acts))) - 1 for q in quantiles], device=device)
    out = torch.empty(len(quantiles), acts.shape[1])
    for c in range(0, acts.shape[1], NEURON_CHUNK):
        chunk = acts[:, c : c + NEURON_CHUNK].to(device).float()
        out[:, c : c + NEURON_CHUNK] = chunk.sort(0).values[ranks].cpu()
    return out
