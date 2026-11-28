"""The poles of every MLP neuron: its low and high quantiles (p3, p97) over general text, and its mean.

1. Each corpus is streamed, tokenized and cut into windows (never across documents), once for all layers.
2. Per layer L, one pass up to L reads `a_n = silu(gate_n) * up_n` for all neurons, at real tokens past `skip_first`.
   - The input of the adapter's `mlp(model, L).down_proj` is `a`. The pass stops there.
   - One layer at a time: one layer's activations over 210k tokens are about 5 GB on the CPU.
3. Per neuron, quantile q is the k-th smallest value with k = int(q * n_tokens).

- The first tokens of each window are left out, because attention-sink effects dominate there.
- The mean is what mean ablation puts in place of a neuron's value (`ablation`).
  - `PolesConfig.mean` chooses it: over all kept tokens, or over the values strictly inside a band of quantiles.
"""

import torch
from jaxtyping import Float, Int
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .config import AllTokensMean, BandMean, PolesConfig, TextSample
from .models import ModelAdapter
from .models.qwen3 import StopForward, forward_until_stop, hooks
from .texts import corpus_windows

NEURON_CHUNK = 256
"""Neurons sorted at once on the GPU; 512 warned of memory on an L4."""


def collect_windows(cfg: TextSample, tokenizer: PreTrainedTokenizerBase) -> list[list[int]]:
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


def read_neurons(
    model: PreTrainedModel,
    adapter: ModelAdapter,
    layer: int,
    input_ids: Int[Tensor, "batch seq"],
    attention_mask: Int[Tensor, "batch seq"],
) -> Float[Tensor, "batch seq neuron"]:
    acts: dict[str, Tensor] = {}

    def store_and_stop(_module, args) -> None:
        acts["a"] = args[0].detach()
        raise StopForward

    handle = adapter.mlp(model, layer).down_proj.register_forward_pre_hook(store_and_stop)
    with hooks([handle]):
        forward_until_stop(model, input_ids, attention_mask)
    return acts["a"]


def layer_activations(
    model: PreTrainedModel,
    adapter: ModelAdapter,
    windows: list[list[int]],
    cfg: TextSample,
    layer: int,
    pad_id: int,
) -> Float[Tensor, "token neuron"]:
    """All neurons of one layer at every kept token, in fp16 on the CPU (210k x 12288 is about 5 GB)."""
    rows = []
    for start in range(0, len(windows), cfg.batch_size):
        ids, mask = padded_batch(windows[start : start + cfg.batch_size], pad_id)
        a = read_neurons(model, adapter, layer, ids.to(model.device), mask.to(model.device))
        keep = mask.bool().to(a.device)
        keep[:, : cfg.skip_first] = False
        rows.append(a[keep].half().cpu())
    return torch.cat(rows)


def mean_per_neuron(acts: Float[Tensor, "token neuron"], device: str | torch.device = "cpu") -> Float[Tensor, " neuron"]:
    out = torch.empty(acts.shape[1])
    for c in range(0, acts.shape[1], NEURON_CHUNK):
        out[c : c + NEURON_CHUNK] = acts[:, c : c + NEURON_CHUNK].to(device).float().mean(0).cpu()
    return out


def band_mean_per_neuron(
    acts: Float[Tensor, "token neuron"],
    low: Float[Tensor, " neuron"],
    high: Float[Tensor, " neuron"],
    device: str | torch.device = "cpu",
) -> Float[Tensor, " neuron"]:
    """The mean of the values strictly between `low` and `high`: the same test as `ablation.band_mask`."""
    out = torch.empty(acts.shape[1])
    for c in range(0, acts.shape[1], NEURON_CHUNK):
        chunk = acts[:, c : c + NEURON_CHUNK].to(device).float()
        inside = (chunk > low[c : c + NEURON_CHUNK].to(device)) & (chunk < high[c : c + NEURON_CHUNK].to(device))
        out[c : c + NEURON_CHUNK] = ((chunk * inside).sum(0) / inside.sum(0).clamp(min=1)).cpu()
    return out


def neuron_means(
    acts: Float[Tensor, "token neuron"],
    cfg: PolesConfig,
    quantiles: Float[Tensor, "quantile neuron"],
    device: str | torch.device = "cpu",
) -> Float[Tensor, " neuron"]:
    """`quantiles` holds one row per entry of `cfg.quantiles`, from `quantiles_per_neuron`."""
    match cfg.mean:
        case AllTokensMean():
            return mean_per_neuron(acts, device)
        case BandMean(low=low, high=high):
            row = cfg.quantiles.index
            return band_mean_per_neuron(acts, quantiles[row(low)], quantiles[row(high)], device)


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
